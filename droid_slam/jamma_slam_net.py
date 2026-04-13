"""
jamma_slam_net.py
=================
JamMa-SLAM: End-to-End SLAM with JamMa-enriched features.

Replaces DROID-SLAM's fnet/cnet with:
  - ConvNeXt V2 encoder (pretrained)
  - JEGO enrichment (Mamba-based cross-view interaction)
  - Sc-based uncertainty for DBA weight modulation

Keeps DROID-SLAM's:
  - CorrBlock (4D correlation volume)
  - ConvGRU Update Operator
  - Dense Bundle Adjustment (DBA)
  - Projective geometry / frame graph

Architecture:
  Image_i, Image_j
      ↓
  ConvNeXt V2 → Fc_coarse (H/8, 128ch), context (H/8, 256ch → net 128 + inp 128)
      ↓
  JEGO (scan → Mamba ×4 → merge → aggregator) → F̂c (enriched, 128ch)
      ↓
  ├── F̂c → 4D Correlation Volume → Lookup → ConvGRU → flow revision + wij
  └── F̂c → Sc → P → uncertainty map → weight modulation: w'ij = wij * (1 - α·u)
      ↓
  DBA → pose ΔG, depth Δd
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from functools import partial

from modules.corr import CorrBlock
from modules.gru import ConvGRU
from modules.clipping import GradientClip
from modules.jego_module import JEGOModule

from lietorch import SE3
from geom.ba import BA
import geom.projective_ops as pops
from geom.graph_utils import graph_to_edge_list, keyframe_indicies

from torch_scatter import scatter_mean
from jamma_encoder import build_convnext_backbone


# ──────────────────────────────────────────────
# Utility functions (same as DROID)
# ──────────────────────────────────────────────

def cvx_upsample(data, mask):
    """Upsample pixel-wise transformation field."""
    batch, ht, wd, dim = data.shape
    data = data.permute(0, 3, 1, 2)
    mask = mask.view(batch, 1, 9, 8, 8, ht, wd)
    mask = torch.softmax(mask, dim=2)

    up_data = F.unfold(data, [3, 3], padding=1)
    up_data = up_data.view(batch, dim, 9, 1, 1, ht, wd)

    up_data = torch.sum(mask * up_data, dim=2)
    up_data = up_data.permute(0, 4, 2, 5, 3, 1)
    up_data = up_data.reshape(batch, 8 * ht, 8 * wd, dim)
    return up_data


def upsample_disp(disp, mask):
    batch, num, ht, wd = disp.shape
    disp = disp.view(batch * num, ht, wd, 1)
    mask = mask.view(batch * num, -1, ht, wd)
    return cvx_upsample(disp, mask).view(batch, num, 8 * ht, 8 * wd)


# ──────────────────────────────────────────────
# ConvNeXt V2 Encoder (placeholder — implemented in jamma_encoder.py)
# ──────────────────────────────────────────────

class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt V2 based encoder replacing DROID's BasicEncoder.
    
    Outputs:
        coarse_feats: [B*N, 128, H/8, W/8]  — for JEGO enrichment + correlation
        context_feats: [B*N, 256, H/8, W/8] — split into net(128) + inp(128) for GRU
    
    Uses first two stages of ConvNeXt V2-Nano (0.65M params) with projection heads.
    Full implementation in jamma_encoder.py.
    """

    def __init__(self, coarse_dim=128, context_dim=256):
        super().__init__()
        self.coarse_dim = coarse_dim
        self.context_dim = context_dim

        # ---- ConvNeXt V2 backbone (first 2 stages) ----
        # Stage 0: stride 4, channels 80
        # Stage 1: stride 8, channels 160
        # Will be loaded from pretrained weights
        self.backbone = build_convnext_backbone(pretrained=True)

        # ---- Projection heads ----
        # backbone output channels (Nano stage 1 = 160) → target dimensions
        backbone_ch = 160  # ConvNeXt V2 Nano stage 1 output channels
        
        self.coarse_proj = nn.Sequential(
            nn.Conv2d(backbone_ch, coarse_dim, 1, bias=False),
            nn.InstanceNorm2d(coarse_dim),
        )

        self.context_proj = nn.Sequential(
            nn.Conv2d(backbone_ch, context_dim, 1, bias=False),
        )

    def forward(self, images):
        """
        Args:
            images: [B, N, 3, H, W] — raw images (0-255, BGR)
        Returns:
            coarse_feats: [B, N, 128, H/8, W/8]
            net: [B, N, 128, H/8, W/8]  (tanh activated)
            inp: [B, N, 128, H/8, W/8]  (relu activated)
        """
        b, n, c, h, w = images.shape

        # Normalize images (ImageNet stats, BGR → RGB)
        images = images[:, :, [2, 1, 0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        # Flatten batch and frames
        x = images.view(b * n, c, h, w)

        # ConvNeXt backbone → H/8 resolution features
        backbone_feat = self.backbone(x)  # [B*N, 80, H/8, W/8]

        # Projection heads
        coarse = self.coarse_proj(backbone_feat)  # [B*N, 128, H/8, W/8]
        context = self.context_proj(backbone_feat)  # [B*N, 256, H/8, W/8]

        # Split context into net + inp (same as DROID's cnet)
        net, inp = context.split([128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        # Reshape to [B, N, C, H/8, W/8]
        _, c_f, h_f, w_f = coarse.shape
        coarse = coarse.view(b, n, self.coarse_dim, h_f, w_f)
        net = net.view(b, n, 128, h_f, w_f)
        inp = inp.view(b, n, 128, h_f, w_f)

        return coarse, net, inp


# ──────────────────────────────────────────────
# Uncertainty Module (Sc → P → uncertainty map)
# ──────────────────────────────────────────────

class UncertaintyModule(nn.Module):
    """
    Extracts pixel-level uncertainty from JamMa's coarse similarity matrix.
    
    F̂c_i, F̂c_j → Sc = (1/τ) * <F̂c_i, F̂c_j> → Softmax → P → uncertainty
    
    The uncertainty map modulates DBA confidence weights:
        w'_ij = w_ij * (1 - α * u_i)
    
    where u_i = 1 - max_j(P_A→B[i, j]) per spatial location.
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        # Learnable scaling factor for uncertainty influence
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def compute_uncertainty_map(self, fi, fj):
        """
        Compute per-pixel uncertainty from enriched feature pair.
        
        Args:
            fi: [B, C, H, W] — enriched coarse features for frame i
            fj: [B, C, H, W] — enriched coarse features for frame j
        Returns:
            uncertainty: [B, 1, H, W] — uncertainty map (0=certain, 1=uncertain)
        """
        B, C, H, W = fi.shape

        # Flatten spatial dims: [B, C, H*W]
        fi_flat = fi.view(B, C, -1)  # [B, C, N] where N = H*W
        fj_flat = fj.view(B, C, -1)

        # L2 normalize along channel dim
        fi_norm = F.normalize(fi_flat, dim=1)
        fj_norm = F.normalize(fj_flat, dim=1)

        # Coarse similarity matrix: Sc = (1/τ) * fi^T · fj → [B, N, N]
        Sc = torch.bmm(fi_norm.transpose(1, 2), fj_norm) / self.temperature

        # Row softmax → P_A→B: [B, N, N]
        P = F.softmax(Sc, dim=-1)

        # Confidence = max probability per row: [B, N]
        confidence, _ = P.max(dim=-1)

        # Uncertainty = 1 - confidence: [B, N]
        uncertainty = 1.0 - confidence

        # Reshape to spatial map: [B, 1, H, W]
        uncertainty = uncertainty.view(B, 1, H, W)

        return uncertainty

    def modulate_weight(self, weight, uncertainty_maps):
        alpha = torch.sigmoid(self.alpha)
        # weight: [B, E, H, W, 2]
        # uncertainty_maps: [B, E, 1, H, W] → [B, E, H, W, 1] for broadcasting
        u = uncertainty_maps.permute(0, 1, 3, 4, 2)
        w_modulated = weight * (1.0 - alpha * u)
        return w_modulated


# ──────────────────────────────────────────────
# Graph Aggregation (same as DROID)
# ──────────────────────────────────────────────

class GraphAgg(nn.Module):
    def __init__(self):
        super(GraphAgg, self).__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.eta = nn.Sequential(
            nn.Conv2d(128, 1, 3, padding=1),
            GradientClip(),
            nn.Softplus(),
        )

        self.upmask = nn.Sequential(
            nn.Conv2d(128, 8 * 8 * 9, 1, padding=0),
        )

    def forward(self, net, ii):
        batch, num, ch, ht, wd = net.shape
        net = net.view(batch * num, ch, ht, wd)

        _, ix = torch.unique(ii, return_inverse=True)
        net = self.relu(self.conv1(net))

        net = net.view(batch, num, 128, ht, wd)
        net = scatter_mean(net, ix, dim=1)
        net = net.view(-1, 128, ht, wd)

        net = self.relu(self.conv2(net))

        eta = self.eta(net).view(batch, -1, ht, wd)
        upmask = self.upmask(net).view(batch, -1, 8 * 8 * 9, ht, wd)

        return 0.01 * eta, upmask


# ──────────────────────────────────────────────
# Update Module (same as DROID — ConvGRU core)
# ──────────────────────────────────────────────

class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2 * 3 + 1) ** 2  # 4 levels, radius 3

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(4, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
            nn.Sigmoid(),
        )

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
            GradientClip(),
        )

        self.gru = ConvGRU(128, 128 + 128 + 64)
        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """RaftSLAM update operator."""
        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch * num, -1, ht, wd)
        inp = inp.view(batch * num, -1, ht, wd)
        corr = corr.view(batch * num, -1, ht, wd)
        flow = flow.view(batch * num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0, 1, 3, 4, 2)[..., :2].contiguous()
        weight = weight.permute(0, 1, 3, 4, 2)[..., :2].contiguous()

        net = net.view(*output_dim)

        if ii is not None:
            eta, upmask = self.agg(net, ii.to(net.device))
            return net, delta, weight, eta, upmask
        else:
            return net, delta, weight


# ══════════════════════════════════════════════
# JamMa-SLAM Network (main module)
# ══════════════════════════════════════════════

class JammaSlamNet(nn.Module):
    """
    JamMa-SLAM: DROID-SLAM with JamMa-enriched features.
    
    Changes from DroidNet:
        1. fnet/cnet → ConvNeXt V2 encoder (shared backbone, separate heads)
        2. JEGO enrichment on coarse features (per edge pair)
        3. Sc-based uncertainty → DBA weight modulation
    
    Unchanged from DroidNet:
        - CorrBlock (4D correlation volume + pyramid)
        - ConvGRU update operator
        - DBA (Gauss-Newton optimization)
        - Forward loop structure (iterative update)
    """

    def __init__(self, uncertainty_temperature=0.1, use_uncertainty=True):
        super(JammaSlamNet, self).__init__()
        self.use_uncertainty = use_uncertainty

        # ---- Encoder (replaces fnet + cnet) ----
        self.encoder = ConvNeXtEncoder(coarse_dim=128, context_dim=256)

        # MotionFilter 
        self.fnet = self._fnet_wrapper
        self.cnet = self._cnet_wrapper

        # ---- JEGO enrichment (cross-view Mamba interaction) ----
        self.jego = JEGOModule(feature_dim=128, depth=4)

        # ---- Uncertainty module (optional) ----
        if self.use_uncertainty:
            self.uncertainty = UncertaintyModule(temperature=uncertainty_temperature)

        # ---- Update operator (same as DROID) ----
        self.update = UpdateModule()

    # ---- MotionFilter(Internal Wrappers) ----
    def _fnet_wrapper(self, x):
        """ x : [1, 1, 3, H, W] """
        fmaps, _, _ = self.encoder(x)
        return fmaps # [1, 1, 128, H/8, W/8] 

    def _cnet_wrapper(self, x):
        """ x : [1, 1, 3, H, W] """
        _, net, inp = self.encoder(x)        
        return torch.cat([net, inp], dim=2)

    def extract_features(self, images):
        """
        Extract features from all frames.
        
        Args:
            images: [B, N, 3, H, W]
        Returns:
            fmaps: [B, N, 128, H/8, W/8]  — coarse features (pre-JEGO)
            net:   [B, N, 128, H/8, W/8]  — GRU hidden state init
            inp:   [B, N, 128, H/8, W/8]  — GRU context injection
        """
        return self.encoder(images)

    def jego_enrich(self, fmaps, ii, jj):
        """
        Apply JEGO enrichment to feature pairs on graph edges.
        
        Args:
            fmaps: [B, N, 128, H/8, W/8] — per-frame coarse features
            ii: [E] — source frame indices
            jj: [E] — target frame indices
        Returns:
            fmaps_i_enriched: [B, E, 128, H/8, W/8] — enriched features for frame i
            fmaps_j_enriched: [B, E, 128, H/8, W/8] — enriched features for frame j
        """
        B = fmaps.shape[0]
        fi = fmaps[:, ii]  # [B, E, 128, H, W]
        fj = fmaps[:, jj]  # [B, E, 128, H, W]

        E = ii.shape[0]
        _, _, C, H, W = fi.shape

        # Flatten batch and edges for JEGO: [B*E, C, H, W]
        fi_flat = fi.view(B * E, C, H, W)
        fj_flat = fj.view(B * E, C, H, W)

        # JEGO enrichment (scan → Mamba → merge → aggregator)
        fi_enriched, fj_enriched = self.jego(fi_flat, fj_flat)

        # Reshape back: [B, E, C, H, W]
        fi_enriched = fi_enriched.view(B, E, C, H, W)
        fj_enriched = fj_enriched.view(B, E, C, H, W)

        return fi_enriched, fj_enriched

    def compute_edge_uncertainty(self, fi_enriched, fj_enriched):
        """
        Compute uncertainty maps for all edges.
        
        Args:
            fi_enriched: [B, E, 128, H, W]
            fj_enriched: [B, E, 128, H, W]
        Returns:
            uncertainty_maps: [B, E, 1, H, W]
        """
        B, E, C, H, W = fi_enriched.shape

        # Flatten: [B*E, C, H, W]
        fi_flat = fi_enriched.view(B * E, C, H, W)
        fj_flat = fj_enriched.view(B * E, C, H, W)

        # Compute uncertainty: [B*E, 1, H, W]
        u = self.uncertainty.compute_uncertainty_map(fi_flat, fj_flat)

        return u.view(B, E, 1, H, W)

    def forward(self, Gs, images, disps, intrinsics, graph=None, num_steps=12, fixedp=2):
        """
        Full forward pass (training).
        
        Same structure as DroidNet.forward() but with:
        - ConvNeXt features instead of BasicEncoder
        - JEGO enrichment on edge pairs
        - Uncertainty-modulated DBA weights
        
        Args:
            Gs: [B, N, 7] — initial pose estimates (SE3)
            images: [B, N, 3, H, W] — input images
            disps: [B, N, H/8, W/8] — initial disparity estimates
            intrinsics: [B, N, 4] — camera intrinsics
            graph: OrderedDict — frame graph (node → [neighbors])
            num_steps: int — number of GRU iterations
            fixedp: int — number of fixed poses (gauge freedom)
        Returns:
            Gs_list: list of SE3 pose estimates per iteration
            disp_list: list of upsampled disparity maps per iteration
            residual_list: list of flow residuals per iteration
        """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        # ──────────────────────────────────────
        # 1. Feature extraction (ConvNeXt)
        # ──────────────────────────────────────
        fmaps, net, inp = self.extract_features(images)

        # ──────────────────────────────────────
        # 2. JEGO enrichment (per edge pair)
        # ──────────────────────────────────────
        fi_enriched, fj_enriched = self.jego_enrich(fmaps, ii, jj)

        # ──────────────────────────────────────
        # 3. Correlation volume (DROID-style, with enriched features)
        # ──────────────────────────────────────
        corr_fn = CorrBlock(
            fi_enriched,  # [E, 128, H, W] (B=1 during training)
            fj_enriched,
            num_levels=4, radius=3,
        )

        # ──────────────────────────────────────
        # 4. Uncertainty maps (from enriched features)
        # ──────────────────────────────────────
        if self.use_uncertainty:
            uncertainty_maps = self.compute_edge_uncertainty(fi_enriched, fj_enriched)
            # [B, E, 1, H, W]

        # ──────────────────────────────────────
        # 5. Context features for GRU (per edge)
        # ──────────────────────────────────────
        net, inp = net[:, ii], inp[:, ii]
        # [B, E, 128, H, W]

        # ──────────────────────────────────────
        # 6. Iterative update loop (same structure as DROID)
        # ──────────────────────────────────────
        ht, wd = images.shape[-2:]
        coords0 = pops.coords_grid(ht // 8, wd // 8, device=images.device)

        coords1, _ = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
        target = coords1.clone()

        Gs_list, disp_list, residual_list = [], [], []

        for step in range(num_steps):
            Gs = Gs.detach()
            disps = disps.detach()
            coords1 = coords1.detach()
            target = target.detach()

            # -- Correlation lookup at current correspondence --
            corr = corr_fn(coords1)

            # -- Motion features --
            resd = target - coords1
            flow = coords1 - coords0
            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            # -- GRU update --
            net, delta, weight, eta, upmask = self.update(
                net, inp, corr, motion, ii, jj
            )

            # -- Uncertainty-modulated weight --
            if self.use_uncertainty:
                weight = self.uncertainty.modulate_weight(weight, uncertainty_maps)

            # -- Update target correspondence --
            target = coords1 + delta

            # -- Dense Bundle Adjustment --
            for i in range(2):
                Gs, disps = BA(
                    target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=fixedp
                )

            # -- Reproject with updated pose/depth --
            coords1, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj)
            residual = target - coords1

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)

        return Gs_list, disp_list, residual_list
