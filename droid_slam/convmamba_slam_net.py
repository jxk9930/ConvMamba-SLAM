"""
convmamba_slam_net.py
=====================
ConvMamba-SLAM: DROID-SLAM with ConvMamba update operator.

Replaces DROID-SLAM's ConvGRU with ConvMamba (omnidirectional Mamba SSM)
to achieve global receptive field per iteration.

Changes from DROID-SLAM (DroidNet):
    1. fnet/cnet → ConvNeXt V2 encoder (shared backbone, separate heads)
    2. ConvGRU → ConvMamba in UpdateModule

Changes from JamMa-SLAM (JammaSlamNet):
    1. JEGO enrichment REMOVED (no pre-CorrBlock enrichment)
    2. Uncertainty module REMOVED
    3. ConvGRU → ConvMamba

Unchanged from both:
    - CorrBlock (4D correlation volume + pyramid)
    - DBA (Gauss-Newton optimization)
    - Forward loop structure (iterative update)
    - GraphAgg (pooling over source views)

Architecture:
    Image_i, Image_j
        ↓
    ConvNeXt V2 → fmaps (128ch), net (128ch tanh), inp (128ch relu)
        ↓
    Raw fmaps → 4D Correlation Volume → Lookup
        ↓
    [ConvMamba(h, inp, corr_enc, flow_enc) × N iterations] → δ flow, weight
        ↓
    DBA → pose ΔG, depth Δd
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from modules.corr import CorrBlock
from modules.conv_mamba import ConvMamba
from modules.clipping import GradientClip

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
# ConvNeXt V2 Encoder
# ──────────────────────────────────────────────

class ConvNeXtEncoder(nn.Module):
    """
    ConvNeXt V2 based encoder replacing DROID's BasicEncoder.

    Outputs:
        coarse_feats: [B*N, 128, H/8, W/8] — for correlation volume
        net: [B*N, 128, H/8, W/8] — hidden state init (tanh)
        inp: [B*N, 128, H/8, W/8] — per-iteration context (relu)
    """

    def __init__(self, coarse_dim=128, context_dim=256):
        super().__init__()
        self.coarse_dim = coarse_dim
        self.context_dim = context_dim

        # ConvNeXt V2 backbone (first 2 stages)
        self.backbone = build_convnext_backbone(pretrained=True)

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
            coarse: [B, N, 128, H/8, W/8]
            net:    [B, N, 128, H/8, W/8] (tanh)
            inp:    [B, N, 128, H/8, W/8] (relu)
        """
        b, n, c, h, w = images.shape

        images = images[:, :, [2, 1, 0]] / 255.0
        mean = torch.as_tensor([0.485, 0.456, 0.406], device=images.device)
        std = torch.as_tensor([0.229, 0.224, 0.225], device=images.device)
        images = images.sub_(mean[:, None, None]).div_(std[:, None, None])

        x = images.view(b * n, c, h, w)
        backbone_feat = self.backbone(x)

        coarse = self.coarse_proj(backbone_feat)
        context = self.context_proj(backbone_feat)

        net, inp = context.split([128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        _, _, h_f, w_f = coarse.shape
        coarse = coarse.view(b, n, self.coarse_dim, h_f, w_f)
        net = net.view(b, n, 128, h_f, w_f)
        inp = inp.view(b, n, 128, h_f, w_f)

        return coarse, net, inp


# ──────────────────────────────────────────────
# Graph Aggregation (same as DROID)
# ──────────────────────────────────────────────

class GraphAgg(nn.Module):
    def __init__(self):
        super().__init__()
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
# Update Module (ConvMamba replaces ConvGRU)
# ──────────────────────────────────────────────

class UpdateModule(nn.Module):
    """
    DROID-SLAM update operator with ConvMamba instead of ConvGRU.

    Input encoders (corr_encoder, flow_encoder) are identical to DROID.
    Output heads (delta, weight) are identical to DROID.
    Only the core recurrent module changes: ConvGRU → ConvMamba.
    """

    def __init__(self):
        super().__init__()
        cor_planes = 4 * (2 * 3 + 1) ** 2  # 4 levels × 7×7 = 196

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

        # ★ Core change: ConvGRU → ConvMamba ★
        # i_planes = inp(128) + corr_enc(128) + flow_enc(64) = 320
        self.mamba = ConvMamba(h_planes=128, i_planes=128 + 128 + 64)

        self.agg = GraphAgg()

    def forward(self, net, inp, corr, flow=None, ii=None, jj=None):
        """
        Update operator forward pass.
        Same interface as DROID's UpdateModule.

        Args:
            net:  [B, E, 128, H, W] — hidden state
            inp:  [B, E, 128, H, W] — context injection
            corr: [B, E, 196, H, W] — correlation lookup
            flow: [B, E, 4, H, W]   — motion features
            ii:   [E] — source frame indices
            jj:   [E] — target frame indices

        Returns:
            net:    [B, E, 128, H, W] — updated hidden state
            delta:  [B, E, H, W, 2]   — flow revision
            weight: [B, E, H, W, 2]   — confidence
            eta:    [B, K, H, W]       — damping (if ii provided)
            upmask: [B, K, 8*8*9, H, W] — upsample mask (if ii provided)
        """
        batch, num, ch, ht, wd = net.shape

        if flow is None:
            flow = torch.zeros(batch, num, 4, ht, wd, device=net.device)

        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch * num, -1, ht, wd)
        inp = inp.view(batch * num, -1, ht, wd)
        corr = corr.view(batch * num, -1, ht, wd)
        flow = flow.view(batch * num, -1, ht, wd)

        corr = self.corr_encoder(corr)    # [B*E, 128, H, W]
        flow = self.flow_encoder(flow)    # [B*E, 64, H, W]

        # ★ ConvMamba instead of ConvGRU ★
        net = self.mamba(net, inp, corr, flow)

        # Output heads (identical to DROID)
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
# ConvMamba-SLAM Network
# ══════════════════════════════════════════════

class ConvMambaSlamNet(nn.Module):
    """
    ConvMamba-SLAM: DROID-SLAM with omnidirectional Mamba update operator.

    Compared to DroidNet:
        - fnet/cnet → ConvNeXt V2 encoder (shared backbone)
        - ConvGRU → ConvMamba in update operator

    Compared to JammaSlamNet:
        - JEGO enrichment REMOVED
        - Uncertainty module REMOVED
        - ConvGRU → ConvMamba
    """

    def __init__(self):
        super().__init__()

        # Encoder (ConvNeXt V2, replaces fnet + cnet)
        self.encoder = ConvNeXtEncoder(coarse_dim=128, context_dim=256)

        # MotionFilter compatibility wrappers
        self.fnet = self._fnet_wrapper
        self.cnet = self._cnet_wrapper

        # Update operator (ConvMamba)
        self.update = UpdateModule()

    # ---- MotionFilter wrappers ----
    def _fnet_wrapper(self, x):
        """x: [1, 1, 3, H, W]"""
        fmaps, _, _ = self.encoder(x)
        return fmaps

    def _cnet_wrapper(self, x):
        """x: [1, 1, 3, H, W]"""
        _, net, inp = self.encoder(x)
        return torch.cat([net, inp], dim=2)

    def extract_features(self, images):
        """
        Extract features from all frames.

        Args:
            images: [B, N, 3, H, W]
        Returns:
            fmaps: [B, N, 128, H/8, W/8]
            net:   [B, N, 128, H/8, W/8] (tanh)
            inp:   [B, N, 128, H/8, W/8] (relu)
        """
        return self.encoder(images)

    def forward(self, Gs, images, disps, intrinsics, graph=None,
                num_steps=12, fixedp=2):
        """
        Full forward pass (training).

        Same structure as DroidNet.forward() but with:
            - ConvNeXt features instead of BasicEncoder
            - ConvMamba instead of ConvGRU
            - No JEGO enrichment (raw features → CorrBlock)
            - No uncertainty modulation

        Args:
            Gs: [B, N, 7] — initial pose estimates (SE3)
            images: [B, N, 3, H, W] — input images
            disps: [B, N, H/8, W/8] — initial disparity estimates
            intrinsics: [B, N, 4] — camera intrinsics
            graph: OrderedDict — frame graph
            num_steps: int — number of update iterations
            fixedp: int — number of fixed poses
        """

        u = keyframe_indicies(graph)
        ii, jj, kk = graph_to_edge_list(graph)

        ii = ii.to(device=images.device, dtype=torch.long)
        jj = jj.to(device=images.device, dtype=torch.long)

        # 1. Feature extraction
        fmaps, net, inp = self.extract_features(images)

        # 2. Correlation volume (raw features, no JEGO)
        corr_fn = CorrBlock(
            fmaps[:, ii],
            fmaps[:, jj],
            num_levels=4, radius=3,
        )

        # 3. Context features for update operator (per edge)
        net = net[:, ii]   # [B, E, 128, H, W]
        inp = inp[:, ii]   # [B, E, 128, H, W]

        # 4. Iterative update loop
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

            # Correlation lookup
            corr = corr_fn(coords1)

            # Motion features
            resd = target - coords1
            flow = coords1 - coords0
            motion = torch.cat([flow, resd], dim=-1)
            motion = motion.permute(0, 1, 4, 2, 3).clamp(-64.0, 64.0)

            # ★ ConvMamba update ★
            net, delta, weight, eta, upmask = self.update(
                net, inp, corr, motion, ii, jj
            )

            # Update target
            target = coords1 + delta

            # Dense Bundle Adjustment
            for i in range(2):
                Gs, disps = BA(
                    target, weight, eta, Gs, disps, intrinsics, ii, jj, fixedp=fixedp
                )

            # Reproject
            coords1, valid_mask = pops.projective_transform(
                Gs, disps, intrinsics, ii, jj
            )
            residual = target - coords1

            Gs_list.append(Gs)
            disp_list.append(upsample_disp(disps, upmask))
            residual_list.append(valid_mask * residual)

        return Gs_list, disp_list, residual_list
