"""
jego_module.py
JEGO (Joint Efficient Global Omnidirectional) scan module
extracted from JamMa for DROID-SLAM integration.

Usage:
    from modules.jego_module import JEGOModule
    self.jego = JEGOModule(feature_dim=128, depth=4)

    # fi, fj: [B*E, 128, H/8, W/8]
    fi_enriched, fj_enriched = self.jego(fi, fj)
"""

import torch
import torch.nn as nn
import math
from functools import partial

from mamba_ssm import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
    LayerNorm = nn.LayerNorm
except ImportError:
    RMSNorm, LayerNorm = None, None


# ──────────────────────────────────────────────
# 1. Mamba Block
# ──────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm,
                 fused_add_norm=False, residual_in_fp32=False):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm))

    def forward(self, desc, inference_params=None):
        hidden_states = self.norm(desc.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            desc = desc.to(torch.float32)
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return desc + hidden_states


def create_block(d_model, ssm_cfg=None, norm_epsilon=1e-5,
                 rms_norm=False, residual_in_fp32=False,
                 fused_add_norm=False, layer_idx=None,
                 device=None, dtype=None):
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(
        nn.LayerNorm if not rms_norm else RMSNorm,
        eps=norm_epsilon, **factory_kwargs
    )
    block = Block(d_model, mixer_cls, norm_cls=norm_cls,
                  fused_add_norm=fused_add_norm,
                  residual_in_fp32=residual_in_fp32)
    block.layer_idx = layer_idx
    return block


# ──────────────────────────────────────────────
# 2. JEGO Scan / Merge
# ──────────────────────────────────────────────

def scan_jego(desc0, desc1, step_size):
    desc_2w = torch.cat([desc0, desc1], 3)   # concat along W
    desc_2h = torch.cat([desc0, desc1], 2)   # concat along H
    _, _, org_h, org_2w = desc_2w.shape
    B, C, org_2h, org_w = desc_2h.shape

    H = org_h // step_size
    W = org_2w // step_size

    xs = desc_2w.new_empty((B, 4, C, H * W))
    xs[:, 0] = desc_2w[:, :, ::step_size, ::step_size].contiguous().view(B, C, -1)
    xs[:, 1] = desc_2h.transpose(2, 3)[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1)
    xs[:, 2] = desc_2w[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])
    xs[:, 3] = desc_2h.transpose(2, 3)[:, :, ::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])

    xs = xs.view(B, 4, C, -1).transpose(2, 3)
    return xs, org_h, org_w


def merge_jego(ys, ori_h, ori_w, step_size=2):
    B, K, C, L = ys.shape
    H = math.ceil(ori_h / step_size)
    W = math.ceil(ori_w / step_size)
    new_h, new_w = H * step_size, W * step_size

    y_2w = torch.zeros((B, C, new_h, 2 * new_w), device=ys.device, dtype=ys.dtype)
    y_2h = torch.zeros((B, C, 2 * new_h, new_w), device=ys.device, dtype=ys.dtype)

    y_2w[:, :, ::step_size, ::step_size]   = ys[:, 0].reshape(B, C, H, 2 * W)
    y_2h[:, :, 1::step_size, 1::step_size] = ys[:, 1].reshape(B, C, W, 2 * H).transpose(2, 3)
    y_2w[:, :, ::step_size, 1::step_size]  = ys[:, 2].flip([2]).reshape(B, C, H, 2 * W)
    y_2h[:, :, 1::step_size, ::step_size]  = ys[:, 3].flip([2]).reshape(B, C, W, 2 * H).transpose(2, 3)

    if ori_h != new_h or ori_w != new_w:
        y_2w = y_2w[:, :, :ori_h, :ori_w].contiguous()
        y_2h = y_2h[:, :, :ori_h, :ori_w].contiguous()

    desc0_2w, desc1_2w = torch.chunk(y_2w, 2, dim=3)
    desc0_2h, desc1_2h = torch.chunk(y_2h, 2, dim=2)
    return desc0_2w + desc0_2h, desc1_2w + desc1_2h


# ──────────────────────────────────────────────
# 3. Aggregator (GLU_3 - exact copy from JamMa)
# ──────────────────────────────────────────────

class Aggregator(nn.Module):
    """
    GLU_3 from JamMa src/jamma/utils/utils.py — exact copy.
    3x3 conv gating: GELU(W(x)) * V(x) → W2
    bias=False to match original.
    """
    def __init__(self, dim, mid_dim):
        super().__init__()
        self.W  = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.V  = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.W2 = nn.Conv2d(mid_dim, dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()
 
    def forward(self, feat):
        feat_act    = self.act(self.W(feat))   # gate path
        feat_linear = self.V(feat)             # linear path
        feat        = feat_act * feat_linear   # gating
        feat        = self.W2(feat)
        return feat

# ──────────────────────────────────────────────
# 4. Weight Initialization (from JamMa)
# ──────────────────────────────────────────────
 
def _init_weights(module, n_layer, initializer_range=0.02,
                  rescale_prenorm_residual=True, n_residuals_per_layer=1):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)
 
    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)

# ──────────────────────────────────────────────
# 5. JEGOModule — drop-in for DROID-SLAM
# ──────────────────────────────────────────────

class JEGOModule(nn.Module):
    """
    Standalone JEGO enrichment module for DROID-SLAM.

    Input : fi, fj  — [B, C, H, W]  (C=128 for DROID-SLAM)
    Output: fi_enriched, fj_enriched — same shape [B, C, H, W]
    """

    def __init__(self, feature_dim=128, depth=4,
                 rms_norm=True, residual_in_fp32=True, fused_add_norm=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.depth = depth

        self.layers = nn.ModuleList([
            create_block(
                feature_dim,
                rms_norm=rms_norm,
                residual_in_fp32=residual_in_fp32,
                fused_add_norm=fused_add_norm,
                layer_idx=i,
            )
            for i in range(depth)
        ])

        self.aggregator = Aggregator(feature_dim, feature_dim)

        # Mamba weight initialization (from JamMa)
        self.apply(
            partial(_init_weights, n_layer=depth)
        )

    def forward(self, fi, fj):
        """
        Args:
            fi: [B, C, H, W]  — feature map for frame i
            fj: [B, C, H, W]  — feature map for frame j
        Returns:
            fi_enriched, fj_enriched: same shape as input
        """
        # JEGO scan: interleave fi and fj into 4 directional sequences
        x, ori_h, ori_w = scan_jego(fi, fj, step_size=2)
        # x: [B, 4, seq_len, C]

        # Run 4 Mamba blocks (one per direction)
        assert len(self.layers) == 4, "depth must be 4 for JEGO (one per direction)"
        y0 = self.layers[0](x[:, 0])
        y1 = self.layers[1](x[:, 1])
        y2 = self.layers[2](x[:, 2])
        y3 = self.layers[3](x[:, 3])

        y = torch.stack([y0, y1, y2, y3], dim=1).transpose(2, 3)
        # y: [B, 4, C, seq_len]

        # Merge back to 2D
        fi_out, fj_out = merge_jego(y, ori_h, ori_w, step_size=2)
        # fi_out, fj_out: [B, C, H, W]

        # Aggregator (local 3x3 gated conv)
        desc = self.aggregator(torch.cat([fi_out, fj_out], dim=0))
        fi_enriched, fj_enriched = torch.chunk(desc, 2, dim=0)

        return fi + fi_enriched, fj + fj_enriched


# ──────────────────────────────────────────────
# 6. Quick sanity check
# ──────────────────────────────────────────────

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    module = JEGOModule(feature_dim=128, depth=4).to(device)

    # Simulate DROID-SLAM fnet output (B*E=24, 128, 60, 80)
    fi = torch.randn(24, 128, 60, 80, device=device)
    fj = torch.randn(24, 128, 60, 80, device=device)

    fi_out, fj_out = module(fi, fj)
    print(f"Input  shape: {fi.shape}")
    print(f"Output shape: {fi_out.shape}")
    assert fi_out.shape == fi.shape, "Shape mismatch!"
    print("jego_module sanity check PASSED")