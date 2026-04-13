"""
conv_mamba.py
=============
ConvMamba: drop-in replacement for ConvGRU in DROID-SLAM's update operator.

Instead of 3x3 ConvGRU with crude global context (spatial mean),
this uses omnidirectional skip-scan + 4 independent Mamba SSM blocks
to achieve global receptive field in a single iteration.

Interface (identical to ConvGRU):
    forward(net, *inputs) → net_new
    where net = hidden state [B, 128, H, W]
          *inputs = (inp, corr_enc, flow_enc) concatenated

Usage:
    from modules.conv_mamba import ConvMamba

    # Replace: self.gru = ConvGRU(128, 128+128+64)
    # With:    self.mamba = ConvMamba(h_planes=128, i_planes=320)
"""

import torch
import torch.nn as nn
from functools import partial

from mamba_ssm import Mamba
try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm
except ImportError:
    RMSNorm = None

from modules.omni_scan import omni_scan, omni_merge, Aggregator


# ──────────────────────────────────────────────
# Mamba Block (pre-norm residual, from JamMa)
# ──────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Single Mamba block with pre-norm and residual connection."""

    def __init__(self, dim, layer_idx=None, residual_in_fp32=True):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = nn.LayerNorm(dim)
        self.mixer = Mamba(dim, layer_idx=layer_idx)

    def forward(self, x):
        """
        Args:
            x: [B, L, C] — 1D sequence
        Returns:
            x_out: [B, L, C] — same shape, with residual
        """
        residual = x
        hidden = self.norm(x.to(dtype=self.norm.weight.dtype))
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        hidden = self.mixer(hidden)
        return residual + hidden


# ──────────────────────────────────────────────
# Weight initialization (from JamMa / Mamba)
# ──────────────────────────────────────────────

def _init_weights(module, n_layer, initializer_range=0.02, rescale_prenorm_residual=True):
    """Mamba-specific weight initialization."""
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                nn.init.kaiming_uniform_(p, a=2.0**0.5)
                with torch.no_grad():
                    p /= (2 * n_layer) ** 0.5


# ──────────────────────────────────────────────
# ConvMamba
# ──────────────────────────────────────────────

class ConvMamba(nn.Module):
    """
    ConvMamba: omnidirectional Mamba-based update operator.

    Replaces ConvGRU by:
      1. Projecting concatenated inputs (h + inp + corr + flow) to D=128
      2. Omnidirectional skip-scan → 4 directional 1D sequences
      3. 4 independent Mamba blocks (one per direction)
      4. Merge back to 2D + residual from h
      5. Aggregator (GLU 3x3) for local mixing

    Args:
        h_planes: int — hidden state channels (default 128)
        i_planes: int — total input channels: inp(128) + corr(128) + flow(64) = 320
        d_model:  int — Mamba internal dimension (default: same as h_planes)
        step_size: int — skip scan step (default 2)
    """

    def __init__(self, h_planes=128, i_planes=320, d_model=None, step_size=2):
        super().__init__()

        if d_model is None:
            d_model = h_planes

        self.h_planes = h_planes
        self.d_model = d_model
        self.step_size = step_size

        # Input projection: cat(h, inp, corr, flow) → d_model
        total_in = h_planes + i_planes
        self.input_proj = nn.Sequential(
            nn.Conv2d(total_in, d_model, 1, bias=False),
            nn.InstanceNorm2d(d_model),
            nn.GELU(),
        )

        # 4 independent Mamba blocks (one per scan direction)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, layer_idx=i)
            for i in range(4)
        ])

        # Aggregator: post-merge local mixing
        self.aggregator = Aggregator(d_model)

        # Output projection (only needed if d_model != h_planes)
        if d_model != h_planes:
            self.output_proj = nn.Conv2d(d_model, h_planes, 1, bias=False)
        else:
            self.output_proj = nn.Identity()

        # Initialize Mamba weights
        self.apply(partial(_init_weights, n_layer=4))

    def forward(self, net, *inputs):
        """
        Drop-in replacement for ConvGRU.forward(net, *inputs).

        Args:
            net: [B, 128, H, W] — hidden state (outer h)
            *inputs: tuple of tensors to concatenate
                     typically (inp[128], corr_enc[128], flow_enc[64])

        Returns:
            net_new: [B, 128, H, W] — updated hidden state
        """
        # 1. Concatenate all inputs
        inp = torch.cat(inputs, dim=1)          # [B, 320, H, W]
        x = torch.cat([net, inp], dim=1)        # [B, 448, H, W]

        # 2. Project to d_model
        x = self.input_proj(x)                  # [B, D, H, W]

        # 3. Omnidirectional scan: 2D → 4 × 1D sequences
        seqs, H_orig, W_orig = omni_scan(x, step_size=self.step_size)
        # seqs: [B, 4, L, D]

        # 4. Run 4 independent Mamba blocks
        y0 = self.mamba_blocks[0](seqs[:, 0])   # [B, L, D]  →
        y1 = self.mamba_blocks[1](seqs[:, 1])   # [B, L, D]  ←
        y2 = self.mamba_blocks[2](seqs[:, 2])   # [B, L, D]  ↓
        y3 = self.mamba_blocks[3](seqs[:, 3])   # [B, L, D]  ↑

        # Stack and transpose for merge: [B, 4, L, D] → [B, 4, D, L]
        y = torch.stack([y0, y1, y2, y3], dim=1).transpose(2, 3)

        # 5. Merge back to 2D
        out = omni_merge(y, H_orig, W_orig, step_size=self.step_size)
        # out: [B, D, H, W]

        # 6. Aggregator + residual from hidden state
        out = self.aggregator(out)
        out = self.output_proj(out)

        # Residual connection with the previous hidden state
        net_new = net + out

        return net_new
