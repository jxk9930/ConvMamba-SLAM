"""
omni_scan.py
============
Single-image omnidirectional scan/merge for ConvMamba-SLAM.

Adapted from JamMa's JEGO scan strategy, but for a SINGLE feature map
(not joint two-image). We keep only:
  - Skip scan (step_size=2) for efficiency
  - 4-directional scan for omnidirectional coverage
  - Aggregator (GLU-gated 3x3 conv) for local mixing

Note on padding:
  The original JEGO concatenates two images (2H×W, H×2W) guaranteeing even
  dimensions. For single-image scan, we pad H and W to multiples of step_size
  to ensure all 4 sub-grids have identical sequence lengths.

Usage:
    from modules.omni_scan import omni_scan, omni_merge, Aggregator

    seqs, H_orig, W_orig = omni_scan(feat_2d, step_size=2)
    # seqs: [B, 4, seq_len, C]  — all 4 have same seq_len

    feat_2d_out = omni_merge(seqs_processed, H_orig, W_orig, step_size=2)
    # feat_2d_out: [B, C, H, W]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def omni_scan(feat, step_size=2):
    """
    Scan a single 2D feature map into 4 directional 1D sequences.

    Scan directions:
        S0: right scan  → (even row, even col), row-major order
        S1: left scan   ← (odd col, odd row) transposed, reversed
        S2: down scan   ↓ (even row, odd col), reversed
        S3: up scan     ↑ (even col, odd row) transposed, reversed

    Together, the 4 sub-grids partition ALL pixels exactly once
    (with step_size=2: each covers H/2 × W/2 pixels).

    The Aggregator's 3×3 conv then mixes neighboring pixels from
    different directions, enabling omnidirectional information flow.

    Args:
        feat: [B, C, H, W] — single 2D feature map
        step_size: int — skip step (default 2)

    Returns:
        seqs:   [B, 4, L, C] — 4 directional sequences (equal length L)
        H_orig: int — original height (for merge)
        W_orig: int — original width (for merge)
    """
    B, C, H, W = feat.shape

    # Pad to multiples of step_size so all sub-grids are equal sized
    H_pad = math.ceil(H / step_size) * step_size
    W_pad = math.ceil(W / step_size) * step_size
    if H_pad != H or W_pad != W:
        feat = F.pad(feat, (0, W_pad - W, 0, H_pad - H))

    H_sub = H_pad // step_size
    W_sub = W_pad // step_size
    L = H_sub * W_sub  # guaranteed equal for all 4 directions

    seqs = feat.new_empty((B, 4, C, L))

    # Direction 0: right scan → even rows, even cols (row-major)
    seqs[:, 0] = feat[:, :, 0::step_size, 0::step_size].contiguous().view(B, C, -1)

    # Direction 1: left scan ← odd cols, odd rows (transposed, reversed)
    feat_t = feat.transpose(2, 3)  # [B, C, W_pad, H_pad]
    seqs[:, 1] = feat_t[:, :, 1::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])

    # Direction 2: down scan ↓ even rows, odd cols (reversed)
    seqs[:, 2] = feat[:, :, 0::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])

    # Direction 3: up scan ↑ even cols, odd rows (transposed, reversed)
    seqs[:, 3] = feat_t[:, :, 0::step_size, 1::step_size].contiguous().view(B, C, -1).flip([2])

    # [B, 4, C, L] → [B, 4, L, C] for Mamba (expects seq_len before channels)
    seqs = seqs.transpose(2, 3)

    return seqs, H, W


def omni_merge(seqs, H_orig, W_orig, step_size=2):
    """
    Merge 4 directional 1D sequences back into a single 2D feature map.

    Inverse of omni_scan(). Each direction fills its sub-grid positions,
    then horizontal and vertical grids are summed (they occupy disjoint
    rows, so the sum is effectively a union).

    Args:
        seqs:   [B, 4, C, L] — processed sequences (C before L!)
                Caller must transpose from [B,4,L,C] → [B,4,C,L] before calling.
        H_orig: int — original height
        W_orig: int — original width
        step_size: int

    Returns:
        feat: [B, C, H_orig, W_orig] — reconstructed 2D feature map
    """
    B, _, C, L = seqs.shape

    H_pad = math.ceil(H_orig / step_size) * step_size
    W_pad = math.ceil(W_orig / step_size) * step_size
    H_sub = H_pad // step_size
    W_sub = W_pad // step_size

    # Horizontal grid: directions 0 (right) and 2 (down) fill even rows
    feat_h = seqs.new_zeros((B, C, H_pad, W_pad))
    feat_h[:, :, 0::step_size, 0::step_size] = seqs[:, 0].reshape(B, C, H_sub, W_sub)
    feat_h[:, :, 0::step_size, 1::step_size] = seqs[:, 2].flip([2]).reshape(B, C, H_sub, W_sub)

    # Vertical grid: directions 1 (left) and 3 (up) fill odd rows
    feat_v = seqs.new_zeros((B, C, H_pad, W_pad))
    feat_v[:, :, 1::step_size, 1::step_size] = \
        seqs[:, 1].flip([2]).reshape(B, C, W_sub, H_sub).transpose(2, 3)
    feat_v[:, :, 1::step_size, 0::step_size] = \
        seqs[:, 3].flip([2]).reshape(B, C, W_sub, H_sub).transpose(2, 3)

    # Sum (disjoint rows → effectively union) and crop to original size
    feat = feat_h + feat_v
    if H_orig != H_pad or W_orig != W_pad:
        feat = feat[:, :, :H_orig, :W_orig].contiguous()

    return feat


class Aggregator(nn.Module):
    """
    GLU-gated 3x3 conv aggregator from JamMa.

    After merge, each pixel has global info from exactly ONE direction.
    The aggregator's 3x3 conv allows neighboring pixels (from different
    directions) to share information, achieving true omnidirectional
    coverage. This is the key insight from the JamMa paper:
    "balanced receptive field + aggregator = omnidirectional features."

    Structure: GELU(W(x)) * V(x) → W2(x)
    """

    def __init__(self, dim, mid_dim=None):
        super().__init__()
        if mid_dim is None:
            mid_dim = dim
        self.W = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.V = nn.Conv2d(dim, mid_dim, kernel_size=3, padding=1, bias=False)
        self.W2 = nn.Conv2d(mid_dim, dim, kernel_size=3, padding=1, bias=False)
        self.act = nn.GELU()

    def forward(self, feat):
        return self.W2(self.act(self.W(feat)) * self.V(feat))
