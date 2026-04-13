"""
jamma_encoder.py
================
ConvNeXt V2 encoder for JamMa-SLAM.

Adapted from JamMa's src/jamma/backbone.py (CovNextV2_nano).
Uses ConvNeXt V2-Nano first 2 stages (pretrained):
  - Stage 0: stride 4,  80 channels  (fine features — not used in v1)
  - Stage 1: stride 8, 160 channels  (coarse features)

Projection heads map 160ch → DROID-compatible dimensions:
  - coarse_proj: 160 → 128  (for JEGO enrichment + correlation volume)
  - context_proj: 160 → 256 → split to net(128, tanh) + inp(128, relu)

ConvNeXt V2 Nano specs:
  - depths = [2, 2, 8, 2], dims = [80, 160, 320, 640]
  - We use only stages 0-1 (0.65M params)
  - Pretrained on ImageNet (self-supervised FCMAE)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

# ──────────────────────────────────────────────
# ConvNeXt V2 components (from src/convnextv2/)
# ──────────────────────────────────────────────

class LayerNorm(nn.Module):
    """LayerNorm supporting channels_first and channels_last formats."""
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """Global Response Normalization layer."""
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 Block: DWConv → LayerNorm → PW1 → GELU → GRN → PW2."""
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # NCHW → NHWC
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # NHWC → NCHW
        x = residual + self.drop_path(x)
        return x


# ──────────────────────────────────────────────
# ConvNeXt V2 Backbone (first 2 stages only)
# ──────────────────────────────────────────────

class ConvNeXtV2Backbone(nn.Module):
    """
    ConvNeXt V2 backbone using only the first 2 stages.
    
    Stage 0: 3 → 80ch, stride 4 (H/4 × W/4)
    Stage 1: 80 → 160ch, stride 2 (H/8 × W/8)
    
    Total stride: 8 (matches DROID-SLAM feature resolution)
    """

    def __init__(self, depths=(2, 2), dims=(80, 160), drop_path_rate=0.):
        super().__init__()
        self.num_stages = len(depths)

        # Stem: stride 4 patchify
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )

        # Downsample between stages
        self.downsample = nn.Sequential(
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2),
        )

        # Build stages
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.stage0 = nn.Sequential(
            *[ConvNeXtV2Block(dim=dims[0], drop_path=dp_rates[j]) for j in range(depths[0])]
        )

        self.stage1 = nn.Sequential(
            *[ConvNeXtV2Block(dim=dims[1], drop_path=dp_rates[depths[0] + j]) for j in range(depths[1])]
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: [B, 3, H, W]
        Returns:
            feat_8: [B, 160, H/8, W/8] — coarse features at 1/8 resolution
        """
        # Stage 0: H/4 resolution, 80ch
        x = self.stem(x)
        x = self.stage0(x)
        # feat_4 = x  # [B, 80, H/4, W/4] — available for future fine features

        # Stage 1: H/8 resolution, 160ch
        x = self.downsample(x)
        x = self.stage1(x)
        feat_8 = x  # [B, 160, H/8, W/8]

        return feat_8

    def forward_multi_scale(self, x):
        """
        Return features at both 1/4 and 1/8 resolution.
        (For future fine matching support.)
        
        Returns:
            feat_4: [B, 80, H/4, W/4]
            feat_8: [B, 160, H/8, W/8]
        """
        x = self.stem(x)
        x = self.stage0(x)
        feat_4 = x

        x = self.downsample(x)
        x = self.stage1(x)
        feat_8 = x

        return feat_4, feat_8


# ──────────────────────────────────────────────
# Pretrained weight loading
# ──────────────────────────────────────────────

PRETRAINED_URL = (
    "https://github.com/leoluxxx/JamMa/releases/download/v0.1/"
    "convnextv2_nano_pretrain.ckpt"
)


def _load_pretrained_weights(backbone, url=PRETRAINED_URL):
    """
    Load pretrained ConvNeXt V2 Nano weights into 2-stage backbone.
    
    The pretrained checkpoint has full 4-stage ConvNeXt V2 Nano
    (depths=[2,2,8,2], dims=[80,160,320,640]).
    We extract only stages 0 and 1.
    """
    try:
        state_dict = torch.hub.load_state_dict_from_url(url, file_name="convnextv2_nano_pretrain.ckpt")
    except Exception as e:
        print(f"[jamma_encoder] Could not download pretrained weights: {e}")
        print("[jamma_encoder] Proceeding with random initialization.")
        return

    # Map full ConvNeXtV2 keys to our 2-stage backbone keys
    mapping = {}

    # Stem
    mapping["downsample_layers.0.0.weight"] = "stem.0.weight"
    mapping["downsample_layers.0.0.bias"] = "stem.0.bias"
    mapping["downsample_layers.0.1.weight"] = "stem.1.weight"
    mapping["downsample_layers.0.1.bias"] = "stem.1.bias"

    # Downsample (between stage 0 and 1)
    mapping["downsample_layers.1.0.weight"] = "downsample.0.weight"
    mapping["downsample_layers.1.0.bias"] = "downsample.0.bias"
    mapping["downsample_layers.1.1.weight"] = "downsample.1.weight"
    mapping["downsample_layers.1.1.bias"] = "downsample.1.bias"

    # Stage 0 blocks
    for i in range(2):  # depths[0] = 2
        prefix_src = f"stages.0.{i}."
        prefix_dst = f"stage0.{i}."
        for key in state_dict:
            if key.startswith(prefix_src):
                mapping[key] = key.replace(prefix_src, prefix_dst)

    # Stage 1 blocks
    for i in range(2):  # depths[1] = 2 (first 2 of potentially 2)
        prefix_src = f"stages.1.{i}."
        prefix_dst = f"stage1.{i}."
        for key in state_dict:
            if key.startswith(prefix_src):
                mapping[key] = key.replace(prefix_src, prefix_dst)

    # Build filtered state dict
    new_state_dict = {}
    for src_key, dst_key in mapping.items():
        if src_key in state_dict:
            new_state_dict[dst_key] = state_dict[src_key]

    missing, unexpected = backbone.load_state_dict(new_state_dict, strict=False)
    print(f"[jamma_encoder] Loaded pretrained weights: "
          f"{len(new_state_dict)} params loaded, "
          f"{len(missing)} missing, {len(unexpected)} unexpected")


# ──────────────────────────────────────────────
# Public API: build_convnext_backbone()
# ──────────────────────────────────────────────

def build_convnext_backbone(pretrained=True):
    """
    Build ConvNeXt V2 Nano backbone (first 2 stages).
    
    Args:
        pretrained: if True, load pretrained ImageNet weights
    Returns:
        ConvNeXtV2Backbone — outputs [B, 160, H/8, W/8]
    """
    backbone = ConvNeXtV2Backbone(
        depths=(2, 2),
        dims=(80, 160),
        drop_path_rate=0.0,
    )

    if pretrained:
        _load_pretrained_weights(backbone)

    return backbone


# ──────────────────────────────────────────────
# Sanity check
# ──────────────────────────────────────────────

if __name__ == "__main__":
    backbone = build_convnext_backbone(pretrained=False)

    # Count parameters
    n_params = sum(p.numel() for p in backbone.parameters())
    print(f"Backbone parameters: {n_params / 1e6:.2f}M")

    # Test forward pass
    x = torch.randn(2, 3, 384, 512)
    feat_8 = backbone(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {feat_8.shape}")  # expect [2, 160, 48, 64]

    # Multi-scale test
    feat_4, feat_8 = backbone.forward_multi_scale(x)
    print(f"Feat 1/4: {feat_4.shape}")  # expect [2, 80, 96, 128]
    print(f"Feat 1/8: {feat_8.shape}")  # expect [2, 160, 48, 64]
