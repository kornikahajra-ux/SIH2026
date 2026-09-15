"""
Deep Learning Architecture for OceanEmbed - split thermocline / remaining-depths submodels.

Rather than one UNet predicting all 35 subsurface depth levels at once, the
target column is split into two disjoint depth bands, each handled by its
own independently-trained OceanUNet instance:

  - Thermocline submodel:      depth indices 17-24        (8 levels)
  - Remaining-depths submodel: depth indices 0-16 + 25-34 (27 levels)

Both submodels share the exact same encoder/decoder architecture (OceanUNet
below) - only the final 1x1 projection head's channel count differs. This
keeps the thermocline band (the fastest-varying, hardest-to-predict part of
the profile) trainable/tunable/checkpointable independently from the rest
of the column, without duplicating the architecture code twice.

split_full_depth_profile() / assemble_full_depth_profile() convert between
the full (B, 35, H, W) tensors dataset.py produces and the two submodels'
sliced tensors, so train.py and any evaluation script only have to call
them once each instead of re-deriving the index math.

Usage:
    python model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- Depth-band split (Python slice convention: [START, END)) ----
TOTAL_DEPTH_LEVELS = 35

THERMOCLINE_DEPTH_START = 17
THERMOCLINE_DEPTH_END = 25          # exclusive -> covers indices 17..24
THERMOCLINE_NUM_LEVELS = THERMOCLINE_DEPTH_END - THERMOCLINE_DEPTH_START  # 8

SHALLOW_DEPTH_RANGE = (0, 17)       # indices 0-16
DEEP_DEPTH_RANGE = (25, 35)         # indices 25-34
REMAINING_NUM_LEVELS = (
    (SHALLOW_DEPTH_RANGE[1] - SHALLOW_DEPTH_RANGE[0])
    + (DEEP_DEPTH_RANGE[1] - DEEP_DEPTH_RANGE[0])
)  # 27

assert THERMOCLINE_NUM_LEVELS + REMAINING_NUM_LEVELS == TOTAL_DEPTH_LEVELS, (
    "Thermocline + remaining depth counts must reconstruct the full 35-level column."
)


class DoubleConv(nn.Module):
    """(Convolution -> Batch Normalization -> ReLU) * 2"""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class OceanUNet(nn.Module):
    """
    Shared encoder/decoder backbone for both submodels. Identical for the
    thermocline and remaining-depths variants - only `out_channels` (the
    final 1x1 projection's channel count) differs between the two, which is
    why this one class replaces what would otherwise be two near-duplicate
    model definitions.
    """
    def __init__(self, in_channels: int = 12, out_channels: int = TOTAL_DEPTH_LEVELS):
        super().__init__()
        # Encoder (Downsampling)
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Bottleneck Latent Embedding
        self.bottleneck = DoubleConv(256, 512)

        # Decoder Level 1 (512 -> 256)
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256 + 128, 256)

        # Decoder Level 2 (256 -> 128)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128 + 64, 128)

        # Final projection head - channel count set by the caller
        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)             # (B, 64, 101, 241)
        x2 = self.down1(x1)          # (B, 128, 50, 120)
        x3 = self.down2(x2)          # (B, 256, 25, 60)

        # Bottleneck
        x_b = self.bottleneck(x3)    # (B, 512, 25, 60)

        # Decoder Step 1
        x = self.up1(x_b)            # (B, 256, 50, 120)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up1(x)         # (B, 256, 50, 120)

        # Decoder Step 2
        x = self.up2(x)              # (B, 128, 100, 240)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)         # (B, 128, 101, 241)

        # Output Head - channel count depends on which submodel this is
        logits = self.outc(x)
        return logits


def build_thermocline_model(in_channels: int = 12) -> OceanUNet:
    """OceanUNet narrowed to the 8 thermocline depth levels (indices 17-24)."""
    return OceanUNet(in_channels=in_channels, out_channels=THERMOCLINE_NUM_LEVELS)


def build_remaining_depths_model(in_channels: int = 12) -> OceanUNet:
    """OceanUNet narrowed to the 27 non-thermocline depth levels (indices 0-16, 25-34)."""
    return OceanUNet(in_channels=in_channels, out_channels=REMAINING_NUM_LEVELS)


def split_full_depth_profile(full: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Slice a full (B, 35, H, W) target or depth-mask tensor (as produced by
    dataset.py) into the (thermocline, remaining) pieces each submodel is
    trained against.

    Returns:
        thermocline: (B, 8, H, W)  - depth indices 17-24
        remaining:   (B, 27, H, W) - depth indices 0-16 concatenated with 25-34
    """
    thermocline = full[:, THERMOCLINE_DEPTH_START:THERMOCLINE_DEPTH_END]
    remaining = torch.cat(
        [
            full[:, SHALLOW_DEPTH_RANGE[0]:SHALLOW_DEPTH_RANGE[1]],
            full[:, DEEP_DEPTH_RANGE[0]:DEEP_DEPTH_RANGE[1]],
        ],
        dim=1,
    )
    return thermocline, remaining


def assemble_full_depth_profile(thermocline: torch.Tensor, remaining: torch.Tensor) -> torch.Tensor:
    """
    Inverse of split_full_depth_profile(): reassemble the two submodels'
    predictions (or targets/masks) back into a single (B, 35, H, W) tensor
    in native-depth order, so downstream code (evaluation, plotting,
    prediction) can treat the pair of submodels as a single 35-depth model.
    """
    B, _, H, W = thermocline.shape
    shallow_len = SHALLOW_DEPTH_RANGE[1] - SHALLOW_DEPTH_RANGE[0]

    full = thermocline.new_empty((B, TOTAL_DEPTH_LEVELS, H, W))
    full[:, THERMOCLINE_DEPTH_START:THERMOCLINE_DEPTH_END] = thermocline
    full[:, SHALLOW_DEPTH_RANGE[0]:SHALLOW_DEPTH_RANGE[1]] = remaining[:, :shallow_len]
    full[:, DEEP_DEPTH_RANGE[0]:DEEP_DEPTH_RANGE[1]] = remaining[:, shallow_len:]
    return full


class MaskedMSELoss(nn.Module):
    """
    Mean Squared Error computed only over valid ocean cells. Depth-count
    agnostic - the same class works for the 8-level thermocline tensors,
    the 27-level remaining-depths tensors, or a full 35-level tensor, as
    long as pred/target/mask shapes line up with each other.

    Accepts either:
      - a legacy 2D mask (B, H, W), broadcast identically across every
        depth channel in pred, or
      - a depth-aware mask (B, Depth, H, W) already sliced (via
        split_full_depth_profile) to match pred's own depth count.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == pred.dim() - 1:
            mask_expanded = mask.unsqueeze(1).expand_as(pred)
        elif mask.dim() == pred.dim():
            mask_expanded = mask
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(mask.shape)} for pred shape {tuple(pred.shape)}. "
                f"Expected either (B, H, W) or (B, {pred.shape[1]}, H, W)."
            )

        diff = (pred - target) ** 2
        masked_diff = diff * mask_expanded
        return masked_diff.sum() / (mask_expanded.sum() + 1e-8)


if __name__ == "__main__":
    torch.manual_seed(0)

    model_thermo = build_thermocline_model()
    model_remaining = build_remaining_depths_model()
    criterion = MaskedMSELoss()

    dummy_input = torch.randn(4, 12, 101, 241)
    dummy_full_target = torch.randn(4, TOTAL_DEPTH_LEVELS, 101, 241)
    dummy_full_mask = torch.ones(4, TOTAL_DEPTH_LEVELS, 101, 241)

    target_thermo, target_remaining = split_full_depth_profile(dummy_full_target)
    mask_thermo, mask_remaining = split_full_depth_profile(dummy_full_mask)

    pred_thermo = model_thermo(dummy_input)
    pred_remaining = model_remaining(dummy_input)

    loss_thermo = criterion(pred_thermo, target_thermo, mask_thermo)
    loss_remaining = criterion(pred_remaining, target_remaining, mask_remaining)

    full_pred = assemble_full_depth_profile(pred_thermo, pred_remaining)
    roundtrip_ok = torch.equal(full_pred[:, THERMOCLINE_DEPTH_START:THERMOCLINE_DEPTH_END], pred_thermo)

    print("Split OceanUNet Verification:")
    print(f"  Input Shape                 : {dummy_input.shape}")
    print(f"  Thermocline Output Shape    : {pred_thermo.shape} (Expected: [4, {THERMOCLINE_NUM_LEVELS}, 101, 241])")
    print(f"  Remaining Output Shape      : {pred_remaining.shape} (Expected: [4, {REMAINING_NUM_LEVELS}, 101, 241])")
    print(f"  Reassembled Full Shape      : {full_pred.shape} (Expected: [4, {TOTAL_DEPTH_LEVELS}, 101, 241])")
    print(f"  Thermocline Loss            : {loss_thermo.item():.4f}")
    print(f"  Remaining Loss              : {loss_remaining.item():.4f}")
    print(f"  Split/assemble round-trip OK: {roundtrip_ok}")
