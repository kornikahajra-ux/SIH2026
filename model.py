"""
Deep Learning Architecture for OceanEmbed.
A 2D Convolutional UNet mapping 2D multi-variable surface inputs (12 channels)
to 3D subsurface temperature profiles across 35 depth layers.

Usage:
    python model.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, in_channels: int = 12, out_channels: int = 35):
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

        # Final projection to 35 subsurface depth levels
        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)         # (B, 64, 101, 241)
        x2 = self.down1(x1)      # (B, 128, 50, 120)
        x3 = self.down2(x2)      # (B, 256, 25, 60)

        # Bottleneck
        x_b = self.bottleneck(x3)  # (B, 512, 25, 60)

        # Decoder Step 1
        x = self.up1(x_b)        # (B, 256, 50, 120)
        if x.shape[2:] != x2.shape[2:]:
            x = F.interpolate(x, size=x2.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x2], dim=1)  # Concat with x2 (B, 384, 50, 120)
        x = self.conv_up1(x)           # (B, 256, 50, 120)

        # Decoder Step 2
        x = self.up2(x)          # (B, 128, 100, 240)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)  # Concat with x1 (B, 192, 101, 241)
        x = self.conv_up2(x)           # (B, 128, 101, 241)

        # Output Head
        logits = self.outc(x)          # (B, 35, 101, 241)
        return logits


class MaskedMSELoss(nn.Module):
    """
    Calculates Mean Squared Error exclusively on valid ocean cells.

    --- FIXED ---
    Previously this only ever accepted a 2D horizontal mask (B, H, W) and
    broadcast it identically across all 35 depths. That's wrong wherever
    real bathymetry is shallower than the deepest levels (much of the
    Arabian Sea / Bay of Bengal shelf, the Persian Gulf, etc.): GLORYS
    stores NaN below the true seafloor, dataset.py's normalize() step
    converts those NaNs to 0.0, and the old uniform mask never excluded
    them - so the model was being trained (and scored) against a
    fabricated "average temperature" target at physically nonexistent
    water, worst at the depths where shelf coverage is largest.

    This version accepts either:
      - a legacy 2D mask (B, H, W), broadcast across depths as before
        (kept only for backward compatibility with old checkpoints/data), or
      - a depth-aware mask (B, Depth, H, W) - what the updated dataset.py
        now produces from the target's own per-depth NaN pattern - which
        correctly excludes below-seafloor cells at each individual depth.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dim() == pred.dim() - 1:
            # Legacy 2D mask (B, H, W) -> broadcast identically to every depth.
            mask_expanded = mask.unsqueeze(1).expand_as(pred)
        elif mask.dim() == pred.dim():
            # Depth-aware mask (B, Depth, H, W) already matches pred's shape.
            mask_expanded = mask
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(mask.shape)} for pred shape {tuple(pred.shape)}. "
                "Expected either (B, H, W) or (B, Depth, H, W)."
            )

        diff = (pred - target) ** 2
        masked_diff = diff * mask_expanded

        # Mean calculated only over valid ocean cells
        return masked_diff.sum() / (mask_expanded.sum() + 1e-8)


if __name__ == "__main__":
    model = OceanUNet(in_channels=12, out_channels=35)
    dummy_input = torch.randn(4, 12, 101, 241)
    dummy_target = torch.randn(4, 35, 101, 241)
    criterion = MaskedMSELoss()
    output = model(dummy_input)

    # Verify both supported mask shapes
    dummy_mask_2d = torch.ones(4, 101, 241)
    dummy_mask_3d = torch.ones(4, 35, 101, 241)

    loss_2d = criterion(output, dummy_target, dummy_mask_2d)
    loss_3d = criterion(output, dummy_target, dummy_mask_3d)

    print("Model Architecture Verification:")
    print(f"  Input Shape        : {dummy_input.shape}")
    print(f"  Output Shape       : {output.shape} (Expected: [4, 35, 101, 241])")
    print(f"  Loss (2D mask)     : {loss_2d.item():.4f}")
    print(f"  Loss (3D depth mask): {loss_3d.item():.4f}")
