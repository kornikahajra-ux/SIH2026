"""
Deep Learning Architecture for OceanEmbed.
A 2D Convolutional UNet mapping 2D multi-variable surface inputs (12 channels)
to 3D subsurface temperature profiles across 35 depth layers.

Integrated with ThermoclinePhysicsLoss to penalize reconstruction, gradient,
and stratification errors strictly in the 50m-200m thermocline region.

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


class BasicThermoclineAttention(nn.Module):
    """
    Lightweight Attention Gate applied strictly to the 50m-200m depth channels.
    Combines basic spatial and channel gating with virtually zero CPU overhead.
    """
    def __init__(self, num_channels: int):
        super().__init__()
        # Spatial attention map
        self.spatial_att = nn.Sequential(
            nn.Conv2d(num_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        # Channel attention weights
        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_channels, num_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_mask = self.spatial_att(x)
        c_mask = self.channel_att(x)
        return x * s_mask * c_mask


class OceanUNetThermoclineAttention(nn.Module):
    def __init__(
        self, 
        in_channels: int = 12, 
        out_channels: int = 35,
        thermocline_start_idx: int = 16,  # Approx 50m depth index
        thermocline_end_idx: int = 26     # Approx 200m depth index
    ):
        super().__init__()
        self.thermocline_start = thermocline_start_idx
        self.thermocline_end = thermocline_end_idx
        num_thermocline_channels = thermocline_end_idx - thermocline_start_idx

        # Standard UNet Encoder
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))

        # Standard UNet Bottleneck
        self.bottleneck = DoubleConv(256, 512)

        # Standard UNet Decoder
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(256 + 128, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128 + 64, 128)

        # Output projection to 35 subsurface depth levels
        self.outc = nn.Conv2d(128, out_channels, kernel_size=1)

        # Basic Attention Module focused only on 50m - 200m depth slice
        self.thermocline_att = BasicThermoclineAttention(num_channels=num_thermocline_channels)

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
        x = self.conv_up1(x)           # (B, 256, 50, 120)

        # Decoder Step 2
        x = self.up2(x)              # (B, 128, 100, 240)
        if x.shape[2:] != x1.shape[2:]:
            x = F.interpolate(x, size=x1.shape[2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up2(x)           # (B, 128, 101, 241)

        # Raw predictions across all 35 depth levels
        logits = self.outc(x)        # (B, 35, 101, 241)

        # Apply basic attention specifically to the 50m-200m depth channels
        top_layers = logits[:, :self.thermocline_start, :, :]
        thermocline_layers = self.thermocline_att(
            logits[:, self.thermocline_start:self.thermocline_end, :, :]
        )
        deep_layers = logits[:, self.thermocline_end:, :, :]

        # Reassemble the 3D temperature profile
        output = torch.cat([top_layers, thermocline_layers, deep_layers], dim=1)
        return output


class MaskedMSELoss(nn.Module):
    """
    Calculates Mean Squared Error exclusively on valid ocean cells.
    Accepts 2D masks (B, H, W) or depth-aware 3D masks (B, Depth, H, W).
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
                "Expected either (B, H, W) or (B, Depth, H, W)."
            )

        diff = (pred - target) ** 2
        masked_diff = diff * mask_expanded

        return masked_diff.sum() / (mask_expanded.sum() + 1e-8)


class MaskedThermoclinePhysicsLoss(nn.Module):
    """
    Physics-Informed Loss that enforces soft constraints ONLY within the thermocline depth slice.
    
    Components:
      1. Global Masked MSE (w_global): Preserves baseline performance on all 35 layers.
      2. Thermocline Masked MSE (w_thermo): Extra penalty on thermocline reconstruction errors.
      3. Vertical Gradient Loss (w_grad): Matches vertical rate of change (dT/dz) in thermocline.
      4. Stratification Loss (w_strat): Penalizes unphysical temperature inversions (dT/dz > 0).
    """
    def __init__(
        self,
        thermocline_start_idx: int = 16,
        thermocline_end_idx: int = 26,
        w_global: float = 1.0,
        w_thermo: float = 2.0,
        w_grad: float = 1.5,
        w_strat: float = 0.5
    ):
        super().__init__()
        self.start_idx = thermocline_start_idx
        self.end_idx = thermocline_end_idx
        self.w_global = w_global
        self.w_thermo = w_thermo
        self.w_grad = w_grad
        self.w_strat = w_strat
        self.base_mse = MaskedMSELoss()

    def _expand_mask(self, mask: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
        if mask.dim() == pred.dim() - 1:
            return mask.unsqueeze(1).expand_as(pred)
        elif mask.dim() == pred.dim():
            return mask
        else:
            raise ValueError(
                f"Unexpected mask shape {tuple(mask.shape)} for pred shape {tuple(pred.shape)}."
            )

    def forward(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_expanded = self._expand_mask(mask, pred)

        # 1. Base Global Masked MSE Loss across all layers
        l_global = self.base_mse(pred, target, mask_expanded)

        # Extract strictly the thermocline slices (Channels 16 to 26)
        pred_thermo = pred[:, self.start_idx:self.end_idx, :, :]
        target_thermo = target[:, self.start_idx:self.end_idx, :, :]
        mask_thermo = mask_expanded[:, self.start_idx:self.end_idx, :, :]

        # 2. Focused Thermocline Data Reconstruction Loss
        diff_thermo = (pred_thermo - target_thermo) ** 2
        l_thermo = (diff_thermo * mask_thermo).sum() / (mask_thermo.sum() + 1e-8)

        # 3. Thermocline Vertical Gradient Loss (dT/dz)
        # Calculates difference between adjacent vertical layers strictly within thermocline
        pred_grad = pred_thermo[:, 1:, :, :] - pred_thermo[:, :-1, :, :]
        target_grad = target_thermo[:, 1:, :, :] - target_thermo[:, :-1, :, :]
        
        # Valid mask for gradients (both adjacent layers must be valid ocean cells)
        mask_grad = mask_thermo[:, 1:, :, :] * mask_thermo[:, :-1, :, :]

        diff_grad = (pred_grad - target_grad) ** 2
        l_grad = (diff_grad * mask_grad).sum() / (mask_grad.sum() + 1e-8)

        # 4. Thermocline Monotonic Stratification Inversion Penalty
        # In stable ocean stratification, T_{z+1} <= T_z. Penalize positive steps (T_{z+1} - T_z > 0)
        inversions = F.relu(pred_grad) ** 2
        l_strat = (inversions * mask_grad).sum() / (mask_grad.sum() + 1e-8)

        # Total Weighted Loss
        total_loss = (
            (self.w_global * l_global) +
            (self.w_thermo * l_thermo) +
            (self.w_grad * l_grad) +
            (self.w_strat * l_strat)
        )
        return total_loss


if __name__ == "__main__":
    model = OceanUNetThermoclineAttention(
        in_channels=12, 
        out_channels=35,
        thermocline_start_idx=16,
        thermocline_end_idx=26
    )
    dummy_input = torch.randn(4, 12, 101, 241)
    dummy_target = torch.randn(4, 35, 101, 241)
    
    # Instantiate physics loss with thermocline slice settings
    physics_criterion = MaskedThermoclinePhysicsLoss(
        thermocline_start_idx=16,
        thermocline_end_idx=26,
        w_global=1.0,
        w_thermo=2.0,
        w_grad=1.5,
        w_strat=0.5
    )
    
    output = model(dummy_input)

    # Verify both supported mask shapes
    dummy_mask_2d = torch.ones(4, 101, 241)
    dummy_mask_3d = torch.ones(4, 35, 101, 241)

    loss_2d = physics_criterion(output, dummy_target, dummy_mask_2d)
    loss_3d = physics_criterion(output, dummy_target, dummy_mask_3d)

    print("Model Architecture & Thermocline Loss Verification:")
    print(f"  Input Shape             : {dummy_input.shape}")
    print(f"  Output Shape            : {output.shape} (Expected: [4, 35, 101, 241])")
    print(f"  Physics Loss (2D mask)  : {loss_2d.item():.4f}")
    print(f"  Physics Loss (3D mask)  : {loss_3d.item():.4f}")
