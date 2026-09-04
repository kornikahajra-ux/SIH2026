import torch
import numpy as np
from dataset import get_dataloaders

# Fetch 1 batch from train set
train_loader, val_loader, test_loader, stats = get_dataloaders(batch_size=1)
x, y, mask = next(iter(train_loader))

print("=== Tensor Dimensions ===")
print(f"Inputs (x) : {x.shape}  # (Batch, 12 Surface Channels, Lat, Lon)")
print(f"Targets (y): {y.shape} # (Batch, 35 Depth Layers, Lat, Lon)")
print(f"Mask       : {mask.shape}   # (Batch, Lat, Lon)")

print("\n=== Ocean vs Land Pixel Breakdown ===")
total_pixels = mask.numel()
ocean_pixels = (mask == 1).sum().item()
land_pixels = (mask == 0).sum().item()
print(f"Ocean pixels : {ocean_pixels:,} ({ocean_pixels / total_pixels:.1%})")
print(f"Land pixels  : {land_pixels:,} ({land_pixels / total_pixels:.1%})")

print("\n=== Normalized Tensor Ranges ===")
print(f"Inputs  -> Min: {x.min():.3f}, Max: {x.max():.3f}, Mean: {x.mean():.3f}")
print(f"Targets -> Min: {y.min():.3f}, Max: {y.max():.3f}, Mean: {y.mean():.3f}")

print("\n=== Channel-wise Statistics (First Sample) ===")
for i in range(x.shape[1]):
    channel_data = x[0, i][mask[0] == 1]  # Extract ocean pixels only
    print(f"Channel {i:2d} -> Mean: {channel_data.mean():.3f}, Std: {channel_data.std():.3f}")