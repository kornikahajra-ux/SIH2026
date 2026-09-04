import matplotlib.pyplot as plt
import numpy as np
from dataset import get_dataloaders

train_loader, _, _, stats = get_dataloaders(batch_size=1)
x, y, mask = next(iter(train_loader))

# Extract numpy matrices for 1 sample
input_sst = x[0, 0].numpy()       # Channel 0 (SST)
target_0m = y[0, 0].numpy()       # Target Level 0 (Surface)
target_500m = y[0, 15].numpy()    # Target Level 15 (~500m depth)
ocean_mask = mask[0].numpy()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1. Input SST Map
im0 = axes[0, 0].imshow(np.where(ocean_mask, input_sst, np.nan), cmap="coolwarm", origin="lower")
axes[0, 0].set_title("Input SST (Normalized)")
plt.colorbar(im0, ax=axes[0, 0])

# 2. Binary Ocean Mask Map
im1 = axes[0, 1].imshow(ocean_mask, cmap="Blues", origin="lower")
axes[0, 1].set_title("Ocean Mask (1=Ocean, 0=Land)")
plt.colorbar(im1, ax=axes[0, 1])

# 3. Target Temperature at Surface (0m)
im2 = axes[1, 0].imshow(np.where(ocean_mask, target_0m, np.nan), cmap="inferno", origin="lower")
axes[1, 0].set_title("Target Temp (Surface / 0m)")
plt.colorbar(im2, ax=axes[1, 0])

# 4. Target Temperature at Subsurface Layer (~500m)
im3 = axes[1, 1].imshow(np.where(ocean_mask, target_500m, np.nan), cmap="inferno", origin="lower")
axes[1, 1].set_title("Target Temp (Subsurface Layer 15)")
plt.colorbar(im3, ax=axes[1, 1])

plt.tight_layout()
plt.show()