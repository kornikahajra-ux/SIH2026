"""
PyTorch Dataset and DataLoader module for OceanEmbed.
Handles data loading, train-only Z-score normalization, land masking, and tensor formatting.

Usage:
    python dataset.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import xarray as xr

from config import PROCESSED_DIR


class OceanDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        stats: dict | None = None,
        data_dir: Path = PROCESSED_DIR,
    ):
        """
        Args:
            split: One of 'train', 'val', or 'test'.
            stats: Dictionary containing 'input_mean', 'input_std', 'target_mean', 'target_std'.
                   If None and split == 'train', stats are computed automatically.
            data_dir: Path to directory containing processed NetCDF files.
        """
        super().__init__()
        self.split = split
        self.data_dir = data_dir

        # Load processed NetCDF files
        inputs_path = self.data_dir / f"surface_inputs_{split}.nc"
        target_path = self.data_dir / f"glorys_target_{split}.nc"

        if not inputs_path.exists() or not target_path.exists():
            raise FileNotFoundError(
                f"Missing split files for '{split}'. Run split_data.py first."
            )

        self.ds_inputs = xr.open_dataset(inputs_path)
        self.ds_target = xr.open_dataset(target_path)

        # Process input variables into standardized (Time, Lat, Lon) 3D arrays
        processed_input_list = []
        print(f"[{split}] Loading and standardizing surface input variables:")

        for var in list(self.ds_inputs.data_vars):
            da = self.ds_inputs[var]

            # Drop extra 1D coordinates or non-spatial dimensions
            for dim in list(da.dims):
                if dim not in ["time", "lat", "lon"] and da.sizes[dim] == 1:
                    da = da.squeeze(dim, drop=True)

            # Enforce exact dimension order: (time, lat, lon)
            da = da.transpose("time", "lat", "lon")
            arr = da.values.astype(np.float32)

            print(f"  - {var:15s}: shape {arr.shape}")
            processed_input_list.append(arr)

        # Inputs shape: (Time, Channels, Lat, Lon)
        self.input_data = np.stack(processed_input_list, axis=1)

        # Process target dataset
        target_var = list(self.ds_target.data_vars)[0]  # thetao
        target_da = self.ds_target[target_var]

        # Ensure depth axis remains present: (time, depth, lat, lon)
        if "depth" in target_da.dims:
            target_da = target_da.transpose("time", "depth", "lat", "lon")

        self.target_data = target_da.values.astype(np.float32)

        # --- Static 2D horizontal ocean mask (kept for reference / any
        # legacy code that still expects a (H, W) mask) ---
        self.ocean_mask = ~np.isnan(self.input_data[0, 0, :, :])
        self.mask_tensor = torch.from_numpy(self.ocean_mask.astype(np.float32))

        # --- FIXED: depth-aware validity mask ---
        # The old code only ever produced the 2D mask above and broadcast it
        # uniformly across all 35 depths. Real bathymetry means large parts
        # of the domain (Arabian Sea / Bay of Bengal shelf, Persian Gulf,
        # etc.) are ocean at the surface but shallower than many of the
        # deeper target levels, so GLORYS stores NaN there below the true
        # seafloor. Those NaNs then got zeroed out by _normalize() and
        # trained/scored against as if they were real "average temperature"
        # targets. This mask instead tracks validity per depth, per cell,
        # directly from the target's own NaN pattern.
        self.depth_mask = ~np.isnan(self.target_data)  # shape (time, depth, lat, lon)

        # Calculate or apply Z-score normalization parameters
        if stats is None:
            if split != "train":
                raise ValueError("Normalization stats from training split must be passed for val/test sets.")
            self.stats = self._compute_stats()
        else:
            self.stats = stats

        # Normalize data and replace land/below-seafloor NaNs with 0.0
        self.norm_inputs = self._normalize(
            self.input_data, self.stats["input_mean"], self.stats["input_std"]
        )
        self.norm_targets = self._normalize(
            self.target_data, self.stats["target_mean"], self.stats["target_std"]
        )

    def _compute_stats(self) -> dict:
        """Compute channel-wise mean and std across spatial and temporal dimensions (ignoring NaNs)."""
        # Channel-wise stats for surface inputs: shape (1, Channels, 1, 1)
        input_mean = np.nanmean(self.input_data, axis=(0, 2, 3), keepdims=True)
        input_std = np.nanstd(self.input_data, axis=(0, 2, 3), keepdims=True)
        input_std[input_std == 0] = 1.0  # Avoid division by zero

        # Depth-wise stats for target ocean temperature: shape (1, Depth, 1, 1)
        target_mean = np.nanmean(self.target_data, axis=(0, 2, 3), keepdims=True)
        target_std = np.nanstd(self.target_data, axis=(0, 2, 3), keepdims=True)
        target_std[target_std == 0] = 1.0

        return {
            "input_mean": input_mean,
            "input_std": input_std,
            "target_mean": target_mean,
            "target_std": target_std,
        }

    def _normalize(self, data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
        """Apply Z-score normalization and fill NaNs with 0.0."""
        normed = (data - mean) / std
        np.nan_to_num(normed, copy=False, nan=0.0)
        return normed

    def __len__(self) -> int:
        return self.input_data.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            x: Input surface tensor of shape (C_in, H, W)
            y: Target temperature profile tensor of shape (C_out, H, W)
            mask: Depth-aware validity mask of shape (Depth, H, W) - True only
                  where GLORYS actually has a real value at that depth/cell,
                  i.e. excludes below-seafloor shelf regions per depth.
        """
        x = torch.from_numpy(self.norm_inputs[idx])
        y = torch.from_numpy(self.norm_targets[idx])
        mask = torch.from_numpy(self.depth_mask[idx].astype(np.float32))
        return x, y, mask


def get_dataloaders(batch_size: int = 8, num_workers: int = 0) -> tuple[DataLoader, DataLoader, DataLoader, dict]:
    """Factory function to build train, val, and test PyTorch DataLoaders."""
    train_dataset = OceanDataset(split="train")
    stats = train_dataset.stats

    val_dataset = OceanDataset(split="val", stats=stats)
    test_dataset = OceanDataset(split="test", stats=stats)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, stats


if __name__ == "__main__":
    train_loader, val_loader, test_loader, stats = get_dataloaders(batch_size=4)
    x_sample, y_sample, mask_sample = next(iter(train_loader))

    print(f"\nDataset verification successful:")
    print(f"  Train batches : {len(train_loader)}")
    print(f"  Val batches   : {len(val_loader)}")
    print(f"  Test batches  : {len(test_loader)}")
    print(f"  Input Tensor  : {x_sample.shape}  (Batch, Channels, Lat, Lon)")
    print(f"  Target Tensor : {y_sample.shape} (Batch, Depths, Lat, Lon)")
    print(f"  Mask Tensor   : {mask_sample.shape}   (Batch, Depths, Lat, Lon) - now depth-aware")
