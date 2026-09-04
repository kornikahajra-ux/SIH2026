"""
Plotting and Visualization Script for OceanEmbed.
Generates:
1. Vertical metric profiles (RMSE, MAE, Pearson R) across physical depth (m) -> metrics_vs_depth.png
2. Spatial comparison slices (Target vs Prediction vs Absolute Error) -> spatial_evaluations.png

Usage:
    python plot_evaluations.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders
from model import OceanUNet
from config import NATIVE_DEPTHS_35, INCOIS_STANDARD_DEPTHS_M


def plot_vertical_metrics():
    # Prefer 15 INCOIS benchmark metrics, fall back to native metrics
    csv_15 = Path("evaluation_metrics_15incois.csv")
    csv_native = Path("evaluation_metrics.csv")

    if csv_15.exists():
        csv_path = csv_15
    elif csv_native.exists():
        csv_path = csv_native
    else:
        print(f"[!] Neither metrics CSV found. Running evaluation to generate metrics...")
        from evaluate import evaluate_model
        evaluate_model()
        csv_path = csv_15 if csv_15.exists() else csv_native

    df = pd.read_csv(csv_path)

    # Attach physical depth (m) if missing
    if "depth_m" not in df.columns:
        if len(df) == len(INCOIS_STANDARD_DEPTHS_M):
            df.insert(1, "depth_m", INCOIS_STANDARD_DEPTHS_M)
        elif len(df) == len(NATIVE_DEPTHS_35):
            df.insert(1, "depth_m", NATIVE_DEPTHS_35)
        else:
            df["depth_m"] = df["depth_index"]

    y_depth = df["depth_m"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Plot RMSE
    axes[0].plot(df["rmse"], y_depth, marker="o", color="crimson", linewidth=2)
    axes[0].set_title("RMSE (°C)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Root Mean Squared Error")
    axes[0].set_ylabel("Depth (meters)", fontsize=11, fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot MAE
    axes[1].plot(df["mae"], y_depth, marker="s", color="darkorange", linewidth=2)
    axes[1].set_title("MAE (°C)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Absolute Error")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Plot Pearson R
    axes[2].plot(df["pearson_r"], y_depth, marker="^", color="teal", linewidth=2)
    axes[2].set_title("Pearson Correlation (r)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Correlation Coefficient")
    axes[2].set_xlim(-0.1, 1.0)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    title_suffix = "(15 INCOIS Standard Depths)" if "15incois" in csv_path.name else "(35 Native Depths)"
    plt.suptitle(f"OceanEmbed: Subsurface Temperature Performance vs. Depth {title_suffix}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_img = Path("metrics_vs_depth.png")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved vertical metric profiles to: {out_img.resolve()}")


def plot_spatial_slices():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path("checkpoints/best_ocean_unet.pth")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = OceanUNet(in_channels=12, out_channels=35).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint.get("stats", {})
    _, _, test_loader, _ = get_dataloaders(batch_size=1)

    # Grab first test sample and enforce 3D shape: (35, H, W)
    x, y, mask = next(iter(test_loader))
    with torch.no_grad():
        pred = model(x.to(device)).cpu().numpy().squeeze()

    target = y.numpy().squeeze()
    mask_2d = mask.numpy().squeeze()

    # Un-normalize targets and predictions if normalization stats exist
    if "target_mean" in stats and "target_std" in stats:
        t_mean = np.squeeze(stats["target_mean"])
        t_std = np.squeeze(stats["target_std"])
        
        if t_mean.ndim == 1:
            t_mean = t_mean[:, None, None]
            t_std = t_std[:, None, None]
            
        pred = pred * t_std + t_mean
        target = target * t_std + t_mean

    # Mask land cells with NaN across all 35 depth levels
    land_mask = (mask_2d == 0)
    for d in range(35):
        pred[d][land_mask] = np.nan
        target[d][land_mask] = np.nan

    # Target specific physical depths (~0m, ~100m, ~150m/500m) in 35 native levels
    target_depths_m = [0.0, 100.0, 150.0]
    depth_indices = [
        int(np.argmin(np.abs(np.array(NATIVE_DEPTHS_35) - td))) 
        for td in target_depths_m
    ]

    fig, axes = plt.subplots(len(depth_indices), 3, figsize=(15, 10))

    for row_idx, d_idx in enumerate(depth_indices):
        t_slice = target[d_idx]
        p_slice = pred[d_idx]
        err_slice = np.abs(p_slice - t_slice)

        v_min = np.nanmin(t_slice)
        v_max = np.nanmax(t_slice)
        depth_label = f"{NATIVE_DEPTHS_35[d_idx]:.1f}m"

        # Ground Truth Target
        im0 = axes[row_idx, 0].imshow(t_slice, cmap="viridis", vmin=v_min, vmax=v_max, origin="lower")
        axes[row_idx, 0].set_title(f"Target Temp (°C) [{depth_label}]")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Model Prediction
        im1 = axes[row_idx, 1].imshow(p_slice, cmap="viridis", vmin=v_min, vmax=v_max, origin="lower")
        axes[row_idx, 1].set_title(f"Predicted Temp (°C) [{depth_label}]")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Absolute Error
        im2 = axes[row_idx, 2].imshow(err_slice, cmap="Reds", origin="lower")
        axes[row_idx, 2].set_title(f"Absolute Error (°C) [{depth_label}]")
        plt.colorbar(im2, ax=axes[row_idx, 2], fraction=0.046, pad=0.04)

        for col_idx in range(3):
            axes[row_idx, col_idx].axis("off")

    plt.suptitle("OceanEmbed Spatial Predictions vs Ground Truth (GLORYS)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    out_img = Path("spatial_evaluations.png")
    plt.savefig(out_img, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved spatial evaluation plots to: {out_img.resolve()}")


def main():
    plot_vertical_metrics()
    plot_spatial_slices()


if __name__ == "__main__":
    main()
