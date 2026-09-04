"""
Plotting and Visualization Script for OceanEmbed.
Generates:
1. Vertical metric profiles (RMSE, MAE, Pearson R) across depth levels -> metrics_vs_depth.png
2. Spatial comparison slices (Target vs Prediction vs Absolute Error) -> spatial_evaluations.png

Usage:
    python plot_evaluations.py
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress argopy/erddapy import warnings

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders
from model import OceanUNet


def plot_vertical_metrics():
    csv_path = Path("evaluation_metrics.csv")
    if not csv_path.exists():
        print(f"[!] {csv_path} not found. Running evaluation to generate metrics...")
        from evaluate import evaluate_model
        evaluate_model()

    df = pd.read_csv(csv_path)

    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True)

    # Plot RMSE
    axes[0].plot(df["rmse"], df["depth_index"], marker="o", color="crimson", linewidth=2)
    axes[0].set_title("RMSE (°C)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Root Mean Squared Error")
    axes[0].set_ylabel("Depth Level Index (0 = Surface)")
    axes[0].invert_yaxis()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # Plot MAE
    axes[1].plot(df["mae"], df["depth_index"], marker="s", color="darkorange", linewidth=2)
    axes[1].set_title("MAE (°C)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Mean Absolute Error")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Plot Pearson R
    axes[2].plot(df["pearson_r"], df["depth_index"], marker="^", color="teal", linewidth=2)
    axes[2].set_title("Pearson Correlation (r)", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Correlation Coefficient")
    axes[2].set_xlim(-0.1, 1.0)
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.suptitle("OceanEmbed: Subsurface Temperature Performance vs. Depth", fontsize=14, fontweight="bold")
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

    # Grab first test sample and enforce 3D shapes: (35, 101, 241)
    x, y, mask = next(iter(test_loader))
    with torch.no_grad():
        pred = model(x.to(device)).cpu().numpy().squeeze()  # Force shape: (35, 101, 241)

    target = y.numpy().squeeze()      # Force shape: (35, 101, 241)
    mask_2d = mask.numpy().squeeze()  # Force shape: (101, 241)

    # Un-normalize targets and predictions if normalization stats exist
    if "target_mean" in stats and "target_std" in stats:
        t_mean = np.squeeze(stats["target_mean"])
        t_std = np.squeeze(stats["target_std"])
        
        # Ensure proper shape expansion for broadcast across spatial dimensions
        if t_mean.ndim == 1:
            t_mean = t_mean[:, None, None]
            t_std = t_std[:, None, None]
            
        pred = pred * t_std + t_mean
        target = target * t_std + t_mean

    # Mask land cells with NaN across all depth levels
    land_mask = (mask_2d == 0)
    for d in range(35):
        pred[d][land_mask] = np.nan
        target[d][land_mask] = np.nan

    # Select representative depth layers: Surface (0), Thermocline (~100m, idx 10), Deep (~500m, idx 20)
    depth_indices = [0, 10, 20]
    fig, axes = plt.subplots(len(depth_indices), 3, figsize=(15, 10))

    for row_idx, d_idx in enumerate(depth_indices):
        t_slice = target[d_idx]        # Guaranteed 2D shape: (101, 241)
        p_slice = pred[d_idx]          # Guaranteed 2D shape: (101, 241)
        err_slice = np.abs(p_slice - t_slice)

        v_min = np.nanmin(t_slice)
        v_max = np.nanmax(t_slice)

        # Ground Truth Target
        im0 = axes[row_idx, 0].imshow(t_slice, cmap="viridis", vmin=v_min, vmax=v_max, origin="lower")
        axes[row_idx, 0].set_title(f"Target Temp (°C) [Depth Level {d_idx}]")
        plt.colorbar(im0, ax=axes[row_idx, 0], fraction=0.046, pad=0.04)

        # Model Prediction
        im1 = axes[row_idx, 1].imshow(p_slice, cmap="viridis", vmin=v_min, vmax=v_max, origin="lower")
        axes[row_idx, 1].set_title(f"Predicted Temp (°C) [Depth Level {d_idx}]")
        plt.colorbar(im1, ax=axes[row_idx, 1], fraction=0.046, pad=0.04)

        # Absolute Error
        im2 = axes[row_idx, 2].imshow(err_slice, cmap="Reds", origin="lower")
        axes[row_idx, 2].set_title(f"Absolute Error (°C) [Depth Level {d_idx}]")
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