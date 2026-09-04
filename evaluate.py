"""
Evaluation and Metrics Framework for OceanEmbed.
Evaluates model performance on the test set across all 35 native depth levels
and generates INCOIS standard 15-depth benchmark evaluation metrics.

Usage:
    python evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from dataset import get_dataloaders
from model import OceanUNet
from config import NATIVE_DEPTHS_35, INCOIS_STANDARD_DEPTHS_M


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    checkpoint_path = Path("checkpoints/best_ocean_unet.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first using train.py.")

    print(f"Loading best checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model = OceanUNet(in_channels=12, out_channels=35).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint.get("stats", {})
    _, _, test_loader, _ = get_dataloaders(batch_size=4)

    all_preds, all_targets, all_masks = [], [], []

    print("Running inference on test split...")
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device)
            preds = model(x)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    masks_arr = np.concatenate(all_masks, axis=0)

    if "target_mean" in stats and "target_std" in stats:
        t_mean, t_std = stats["target_mean"], stats["target_std"]
        preds_arr = preds_arr * t_std + t_mean
        targets_arr = targets_arr * t_std + t_mean

    masks_expanded = np.repeat(np.expand_dims(masks_arr, axis=1), 35, axis=1) == 1.0

    print("\nCalculating metrics across 35 native GLORYS depth levels...")
    metrics_per_depth = []

    for depth_idx in range(35):
        p_depth = preds_arr[:, depth_idx, :, :][masks_expanded[:, depth_idx, :, :]]
        t_depth = targets_arr[:, depth_idx, :, :][masks_expanded[:, depth_idx, :, :]]

        if len(t_depth) == 0:
            continue

        rmse = np.sqrt(np.mean((p_depth - t_depth) ** 2))
        mae = np.mean(np.abs(p_depth - t_depth))
        bias = np.mean(p_depth - t_depth)
        corr = np.corrcoef(p_depth, t_depth)[0, 1] if np.std(p_depth) > 0 and np.std(t_depth) > 0 else 0.0

        metrics_per_depth.append({
            "depth_index": depth_idx,
            "depth_m": NATIVE_DEPTHS_35[depth_idx],
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "pearson_r": corr
        })

    df_metrics = pd.DataFrame(metrics_per_depth)
    out_csv = Path("evaluation_metrics.csv")
    df_metrics.to_csv(out_csv, index=False)

    # Calculate 15 INCOIS Standard Depth metrics via vertical 1D interpolation
    interp_preds = interp1d(NATIVE_DEPTHS_35, preds_arr, axis=1, bounds_error=False, fill_value="extrapolate")(INCOIS_STANDARD_DEPTHS_M)
    interp_targets = interp1d(NATIVE_DEPTHS_35, targets_arr, axis=1, bounds_error=False, fill_value="extrapolate")(INCOIS_STANDARD_DEPTHS_M)
    masks_15 = np.repeat(np.expand_dims(masks_arr, axis=1), 15, axis=1) == 1.0

    incois_metrics = []
    for idx, d_m in enumerate(INCOIS_STANDARD_DEPTHS_M):
        p_d = interp_preds[:, idx, :, :][masks_15[:, idx, :, :]]
        t_d = interp_targets[:, idx, :, :][masks_15[:, idx, :, :]]
        if len(t_d) == 0:
            continue
        incois_metrics.append({
            "depth_index": idx,
            "depth_m": d_m,
            "rmse": np.sqrt(np.mean((p_d - t_d) ** 2)),
            "mae": np.mean(np.abs(p_d - t_d)),
            "bias": np.mean(p_d - t_d),
            "pearson_r": np.corrcoef(p_d, t_d)[0, 1] if np.std(p_d) > 0 and np.std(t_d) > 0 else 0.0
        })
    pd.DataFrame(incois_metrics).to_csv("evaluation_metrics_15incois.csv", index=False)

    print("\n================ Evaluation Summary ================")
    print(f"Overall Test RMSE : {df_metrics['rmse'].mean():.4f} °C")
    print(f"Overall Test MAE  : {df_metrics['mae'].mean():.4f} °C")
    print(f"Overall Pearson R : {df_metrics['pearson_r'].mean():.4f}")
    print(f"Saved Native metrics to: {out_csv.resolve()}")
    print(f"Saved INCOIS 15-depth metrics to: {Path('evaluation_metrics_15incois.csv').resolve()}")
    print("====================================================\n")


if __name__ == "__main__":
    evaluate_model()
