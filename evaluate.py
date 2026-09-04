"""
Evaluation and Metrics Framework for OceanEmbed.
Evaluates model performance on the test set across all 35 depth levels.

Usage:
    python evaluate.py
"""

import warnings
warnings.filterwarnings("ignore")  # Suppress non-critical library warnings

from pathlib import Path
import numpy as np
import pandas as pd
import torch

from dataset import get_dataloaders
from model import OceanUNet


def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    checkpoint_path = Path("checkpoints/best_ocean_unet.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the model first using train.py.")

    print(f"Loading best checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Initialize model and load weights
    model = OceanUNet(in_channels=12, out_channels=35).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint.get("stats", {})

    # Load test dataloader
    _, _, test_loader, _ = get_dataloaders(batch_size=4)

    all_preds = []
    all_targets = []
    all_masks = []

    print("Running inference on test split...")
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device)
            preds = model(x)

            # Ensure batch dimension remains at axis 0: (B, 35, 101, 241)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    # Concatenate along batch dimension (axis 0) -> (N_test_days, 35, 101, 241)
    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    masks_arr = np.concatenate(all_masks, axis=0)

    # Re-scale target predictions if target standardization parameters exist
    if "target_mean" in stats and "target_std" in stats:
        t_mean = stats["target_mean"]
        t_std = stats["target_std"]
        preds_arr = preds_arr * t_std + t_mean
        targets_arr = targets_arr * t_std + t_mean

    # Expand 2D ocean mask (N, 101, 241) across 35 depth levels -> (N, 35, 101, 241)
    masks_expanded = np.repeat(np.expand_dims(masks_arr, axis=1), 35, axis=1) == 1.0

    print("\nCalculating metrics across 35 vertical depth levels...")
    metrics_per_depth = []

    for depth_idx in range(35):
        # Isolate valid ocean pixels for current depth
        p_depth = preds_arr[:, depth_idx, :, :][masks_expanded[:, depth_idx, :, :]]
        t_depth = targets_arr[:, depth_idx, :, :][masks_expanded[:, depth_idx, :, :]]

        if len(t_depth) == 0:
            continue

        rmse = np.sqrt(np.mean((p_depth - t_depth) ** 2))
        mae = np.mean(np.abs(p_depth - t_depth))
        bias = np.mean(p_depth - t_depth)

        if np.std(p_depth) > 0 and np.std(t_depth) > 0:
            corr = np.corrcoef(p_depth, t_depth)[0, 1]
        else:
            corr = 0.0

        metrics_per_depth.append({
            "depth_index": depth_idx,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "pearson_r": corr
        })

    df_metrics = pd.DataFrame(metrics_per_depth)
    out_csv = Path("evaluation_metrics.csv")
    df_metrics.to_csv(out_csv, index=False)

    print("\n================ Evaluation Summary ================")
    print(f"Overall Test RMSE : {df_metrics['rmse'].mean():.4f} °C")
    print(f"Overall Test MAE  : {df_metrics['mae'].mean():.4f} °C")
    print(f"Overall Pearson R : {df_metrics['pearson_r'].mean():.4f}")
    print(f"Saved depth metrics to: {out_csv.resolve()}")
    print("====================================================\n")

    print("Sample Depth Layer Performance (First 5 Levels):")
    print(df_metrics.head(5).to_string(index=False))


if __name__ == "__main__":
    evaluate_model()