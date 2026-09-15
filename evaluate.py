"""
Evaluation and Metrics Framework for OceanEmbed.
Evaluates model performance on the test set across all 35 native depth levels
by assembling predictions from the thermocline and remaining-depths submodels,
and generates INCOIS standard 15-depth benchmark evaluation metrics.

Usage:
    python evaluate.py
    python evaluate.py --thermocline_checkpoint checkpoints/best_ocean_unet_thermocline.pth \
                       --remaining_checkpoint checkpoints/best_ocean_unet_remaining.pth
"""

import argparse
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from dataset import get_dataloaders
from model import (
    build_thermocline_model,
    build_remaining_depths_model,
    assemble_full_depth_profile,
)
from config import NATIVE_DEPTHS_35, INCOIS_STANDARD_DEPTHS_M, verify_native_depths

# Anchor file lookups to script location
BASE_DIR = Path(__file__).parent
DEFAULT_THERMO_CHECKPOINT = BASE_DIR / "checkpoints" / "best_ocean_unet_thermocline.pth"
DEFAULT_REMAINING_CHECKPOINT = BASE_DIR / "checkpoints" / "best_ocean_unet_remaining.pth"


def evaluate_model(thermo_ckpt_path: str = None, remaining_ckpt_path: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for evaluation: {device}")

    verify_native_depths()

    thermo_ckpt_path = Path(thermo_ckpt_path) if thermo_ckpt_path else DEFAULT_THERMO_CHECKPOINT
    remaining_ckpt_path = Path(remaining_ckpt_path) if remaining_ckpt_path else DEFAULT_REMAINING_CHECKPOINT

    if not thermo_ckpt_path.exists():
        raise FileNotFoundError(
            f"Thermocline checkpoint not found at {thermo_ckpt_path}. "
            "Train the models first using train.py."
        )
    if not remaining_ckpt_path.exists():
        raise FileNotFoundError(
            f"Remaining-depths checkpoint not found at {remaining_ckpt_path}. "
            "Train the models first using train.py."
        )

    print(f"Loading thermocline checkpoint from {thermo_ckpt_path}...")
    ckpt_thermo = torch.load(thermo_ckpt_path, map_location=device, weights_only=False)

    print(f"Loading remaining-depths checkpoint from {remaining_ckpt_path}...")
    ckpt_remaining = torch.load(remaining_ckpt_path, map_location=device, weights_only=False)

    # Determine custom output suffix if non-default checkpoints are passed
    is_default_thermo = thermo_ckpt_path.resolve() == DEFAULT_THERMO_CHECKPOINT.resolve()
    is_default_remaining = remaining_ckpt_path.resolve() == DEFAULT_REMAINING_CHECKPOINT.resolve()
    suffix = (
        ""
        if (is_default_thermo and is_default_remaining)
        else f"_{thermo_ckpt_path.stem}_{remaining_ckpt_path.stem}"
    )

    # Instantiate both split submodels
    model_thermo = build_thermocline_model().to(device)
    model_remaining = build_remaining_depths_model().to(device)

    try:
        model_thermo.load_state_dict(ckpt_thermo["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load '{thermo_ckpt_path.name}' into the thermocline submodel (model.py). "
            f"Original error: {e}"
        ) from e

    try:
        model_remaining.load_state_dict(ckpt_remaining["model_state_dict"])
    except RuntimeError as e:
        raise RuntimeError(
            f"Failed to load '{remaining_ckpt_path.name}' into the remaining-depths submodel (model.py). "
            f"Original error: {e}"
        ) from e

    model_thermo.eval()
    model_remaining.eval()

    stats = ckpt_thermo.get("stats", {})
    _, _, test_loader, _ = get_dataloaders(batch_size=4)

    all_preds, all_targets, all_masks = [], [], []

    print("Running inference on test split using submodel ensemble...")
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device)
            pred_thermo = model_thermo(x)
            pred_remaining = model_remaining(x)

            # Reassemble full 35-depth profile from the split outputs
            preds = assemble_full_depth_profile(pred_thermo, pred_remaining)

            all_preds.append(preds.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            all_masks.append(mask.cpu().numpy())

    preds_arr = np.concatenate(all_preds, axis=0)
    targets_arr = np.concatenate(all_targets, axis=0)
    masks_arr = np.concatenate(all_masks, axis=0)

    # Reverse Z-score normalization if training stats are saved
    if "target_mean" in stats and "target_std" in stats:
        t_mean, t_std = stats["target_mean"], stats["target_std"]
        preds_arr = preds_arr * t_std + t_mean
        targets_arr = targets_arr * t_std + t_mean

    n_depths = len(NATIVE_DEPTHS_35)

    if masks_arr.ndim == 3:
        print(
            "[evaluate.py] WARNING: Received 2D mask (B, H, W). "
            "Falling back to uniform depth expansion."
        )
        masks_expanded = np.repeat(np.expand_dims(masks_arr, axis=1), n_depths, axis=1) == 1.0
    else:
        masks_expanded = masks_arr == 1.0

    print(f"\nCalculating metrics across {n_depths} native GLORYS depth levels...")
    metrics_per_depth = []

    for depth_idx in range(n_depths):
        valid = masks_expanded[:, depth_idx, :, :]
        p_depth = preds_arr[:, depth_idx, :, :][valid]
        t_depth = targets_arr[:, depth_idx, :, :][valid]

        if len(t_depth) == 0:
            continue

        rmse = np.sqrt(np.mean((p_depth - t_depth) ** 2))
        mae = np.mean(np.abs(p_depth - t_depth))
        bias = np.mean(p_depth - t_depth)
        corr = (
            np.corrcoef(p_depth, t_depth)[0, 1]
            if np.std(p_depth) > 0 and np.std(t_depth) > 0
            else 0.0
        )

        metrics_per_depth.append({
            "depth_index": depth_idx,
            "depth_m": NATIVE_DEPTHS_35[depth_idx],
            "n_valid_cells": int(valid.sum()),
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "pearson_r": corr,
        })

    df_metrics = pd.DataFrame(metrics_per_depth)
    out_csv = BASE_DIR / f"evaluation_metrics{suffix}.csv"
    df_metrics.to_csv(out_csv, index=False)

    # Vertical interpolation to 15 INCOIS standard depths
    max_native_depth = max(NATIVE_DEPTHS_35)

    interp_preds = interp1d(
        NATIVE_DEPTHS_35, preds_arr, axis=1, bounds_error=False, fill_value="extrapolate"
    )(INCOIS_STANDARD_DEPTHS_M)
    interp_targets = interp1d(
        NATIVE_DEPTHS_35, targets_arr, axis=1, bounds_error=False, fill_value="extrapolate"
    )(INCOIS_STANDARD_DEPTHS_M)

    interp_masks = interp1d(
        NATIVE_DEPTHS_35, masks_expanded.astype(float), axis=1,
        kind="nearest", bounds_error=False, fill_value="extrapolate",
    )(INCOIS_STANDARD_DEPTHS_M) >= 0.5

    incois_metrics = []
    for idx, d_m in enumerate(INCOIS_STANDARD_DEPTHS_M):
        valid = interp_masks[:, idx, :, :]
        p_d = interp_preds[:, idx, :, :][valid]
        t_d = interp_targets[:, idx, :, :][valid]
        if len(t_d) == 0:
            continue

        extrapolated = d_m > max_native_depth
        if extrapolated:
            print(
                f"[evaluate.py] NOTE: {d_m}m is beyond maximum native depth "
                f"({max_native_depth:.1f}m) - extrapolated row."
            )

        incois_metrics.append({
            "depth_index": idx,
            "depth_m": d_m,
            "extrapolated": extrapolated,
            "n_valid_cells": int(valid.sum()),
            "rmse": np.sqrt(np.mean((p_d - t_d) ** 2)),
            "mae": np.mean(np.abs(p_d - t_d)),
            "bias": np.mean(p_d - t_d),
            "pearson_r": (
                np.corrcoef(p_d, t_d)[0, 1]
                if np.std(p_d) > 0 and np.std(t_d) > 0
                else 0.0
            ),
        })

    incois_csv = BASE_DIR / f"evaluation_metrics_15incois{suffix}.csv"
    pd.DataFrame(incois_metrics).to_csv(incois_csv, index=False)

    print("\n================ Evaluation Summary ================")
    print(f"Overall Test RMSE : {df_metrics['rmse'].mean():.4f} °C")
    print(f"Overall Test MAE  : {df_metrics['mae'].mean():.4f} °C")
    print(f"Overall Pearson R : {df_metrics['pearson_r'].mean():.4f}")
    print(f"Saved Native metrics to: {out_csv.resolve()}")
    print(f"Saved INCOIS 15-depth metrics to: {incois_csv.resolve()}")
    print("====================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed Split Model Evaluation")
    parser.add_argument(
        "--thermocline_checkpoint",
        type=str,
        default=str(DEFAULT_THERMO_CHECKPOINT),
        help="Path to the thermocline submodel checkpoint",
    )
    parser.add_argument(
        "--remaining_checkpoint",
        type=str,
        default=str(DEFAULT_REMAINING_CHECKPOINT),
        help="Path to the remaining-depths submodel checkpoint",
    )
    args = parser.parse_args()
    evaluate_model(
        thermo_ckpt_path=args.thermocline_checkpoint,
        remaining_ckpt_path=args.remaining_checkpoint,
    )
