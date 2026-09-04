"""
Independent ARGO Float Profile Validation for OceanEmbed.
Interpolates 3D grid predictions to held-out point-based ARGO float profile locations
using 3D spatial RegularGridInterpolator.

Usage:
    python validate_argo.py
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import torch
from scipy.interpolate import RegularGridInterpolator

from model import OceanUNet
from dataset import get_dataloaders
from config import PROCESSED_DIR, RAW_DIR


def get_coord(ds: xr.Dataset, possible_names: list):
    """Retrieve coordinate or variable array from dataset matching key variations."""
    for name in possible_names:
        if name in ds.coords or name in ds.data_vars:
            return ds[name].values
    raise KeyError(f"None of {possible_names} found in dataset keys: {list(ds.keys()) + list(ds.coords)}")


def validate_against_argo():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device for ARGO validation: {device}")

    checkpoint_path = Path("checkpoints/best_ocean_unet.pth")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint missing at {checkpoint_path}.")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = OceanUNet(in_channels=12, out_channels=35).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint.get("stats", {})

    # 1. Run inference on test split to generate full 3D prediction volume
    _, _, test_loader, _ = get_dataloaders(batch_size=4)
    all_preds = []

    print("[+] Running test set inference to reconstruct 3D temperature grid...")
    with torch.no_grad():
        for x, _, _ in test_loader:
            preds = model(x.to(device)).cpu().numpy()
            all_preds.append(preds)

    # Shape: (N_test_days, 35, 101, 241)
    preds_arr = np.concatenate(all_preds, axis=0)

    # Rescale predictions to physical temperature units (°C)
    if "target_mean" in stats and "target_std" in stats:
        t_mean = stats["target_mean"]
        t_std = stats["target_std"]
        if isinstance(t_mean, np.ndarray) and t_mean.ndim == 1:
            t_mean = t_mean[:, None, None]
            t_std = t_std[:, None, None]
        preds_arr = preds_arr * t_std + t_mean

    # Average predictions over test time-window to form mean 3D ocean state (35, 101, 241)
    mean_pred_3d = np.nanmean(preds_arr, axis=0)

    # 2. Extract grid coordinates
    test_inputs = xr.open_dataset(PROCESSED_DIR / "surface_inputs_test.nc")
    test_target = xr.open_dataset(PROCESSED_DIR / "glorys_target_test.nc")

    lats = get_coord(test_inputs, ["lat", "latitude", "LATITUDE", "y"])
    lons = get_coord(test_inputs, ["lon", "longitude", "LONGITUDE", "x"])
    
    if "depth" in test_target.coords or "depth" in test_target.data_vars:
        depths = test_target["depth"].values
    else:
        depths = np.linspace(0, 1000, 35)  # Standard 35 depth levels (0m to 1000m)

    # 3. Build 3D Interpolator (Depth, Lat, Lon)
    interpolator = RegularGridInterpolator(
        (depths, lats, lons),
        mean_pred_3d,
        bounds_error=False,
        fill_value=np.nan
    )

    # 4. Load independent ARGO float dataset
    argo_dir = RAW_DIR / "argo"
    argo_files = list(argo_dir.glob("*.nc")) if argo_dir.exists() else []

    if not argo_files:
        print(f"[!] No ARGO NetCDF files found in {argo_dir}. Skipping.")
        return

    argo_ds = xr.open_dataset(argo_files[0])
    print(f"[+] Loaded independent ARGO dataset: {argo_files[0].name}")

    argo_temp = get_coord(argo_ds, ["TEMP", "temp", "temperature", "TEMP_ADJUSTED"]).flatten()
    argo_lat = get_coord(argo_ds, ["LATITUDE", "latitude", "lat"]).flatten()
    argo_lon = get_coord(argo_ds, ["LONGITUDE", "longitude", "lon"]).flatten()

    try:
        argo_depth = get_coord(argo_ds, ["PRES", "pres", "pressure", "DEPTH", "depth"]).flatten()
    except KeyError:
        # If ARGO float format is 1D/2D without explicit pressure array, broadcast depth indices
        argo_depth = np.tile(depths[:len(argo_temp) // len(argo_lat)], len(argo_lat))

    # Match dimensions if arrays were broadcast/repeated
    if len(argo_lat) != len(argo_temp):
        argo_lat = np.repeat(argo_lat, len(argo_temp) // len(argo_lat))
        argo_lon = np.repeat(argo_lon, len(argo_temp) // len(argo_lon))

    # Filter valid ARGO points within domain
    valid_mask = (
        ~np.isnan(argo_temp) &
        (argo_lat >= lats.min()) & (argo_lat <= lats.max()) &
        (argo_lon >= lons.min()) & (argo_lon <= lons.max()) &
        (argo_depth >= depths.min()) & (argo_depth <= depths.max())
    )

    obs_temp = argo_temp[valid_mask]
    query_points = np.column_stack((argo_depth[valid_mask], argo_lat[valid_mask], argo_lon[valid_mask]))

    # 5. Interpolate model predictions to exact ARGO float point coordinates
    print("[+] Performing 3D spatial interpolation to float profile locations...")
    model_interp_temp = interpolator(query_points)

    # Filter out any NaN values from land masks
    final_mask = ~np.isnan(model_interp_temp)
    obs_final = obs_temp[final_mask]
    pred_final = model_interp_temp[final_mask]

    # Calculate actual metrics
    rmse = np.sqrt(np.mean((obs_final - pred_final) ** 2))
    mae = np.mean(np.abs(obs_final - pred_final))
    bias = np.mean(pred_final - obs_final)
    corr = np.corrcoef(obs_final, pred_final)[0, 1]

    print("\n================ Genuine ARGO Float Benchmark ================")
    print(f"Evaluated Float Profile Points : {len(obs_final):,}")
    print(f"Observed Temp Range            : {np.min(obs_final):.2f} °C to {np.max(obs_final):.2f} °C")
    print(f"Predicted Temp Range           : {np.min(pred_final):.2f} °C to {np.max(pred_final):.2f} °C")
    print(f"In-Situ Point-Wise RMSE        : {rmse:.4f} °C")
    print(f"In-Situ Point-Wise MAE         : {mae:.4f} °C")
    print(f"Mean Bias                      : {bias:.4f} °C")
    print(f"Pearson Correlation (r)        : {corr:.4f}")
    print("==============================================================\n")

    # Plot Genuine Scatter Comparison
    plt.figure(figsize=(8, 6))
    subsample = np.random.choice(len(obs_final), size=min(10000, len(obs_final)), replace=False)
    plt.scatter(obs_final[subsample], pred_final[subsample], alpha=0.3, color="teal", s=10, label="Subsurface Float Points")
    plt.plot([min(obs_final), max(obs_final)], [min(obs_final), max(obs_final)], "r--", linewidth=2, label="1:1 Perfect Match")
    plt.title(f"OceanEmbed vs Independent ARGO Floats (r = {corr:.3f})", fontsize=12, fontweight="bold")
    plt.xlabel("Observed ARGO Temperature (°C)")
    plt.ylabel("Predicted Ocean Temperature (°C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()

    out_plot = Path("argo_vs_model.png")
    plt.savefig(out_plot, dpi=300)
    plt.close()
    print(f"[+] Saved updated ARGO comparison plot to: {out_plot.resolve()}")


if __name__ == "__main__":
    validate_against_argo()