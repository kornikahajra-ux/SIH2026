"""
OceanEmbed Operational Inference Engine.
Generates 3D subsurface ocean temperature profiles from 2D surface input NetCDF files.

Usage:
    python predict.py --input data/processed/surface_inputs_test.nc --output predictions.nc
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import torch

from model import OceanUNet
from config import PROCESSED_DIR


def get_coord(ds: xr.Dataset, possible_names: list):
    """Retrieve coordinate array from dataset matching key variations."""
    for name in possible_names:
        if name in ds.coords or name in ds.data_vars:
            return ds[name].values
    raise KeyError(f"None of {possible_names} found in dataset keys: {list(ds.keys()) + list(ds.coords)}")


def extract_channel_array(ds: xr.Dataset, var_name: str, target_shape: tuple) -> np.ndarray:
    """Extracts a variable DataArray and transposes/coerces it strictly to target_shape (n_times, n_lats, n_lons)."""
    n_times, n_lats, n_lons = target_shape

    if var_name not in ds.data_vars and var_name not in ds.coords:
        return np.zeros((n_times, n_lats, n_lons), dtype=np.float32)

    da = ds[var_name].squeeze()  # Remove size-1 dimensions (e.g. depth=1)

    # Detect dimension names in xarray
    time_dim = next((d for d in da.dims if "time" in str(d).lower() or str(d).lower() == "t"), None)
    lat_dim = next((d for d in da.dims if "lat" in str(d).lower() or str(d).lower() == "y"), None)
    lon_dim = next((d for d in da.dims if "lon" in str(d).lower() or str(d).lower() == "x"), None)

    if time_dim is None:
        da = da.expand_dims("time")
        time_dim = "time"

    # Transpose xarray DataArray directly using named coordinates
    if lat_dim and lon_dim:
        da = da.transpose(time_dim, lat_dim, lon_dim)

    arr = da.values.astype(np.float32)

    # Handle single timestep repeat
    if arr.ndim == 2:
        arr = np.repeat(arr[None, :, :], n_times, axis=0)

    # Secondary check: if axes were transposed in numpy array shape (e.g. (n_times, n_lons, n_lats))
    if arr.ndim == 3 and arr.shape == (n_times, n_lons, n_lats):
        arr = np.swapaxes(arr, 1, 2)

    if arr.shape[0] != n_times:
        arr = arr[:n_times]

    # Final shape fallback check
    if arr.shape != (n_times, n_lats, n_lons):
        arr = np.broadcast_to(arr, (n_times, n_lats, n_lons)).copy()

    return arr.astype(np.float32)


def run_inference(input_path: str, output_path: str, checkpoint_path: str = "checkpoints/best_ocean_unet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device for inference: {device}")

    # 1. Load model checkpoint
    ckpt_file = Path(checkpoint_path)
    if not ckpt_file.exists():
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_file}")

    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    model = OceanUNet(in_channels=12, out_channels=35).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stats = checkpoint.get("stats", {})

    # 2. Load surface input dataset
    input_file = Path(input_path)
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found at {input_file}")

    print(f"[+] Reading surface inputs from: {input_file.name}")
    ds_in = xr.open_dataset(input_file)

    lats = get_coord(ds_in, ["lat", "latitude", "LATITUDE", "y"])
    lons = get_coord(ds_in, ["lon", "longitude", "LONGITUDE", "x"])
    times = ds_in["time"].values if "time" in ds_in else np.array([np.datetime64("now")])

    n_times = len(times)
    n_lats = len(lats)
    n_lons = len(lons)
    target_shape = (n_times, n_lats, n_lons)

    feature_vars = [
        "analysed_sst", "sos", "sla", "adt", "u", "v",
        "ug", "vg", "uwnd", "vwnd", "ws", "nobs"
    ]

    tensor_channels = []
    for var in feature_vars:
        data = extract_channel_array(ds_in, var, target_shape)
        tensor_channels.append(data)

    # Stack to target tensor shape (N_times, 12, H, W)
    input_array = np.stack(tensor_channels, axis=1)

    # Apply training normalization cleanly using explicit reshaping
    if "input_mean" in stats and "input_std" in stats:
        in_mean = np.array(stats["input_mean"]).reshape(1, 12, 1, 1)
        in_std = np.array(stats["input_std"]).reshape(1, 12, 1, 1)
        input_array = (input_array - in_mean) / (in_std + 1e-7)

    input_array = np.nan_to_num(input_array, nan=0.0)

    # Guarantee 4D shape (N_times, 12, H, W)
    input_array = np.squeeze(input_array)
    if input_array.ndim == 3:  # single timestep edge case (12, H, W)
        input_array = input_array[None, ...]

    print(f"[+] Input tensor shape prepared: {input_array.shape}")

    # 3. Model Inference
    print(f"[+] Running 3D prediction for {input_array.shape[0]} timestep(s)...")
    inputs_tensor = torch.from_numpy(input_array).float().to(device)

    with torch.no_grad():
        preds_tensor = model(inputs_tensor).cpu().numpy()  # (N_times, 35, H, W)

    # 4. Unnormalize target predictions
    if "target_mean" in stats and "target_std" in stats:
        t_mean = np.array(stats["target_mean"]).reshape(1, -1, 1, 1)
        t_std = np.array(stats["target_std"]).reshape(1, -1, 1, 1)
        preds_tensor = preds_tensor * t_std + t_mean

    target_ref = PROCESSED_DIR / "glorys_target_test.nc"
    if target_ref.exists():
        ds_ref = xr.open_dataset(target_ref)
        depths = ds_ref["depth"].values if "depth" in ds_ref else np.linspace(0, 1000, 35)
    else:
        depths = np.linspace(0, 1000, 35)

    # 5. Save output NetCDF file
    out_ds = xr.Dataset(
        data_vars={
            "predicted_temperature": (
                ("time", "depth", "lat", "lon"),
                preds_tensor.astype(np.float32),
                {
                    "long_name": "Reconstructed Subsurface Ocean Temperature",
                    "units": "degrees_C",
                    "standard_name": "sea_water_potential_temperature"
                }
            )
        },
        coords={
            "time": times,
            "depth": ("depth", depths, {"units": "m", "positive": "down"}),
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={
            "title": "OceanEmbed 3D Ocean Temperature Reconstructions",
            "institution": "INCOIS / SIH 2026",
            "source": "OceanUNet Deep Learning Model",
            "resolution": "0.25 degree spatial"
        }
    )

    out_file = Path(output_path)
    out_ds.to_netcdf(out_file)
    print(f"[+] Output successfully saved to: {out_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed 3D Temperature Prediction Engine")
    parser.add_argument("--input", type=str, default=str(PROCESSED_DIR / "surface_inputs_test.nc"), help="Input NetCDF path")
    parser.add_argument("--output", type=str, default="predicted_subsurface_temp.nc", help="Output NetCDF path")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_ocean_unet.pth", help="Model checkpoint path")

    args = parser.parse_args()
    run_inference(args.input, args.output, args.checkpoint)