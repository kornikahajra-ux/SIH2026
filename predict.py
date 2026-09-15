"""
OceanEmbed Operational Inference Engine.
Generates 3D subsurface ocean temperature profiles from 2D surface input NetCDF files.

Usage:
    python predict.py
    python predict.py --input data/processed/surface_inputs_test.nc --output predicted_subsurface_temp.nc
"""

import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import torch
from scipy.interpolate import interp1d

from model import (
    build_thermocline_model,
    build_remaining_depths_model,
    assemble_full_depth_profile,
)
from config import PROCESSED_DIR, NATIVE_DEPTHS_35, INCOIS_STANDARD_DEPTHS_M

BASE_DIR = Path(__file__).parent
DEFAULT_THERMO_CHECKPOINT = BASE_DIR / "checkpoints" / "best_ocean_unet_thermocline.pth"
DEFAULT_REMAINING_CHECKPOINT = BASE_DIR / "checkpoints" / "best_ocean_unet_remaining.pth"


def get_coord(ds: xr.Dataset, possible_names: list):
    """Retrieve coordinate array from dataset matching key variations."""
    for name in possible_names:
        if name in ds.coords or name in ds.data_vars:
            return ds[name].values
    raise KeyError(
        f"None of {possible_names} found in dataset keys: {list(ds.keys()) + list(ds.coords)}"
    )


def extract_channel_array(
    ds: xr.Dataset, var_name: str, target_shape: tuple
) -> np.ndarray:
    """Extracts a variable DataArray and transposes/coerces it strictly to target_shape (n_times, n_lats, n_lons)."""
    n_times, n_lats, n_lons = target_shape

    if var_name not in ds.data_vars and var_name not in ds.coords:
        return np.zeros((n_times, n_lats, n_lons), dtype=np.float32)

    da = ds[var_name].squeeze()
    time_dim = next(
        (d for d in da.dims if "time" in str(d).lower() or str(d).lower() == "t"),
        None,
    )
    lat_dim = next(
        (d for d in da.dims if "lat" in str(d).lower() or str(d).lower() == "y"),
        None,
    )
    lon_dim = next(
        (d for d in da.dims if "lon" in str(d).lower() or str(d).lower() == "x"),
        None,
    )

    if time_dim is None:
        da = da.expand_dims("time")
        time_dim = "time"

    if lat_dim and lon_dim:
        da = da.transpose(time_dim, lat_dim, lon_dim)

    arr = da.values.astype(np.float32)

    if arr.ndim == 2:
        arr = np.repeat(arr[None, :, :], n_times, axis=0)

    if arr.ndim == 3 and arr.shape == (n_times, n_lons, n_lats):
        arr = np.swapaxes(arr, 1, 2)

    if arr.shape[0] != n_times:
        arr = arr[:n_times]

    if arr.shape != (n_times, n_lats, n_lons):
        arr = np.broadcast_to(arr, (n_times, n_lats, n_lons)).copy()

    return arr.astype(np.float32)


def run_inference(
    input_path: str,
    output_path: str,
    thermo_checkpoint_path: str = None,
    remaining_checkpoint_path: str = None,
    target_depth_mode: str = "incois",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[+] Using device for inference: {device}")

    thermo_file = Path(thermo_checkpoint_path) if thermo_checkpoint_path else DEFAULT_THERMO_CHECKPOINT
    remaining_file = Path(remaining_checkpoint_path) if remaining_checkpoint_path else DEFAULT_REMAINING_CHECKPOINT

    if not thermo_file.is_absolute():
        thermo_file = BASE_DIR / thermo_file
    if not remaining_file.is_absolute():
        remaining_file = BASE_DIR / remaining_file

    if not thermo_file.exists():
        raise FileNotFoundError(f"Thermocline checkpoint not found at {thermo_file}")
    if not remaining_file.exists():
        raise FileNotFoundError(f"Remaining depths checkpoint not found at {remaining_file}")

    print(f"[+] Loading thermocline checkpoint: {thermo_file.name}")
    ckpt_thermo = torch.load(thermo_file, map_location=device, weights_only=False)

    print(f"[+] Loading remaining-depths checkpoint: {remaining_file.name}")
    ckpt_remaining = torch.load(remaining_file, map_location=device, weights_only=False)

    model_thermo = build_thermocline_model().to(device)
    model_remaining = build_remaining_depths_model().to(device)

    model_thermo.load_state_dict(ckpt_thermo["model_state_dict"])
    model_remaining.load_state_dict(ckpt_remaining["model_state_dict"])

    model_thermo.eval()
    model_remaining.eval()

    stats = ckpt_thermo.get("stats", {})

    input_file = Path(input_path)
    if not input_file.is_absolute():
        input_file = BASE_DIR / input_file

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found at {input_file}")

    print(f"[+] Reading surface inputs from: {input_file.name}")
    ds_in = xr.open_dataset(input_file)

    lats = get_coord(ds_in, ["lat", "latitude", "LATITUDE", "y"])
    lons = get_coord(ds_in, ["lon", "longitude", "LONGITUDE", "x"])
    times = (
        ds_in["time"].values
        if "time" in ds_in
        else np.array([np.datetime64("now")])
    )

    target_shape = (len(times), len(lats), len(lons))
    feature_vars = [
        "analysed_sst",
        "sos",
        "sla",
        "adt",
        "u",
        "v",
        "ug",
        "vg",
        "uwnd",
        "vwnd",
        "ws",
        "nobs",
    ]

    tensor_channels = [
        extract_channel_array(ds_in, var, target_shape) for var in feature_vars
    ]
    input_array = np.stack(tensor_channels, axis=1)

    if "input_mean" in stats and "input_std" in stats:
        in_mean = np.array(stats["input_mean"]).reshape(1, 12, 1, 1)
        in_std = np.array(stats["input_std"]).reshape(1, 12, 1, 1)
        input_array = (input_array - in_mean) / (in_std + 1e-7)

    input_array = np.nan_to_num(input_array, nan=0.0)
    input_array = np.squeeze(input_array)
    if input_array.ndim == 3:
        input_array = input_array[None, ...]

    print(f"[+] Input tensor shape prepared: {input_array.shape}")
    print(f"[+] Running 3D prediction for {input_array.shape[0]} timestep(s)...")

    inputs_tensor = torch.from_numpy(input_array).float().to(device)
    with torch.no_grad():
        pred_thermo = model_thermo(inputs_tensor)
        pred_remaining = model_remaining(inputs_tensor)
        preds_35_tensor = assemble_full_depth_profile(pred_thermo, pred_remaining)
        preds_35 = preds_35_tensor.cpu().numpy()

    if "target_mean" in stats and "target_std" in stats:
        t_mean = np.array(stats["target_mean"]).reshape(1, -1, 1, 1)
        t_std = np.array(stats["target_std"]).reshape(1, -1, 1, 1)
        preds_35 = preds_35 * t_std + t_mean

    # Interpolate to INCOIS 15 Standard Depths or native 35
    if target_depth_mode == "incois":
        interp_func = interp1d(
            NATIVE_DEPTHS_35,
            preds_35,
            axis=1,
            bounds_error=False,
            fill_value="extrapolate",
        )
        final_preds = interp_func(INCOIS_STANDARD_DEPTHS_M)
        final_depths = INCOIS_STANDARD_DEPTHS_M
    else:
        final_preds = preds_35
        final_depths = NATIVE_DEPTHS_35

    out_ds = xr.Dataset(
        data_vars={
            "predicted_temperature": (
                ("time", "depth", "lat", "lon"),
                final_preds.astype(np.float32),
                {
                    "long_name": "Reconstructed Subsurface Ocean Temperature",
                    "units": "degrees_C",
                    "standard_name": "sea_water_potential_temperature",
                },
            )
        },
        coords={
            "time": times,
            "depth": (
                "depth",
                final_depths,
                {"units": "m", "positive": "down"},
            ),
            "lat": ("lat", lats, {"units": "degrees_north"}),
            "lon": ("lon", lons, {"units": "degrees_east"}),
        },
        attrs={
            "title": "OceanEmbed 3D Ocean Temperature Reconstructions",
            "institution": "INCOIS / SIH 2026",
            "source": "OceanUNet Dual Submodel Deep Learning Engine",
            "resolution": "0.25 degree spatial",
        },
    )

    out_file = Path(output_path)
    if not out_file.is_absolute():
        out_file = BASE_DIR / out_file

    out_ds.to_netcdf(out_file)
    print(f"[+] Output successfully saved to: {out_file.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OceanEmbed 3D Temperature Prediction Engine"
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED_DIR / "surface_inputs_test.nc"),
        help="Input NetCDF path",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predicted_subsurface_temp.nc",
        help="Output NetCDF path",
    )
    parser.add_argument(
        "--thermocline_checkpoint",
        type=str,
        default=str(DEFAULT_THERMO_CHECKPOINT),
        help="Thermocline submodel checkpoint path",
    )
    parser.add_argument(
        "--remaining_checkpoint",
        type=str,
        default=str(DEFAULT_REMAINING_CHECKPOINT),
        help="Remaining-depths submodel checkpoint path",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["incois", "native"],
        default="incois",
        help="Target depth levels output format",
    )

    args = parser.parse_args()
    run_inference(
        args.input,
        args.output,
        args.thermocline_checkpoint,
        args.remaining_checkpoint,
        args.mode,
    )
