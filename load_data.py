"""
Memory-Optimized load_data.py

Prereqs:
    pip install xarray scipy netCDF4 pandas dask

Usage:
    python load_data.py
"""

import gc
import glob
import xarray as xr
import pandas as pd
import numpy as np

from config import (
    SUBDIRS, PROCESSED_DIR,
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, GRID_RESOLUTION_DEG,
)


def open_all(key: str) -> xr.Dataset:
    """Open NetCDF files for a given key using lazy dask streaming."""
    files = sorted(glob.glob(str(SUBDIRS[key] / "*.nc")))
    if not files:
        raise FileNotFoundError(
            f"No files found for '{key}' in {SUBDIRS[key]} — run the matching collect_*.py first."
        )
    
    # Stream in small temporal chunks while keeping spatial axes intact for interpolation
    ds = xr.open_mfdataset(
        files, 
        combine="by_coords", 
        chunks={"time": 10, "lat": -1, "lon": -1, "latitude": -1, "longitude": -1}
    )

    # Drop dummy 1D depth coordinate if present in surface inputs
    if "depth" in ds.dims and ds.dims["depth"] == 1:
        ds = ds.squeeze("depth", drop=True)
    elif "depth" in ds.coords and "depth" not in ds.dims:
        ds = ds.drop_vars("depth", errors="ignore")

    # Standardize time coordinate to daily midnight timestamps
    if "time" in ds.coords:
        time_vals = pd.to_datetime([str(t)[:10] for t in ds.time.values])
        ds = ds.assign_coords(time=time_vals)

    return ds


def standardize_coords_and_regrid(
    ds: xr.Dataset, target_lats: np.ndarray, target_lons: np.ndarray
) -> xr.Dataset:
    """Standardize spatial coordinate names to ('lat', 'lon'), regrid, and cast to float32."""
    rename_dict = {}
    all_keys = set(ds.dims) | set(ds.coords) | set(ds.variables)
    
    for c in all_keys:
        if c.lower() in ["latitude", "lat_0", "y", "latitudes"] and c != "lat":
            rename_dict[c] = "lat"
        elif c.lower() in ["longitude", "lon_0", "x", "longitudes"] and c != "lon":
            rename_dict[c] = "lon"

    if rename_dict:
        ds = ds.rename(rename_dict)

    for dim in ["lat", "lon"]:
        if dim in ds.dims and dim not in ds.indexes:
            ds = ds.set_index({dim: dim})

    if "lat" in ds.dims:
        ds = ds.sortby("lat")
    if "lon" in ds.dims:
        ds = ds.sortby("lon")

    # Interpolate onto common target grid
    ds_r = ds.interp(lat=target_lats, lon=target_lons, method="linear")

    # Convert float64 variables to float32 to cut RAM footprint by 50%
    for var in ds_r.data_vars:
        if ds_r[var].dtype == np.float64:
            ds_r[var] = ds_r[var].astype(np.float32)

    return ds_r


def main():
    target_lats = np.arange(LAT_MIN, LAT_MAX + GRID_RESOLUTION_DEG, GRID_RESOLUTION_DEG)
    target_lons = np.arange(LON_MIN, LON_MAX + GRID_RESOLUTION_DEG, GRID_RESOLUTION_DEG)

    surface_vars = {}
    for key in ["sst", "sss", "ssh", "currents", "winds"]:
        print(f"[load] processing {key} ...")
        ds = open_all(key)
        
        # Aggregate sub-daily data (e.g. 6-hourly winds) to daily means lazily
        ds = ds.groupby("time").mean(dim="time")
        
        ds_r = standardize_coords_and_regrid(ds, target_lats, target_lons)
        
        t_min = str(ds_r.time.values.min())[:10]
        t_max = str(ds_r.time.values.max())[:10]
        print(f"       -> {key}: {len(ds_r.time)} days ({t_min} to {t_max})")
        
        surface_vars[key] = ds_r
        
        # Force Python garbage collector to free unreferenced chunks
        gc.collect()

    print("[load] merging surface inputs ...")
    merged_inputs = xr.merge(list(surface_vars.values()), join="inner")
    print(f"[load] merged surface inputs dims: {dict(merged_inputs.dims)}")

    # GLORYS target — regrid horizontally, keep native depth axis
    print("[load] opening glorys (target) ...")
    glorys = open_all("glorys")
    glorys = glorys.groupby("time").mean(dim="time")
    glorys_r = standardize_coords_and_regrid(glorys, target_lats, target_lons)

    # Align inputs and target on exact shared days
    common_times = np.intersect1d(merged_inputs.time.values, glorys_r.time.values)
    merged_inputs = merged_inputs.sel(time=common_times)
    glorys_r = glorys_r.sel(time=common_times)
    print(f"[load] {len(common_times)} common days after alignment")

    inputs_path = PROCESSED_DIR / "surface_inputs.nc"
    target_path = PROCESSED_DIR / "glorys_target.nc"

    # Write output stream to NetCDF using NetCDF4 engine compression
    print(f"[load] writing {inputs_path} ...")
    merged_inputs.to_netcdf(inputs_path)
    
    print(f"[load] writing {target_path} ...")
    glorys_r.to_netcdf(target_path)
    
    print("[load] finished successfully.")


if __name__ == "__main__":
    main()