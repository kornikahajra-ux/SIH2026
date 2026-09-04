"""
Central configuration for the OceanEmbed data collection pipeline.
Edit these values once; every collection/loading script imports from here.
"""

from pathlib import Path

# ---- Study region: North Indian Ocean ----
LON_MIN, LON_MAX = 45.0, 105.0
LAT_MIN, LAT_MAX = 5.0, 30.0

# ---- Time range ----
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# ---- Target grid / depths ----
GRID_RESOLUTION_DEG = 0.25

# INCOIS Problem Statement #01 Required Standard Depths (15 levels)
INCOIS_STANDARD_DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

# ---- FIXED ----
# The previous list jumped straight from 156.16m to 902.3m, silently
# skipping 9 real GLORYS levels (186 - 763m). That gap meant every depth
# label from index 10 onward no longer lined up with the true GLORYS
# vertical grid, and interp1d() in evaluate.py was effectively bridging
# mislabeled points to fill in the INCOIS 200/300/500/700m rows -
# producing flat, suspiciously "good" numbers that weren't measuring real
# model skill in that band.
#
# These are the standard GLORYS12 continuous z-levels (0.494m -> 902.339m).
# IMPORTANT: this is still a *label*, not a guarantee. It must exactly
# match the depth coordinate actually stored in glorys_target_*.nc (built
# by split_data.py / collect_cmems.py). Run verify_native_depths() below
# once after any change to data extraction, and before trusting
# evaluate.py's output, rather than assuming this hardcoded list is right.
NATIVE_DEPTHS_35 = [
    0.494, 1.541, 2.645, 3.819, 5.078, 6.441, 7.929, 9.573, 11.405, 13.467,
    15.810, 18.496, 21.599, 25.211, 29.445, 34.434, 40.344, 47.373, 55.764, 65.807,
    77.853, 92.326, 109.729, 130.666, 155.851, 186.126, 222.475, 266.040, 318.127, 380.213,
    453.938, 541.089, 643.567, 763.333, 902.339,
]

STANDARD_DEPTHS_M = INCOIS_STANDARD_DEPTHS_M

# ---- Local storage layout ----
DATA_ROOT = Path("./data")
RAW_DIR = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"

SUBDIRS = {
    "sst": RAW_DIR / "sst",
    "sss": RAW_DIR / "sss",
    "ssh": RAW_DIR / "ssh",
    "glorys": RAW_DIR / "glorys",
    "currents": RAW_DIR / "currents",
    "winds": RAW_DIR / "winds",
    "argo": RAW_DIR / "argo",
}

# ---- Copernicus Marine dataset IDs ----
CMEMS_DATASETS = {
    "sst": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
    "sss": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",
    "ssh": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D",
    "glorys": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
}

# ---- PO.DAAC short_names ----
PODAAC_DATASETS = {
    "currents": "OSCAR_L4_OC_FINAL_V2.0",
    "winds": "CCMP_WINDS_10M6HR_L4_V3.1",
}

for d in SUBDIRS.values():
    d.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def verify_native_depths(processed_dir: Path = PROCESSED_DIR, split: str = "train", atol: float = 0.5) -> bool:
    """
    Safety check: confirms NATIVE_DEPTHS_35 actually matches the 'depth'
    coordinate stored in the processed target NetCDF file, instead of
    silently trusting a hardcoded list (which is exactly how the previous
    156m -> 902m gap slipped through unnoticed).

    Run this once after any change to data extraction/splitting, and again
    before trusting evaluate.py's per-depth metrics.
    """
    import numpy as np
    import xarray as xr

    target_path = processed_dir / f"glorys_target_{split}.nc"
    if not target_path.exists():
        print(f"[verify_native_depths] Skipped - {target_path} not found.")
        return False

    ds = xr.open_dataset(target_path)
    if "depth" not in ds.coords and "depth" not in ds.dims:
        print("[verify_native_depths] No 'depth' coordinate found in target file - cannot verify.")
        return False

    actual_depths = np.asarray(ds["depth"].values, dtype=float)
    expected_depths = np.asarray(NATIVE_DEPTHS_35, dtype=float)

    if actual_depths.shape[0] != expected_depths.shape[0]:
        print(
            f"[verify_native_depths] MISMATCH: config lists {expected_depths.shape[0]} depths, "
            f"but {target_path.name} has {actual_depths.shape[0]}."
        )
        return False

    if not np.allclose(actual_depths, expected_depths, atol=atol):
        print("[verify_native_depths] MISMATCH: depth values differ from config.NATIVE_DEPTHS_35:")
        for i, (a, e) in enumerate(zip(actual_depths, expected_depths)):
            if abs(a - e) > atol:
                print(f"    index {i}: file={a:.3f}m  config={e:.3f}m")
        return False

    print("[verify_native_depths] OK - config depths match the processed target file.")
    return True


if __name__ == "__main__":
    verify_native_depths()
