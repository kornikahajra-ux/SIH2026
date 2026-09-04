"""
Split a SINGLE YEAR of surface_inputs.nc / glorys_target.nc into train/val/test.

Why not a simple chronological split (Jan-Aug train, Sep-Oct val, Nov-Dec test)?
The North Indian Ocean has a strong monsoon cycle. A simple chronological split
means test only ever sees late-year conditions and never validates against the
monsoon season -- that's a biased evaluation, not a fair one.

Instead: divide the year into 4 seasonal quarters, and within EACH quarter carve
out contiguous train/val/test blocks (with a purge gap between them). Every split
then sees the full seasonal cycle, while blocks stay contiguous in time so you
still avoid the leakage you'd get from a random day-by-day split.

Usage:
    python split_data.py
"""

import xarray as xr
import numpy as np
import pandas as pd

from config import PROCESSED_DIR

# ---- Seasonal quarters for the North Indian Ocean (edit to your actual year) ----
YEAR = 2023
QUARTERS = [
    ("winter_NE_monsoon", f"{YEAR}-01-01", f"{YEAR}-03-31"),
    ("pre_monsoon", f"{YEAR}-04-01", f"{YEAR}-06-15"),
    ("SW_monsoon", f"{YEAR}-06-16", f"{YEAR}-09-30"),
    ("post_monsoon", f"{YEAR}-10-01", f"{YEAR}-12-31"),
]

# Fraction of each quarter's days assigned to each split
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# Days dropped at each internal boundary to prevent autocorrelation leakage.
# Shorter than the multi-year version (10d) since a 1-year window can't afford
# to lose as many days per quarter -- 4 quarters x multiple boundaries adds up fast.
PURGE_DAYS = 4


def quarter_split(times: pd.DatetimeIndex) -> dict:
    """Given the sorted days in one quarter, return index arrays for train/val/test
    with a purge gap dropped at each internal boundary."""
    n = len(times)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    # test gets the remainder

    train_idx = np.arange(0, max(n_train - PURGE_DAYS, 0))
    val_start = n_train
    val_end = n_train + n_val
    val_idx = np.arange(val_start + PURGE_DAYS, max(val_end - PURGE_DAYS, val_start + PURGE_DAYS))
    test_idx = np.arange(val_end + PURGE_DAYS, n)

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def build_split_time_lists(ds: xr.Dataset) -> dict:
    """Walk each seasonal quarter, split it, and collect the resulting timestamps
    per split across all quarters."""
    all_times = pd.DatetimeIndex(ds.time.values)
    collected = {"train": [], "val": [], "test": []}

    for label, start, end in QUARTERS:
        q_times = all_times[(all_times >= start) & (all_times <= end)]
        q_times = q_times.sort_values()
        if len(q_times) == 0:
            print(f"[warn] no data found for quarter '{label}' ({start} to {end}) -- skipping")
            continue

        idx_map = quarter_split(q_times)
        for split_name, idxs in idx_map.items():
            collected[split_name].extend(q_times[idxs].tolist())
            print(f"[{label}/{split_name}] {len(idxs)} days")

    return collected


def split_and_save(name: str, ds: xr.Dataset, split_times: dict):
    for split_name, times in split_times.items():
        times_sorted = sorted(times)
        split_ds = ds.sel(time=times_sorted)
        out_path = PROCESSED_DIR / f"{name}_{split_name}.nc"
        split_ds.to_netcdf(out_path)
        print(f"[{name}/{split_name}] TOTAL {split_ds.dims.get('time', 0)} days -> {out_path}")


def main():
    inputs = xr.open_dataset(PROCESSED_DIR / "surface_inputs.nc")
    target = xr.open_dataset(PROCESSED_DIR / "glorys_target.nc")

    # Build the split using the inputs' time axis (assumes inputs/target already
    # share a common time axis, as produced by load_data.py's alignment step)
    split_times = build_split_time_lists(inputs)

    split_and_save("surface_inputs", inputs, split_times)
    split_and_save("glorys_target", target, split_times)


if __name__ == "__main__":
    main()
