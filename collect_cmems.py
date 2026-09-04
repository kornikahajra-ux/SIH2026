"""
Download SST, SSS, SSH, and GLORYS subsurface temperature from Copernicus Marine.

Prereqs:
    pip install copernicusmarine
    copernicusmarine login          # one-time, prompts for free account credentials
    (sign up at https://data.marine.copernicus.eu if you don't have one)

Usage:
    python collect_cmems.py --dataset sst
    python collect_cmems.py --dataset all
"""

import argparse
import copernicusmarine

from config import (
    CMEMS_DATASETS, SUBDIRS,
    LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
    START_DATE, END_DATE, STANDARD_DEPTHS_M,
)

# Which variable(s) to pull per dataset — trim/extend as needed once you've
# checked the actual variable names in each product's metadata.
VARIABLES = {
    "sst": ["analysed_sst"],
    "sss": ["sos"],           # sea surface salinity
    "ssh": ["sla", "adt"],    # sea level anomaly + absolute dynamic topography
    "glorys": ["thetao"],     # 3D potential temperature — this is your training TARGET
}


def download_one(key: str):
    dataset_id = CMEMS_DATASETS[key]
    out_dir = SUBDIRS[key]

    kwargs = dict(
        dataset_id=dataset_id,
        variables=VARIABLES[key],
        minimum_longitude=LON_MIN,
        maximum_longitude=LON_MAX,
        minimum_latitude=LAT_MIN,
        maximum_latitude=LAT_MAX,
        start_datetime=START_DATE,
        end_datetime=END_DATE,
        output_directory=str(out_dir),
        output_filename=f"{key}_{START_DATE}_{END_DATE}.nc",
    )

    # GLORYS is 3D — subset the standard depth levels you actually need,
    # instead of pulling every model level.
    if key == "glorys":
        kwargs["minimum_depth"] = min(STANDARD_DEPTHS_M)
        kwargs["maximum_depth"] = max(STANDARD_DEPTHS_M)

    print(f"[{key}] requesting {dataset_id} -> {out_dir}")
    copernicusmarine.subset(**kwargs)
    print(f"[{key}] done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(CMEMS_DATASETS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    keys = list(CMEMS_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for key in keys:
        download_one(key)


if __name__ == "__main__":
    main()
