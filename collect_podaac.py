"""
Download surface currents (OSCAR) and winds (CCMP) from NASA PO.DAAC.

Prereqs:
    pip install earthaccess
    Free NASA Earthdata account: https://urs.earthdata.nasa.gov/users/new
    Either run `earthaccess.login()` interactively once, or set env vars
    EARTHDATA_USERNAME / EARTHDATA_PASSWORD.

Usage:
    python collect_podaac.py --dataset currents
    python collect_podaac.py --dataset all
"""

import argparse
import earthaccess

from config import PODAAC_DATASETS, SUBDIRS, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, START_DATE, END_DATE


def download_one(key: str):
    short_name = PODAAC_DATASETS[key]
    out_dir = SUBDIRS[key]

    print(f"[{key}] searching {short_name} ...")
    results = earthaccess.search_data(
        short_name=short_name,
        temporal=(START_DATE, END_DATE),
        bounding_box=(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX),
    )
    print(f"[{key}] found {len(results)} granules — downloading to {out_dir}")
    earthaccess.download(results, str(out_dir))
    print(f"[{key}] done.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=list(PODAAC_DATASETS.keys()) + ["all"],
        default="all",
    )
    args = parser.parse_args()

    earthaccess.login()  # picks up env vars or prompts interactively

    keys = list(PODAAC_DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    for key in keys:
        download_one(key)


if __name__ == "__main__":
    main()
