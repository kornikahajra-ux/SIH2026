"""
Fetch ARGO profiles for the study region/period, for use as an INDEPENDENT
validation set (do not train on this — GLORYS is your training target;
ARGO is what you check the final model against).

Prereqs:
    pip install argopy "erddapy<3.0.0"

Usage:
    python collect_argo.py
"""
import ssl
import aiohttp

# Bypass SSL certificate verification for both standard library and aiohttp connectors
ssl._create_default_https_context = ssl._create_unverified_context

_orig_tcp_init = aiohttp.TCPConnector.__init__
def _unverified_tcp_init(self, *args, **kwargs):
    kwargs['ssl'] = False
    _orig_tcp_init(self, *args, **kwargs)
aiohttp.TCPConnector.__init__ = _unverified_tcp_init

from argopy import DataFetcher
import xarray as xr

from config import (
    SUBDIRS, LON_MIN, LON_MAX, LAT_MIN, LAT_MAX,
    START_DATE, END_DATE, STANDARD_DEPTHS_M,
)


def fetch_argo() -> xr.Dataset:
    region = [
        LON_MIN, LON_MAX,
        LAT_MIN, LAT_MAX,
        min(STANDARD_DEPTHS_M), max(STANDARD_DEPTHS_M),
        START_DATE, END_DATE,
    ]
    print(f"[argo] fetching region {region}")
    ds = DataFetcher(src="erddap").region(region).to_xarray()
    return ds


def main():
    ds = fetch_argo()
    out_path = SUBDIRS["argo"] / f"argo_{START_DATE}_{END_DATE}.nc"
    ds.to_netcdf(out_path)
    print(f"[argo] saved {ds.dims.get('N_POINTS', 'N/A')} points -> {out_path}")


if __name__ == "__main__":
    main()