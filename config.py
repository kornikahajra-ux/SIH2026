"""
Central configuration for the OceanEmbed data collection pipeline.
Edit these values once; every collection/loading script imports from here.
"""

from pathlib import Path

# ---- Study region: North Indian Ocean ----
LON_MIN, LON_MAX = 45.0, 105.0
LAT_MIN, LAT_MAX = 5.0, 30.0

# ---- Time range (adjust to what you actually want to train on) ----
# Single-year pilot window. Must match the YEAR in split_data.py -- if you
# change one, change the other, or the split script will find zero days
# for every quarter.
START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# ---- Target grid / depths ----
GRID_RESOLUTION_DEG = 0.25
STANDARD_DEPTHS_M = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

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
# NOTE: verified against https://data.marine.copernicus.eu catalog
"""CMEMS_DATASETS = {
    "sst": "METOFFICE-GLO-SST-L4-REP-OBS-SST",                  # OSTIA historical daily L4 SST
    "sss": "cmems_obs-mob_glo_phy-sal_my_multi-oi_P7D-c",         # SMOS/SMAP L4 multi-obs salinity
    "ssh": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", # DUACS L4 reprocessed SSH
    "glorys": "cmems_mod_glo_phy_my_0.083deg_P1D-m",           # GLORYS12V1 daily ocean reanalysis
}"""
# ---- Copernicus Marine dataset IDs ----
CMEMS_DATASETS = {
    "sst": "METOFFICE-GLO-SST-L4-REP-OBS-SST",
    "sss": "cmems_obs-mob_glo_phy-sss_my_multi_P1D",                   # Daily L4 SSS (contains variable 'sos')
    "ssh": "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D", # Daily SSH
    "glorys": "cmems_mod_glo_phy_my_0.083deg_P1D-m",                  # Daily reanalysis
}

# ---- PO.DAAC short_names (used with earthaccess) ----
"""PODAAC_DATASETS = {
    "currents": "OSCAR_L4_OC_FINAL_V2.0",
    "winds": "CMP_WINDS_10M6HR_L4_V3.1",
}"""
# ---- PO.DAAC short_names (used with earthaccess) ----
PODAAC_DATASETS = {
    "currents": "OSCAR_L4_OC_FINAL_V2.0",
    "winds": "CCMP_WINDS_10M6HR_L4_V3.1",  # Fixed missing 'C' prefix
}

for d in SUBDIRS.values():
    d.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)