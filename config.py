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

# Native GLORYS Model Depths (35 physical levels used by OceanUNet)
NATIVE_DEPTHS_35 = [
    0.494, 1.541, 2.645, 3.819, 5.074, 6.424, 7.879, 9.452, 11.159, 13.014,
    15.034, 17.230, 19.617, 22.210, 25.025, 28.080, 31.397, 34.996, 38.902, 43.140,
    47.733, 52.707, 58.092, 63.920, 70.225, 77.045, 84.417, 92.381, 101.000, 110.330,
    120.440, 131.400, 143.280, 156.160, 902.300
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
