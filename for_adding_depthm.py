import pandas as pd
import xarray as xr

# Load your evaluation metrics file
df = pd.read_csv("evaluation_metrics.csv")
ds = xr.open_dataset("predicted_subsurface_temp.nc")

# Insert physical depth in meters as the 2nd column
df.insert(1, "depth_m", ds["depth"].values)
df.to_csv("evaluation_metrics.csv", index=False)
