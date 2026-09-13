import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

# 1. Load the dataset globally ONCE when the server starts
DS_GLOBAL = xr.open_dataset('/content/predicted_subsurface_temp.nc')

def generate_isotherm_image(requested_time: str, target_temp: float = 20.0) -> str:
    """
    Calculates the depth of a specific isotherm (e.g., D20 at 20°C) across the grid 
    to track subsurface thermal eddies, returning a Base64 image string for the frontend.
    
    :param requested_time: ISO format string from frontend (e.g., '2023-04-15')
    :param target_temp: Temperature value to track in Celsius (default 20.0°C)
    :return: Base64 string of the generated PNG
    """
    
    # 2. Select the specific time requested by the user
    try:
        ds_current = DS_GLOBAL.sel(time=requested_time, method='nearest')
    except KeyError:
        return "Error: Requested time is out of dataset bounds."

    # 3. Core Math: D20 / Isotherm Depth Calculation
    temp_3d = ds_current.predicted_temperature  # Dimensions: (depth, lat, lon)
    depths = ds_current.depth.values             # 1D array of depths
    
    # Extract raw numpy arrays for fast vectorized interpolation
    t_vals = temp_3d.values  # Shape: (depth, lat, lon)
    
    # Initialize an output array for isotherm depths with NaNs
    lat_size, lon_size = t_vals.shape[1], t_vals.shape[2]
    d20_grid = np.full((lat_size, lon_size), np.nan)
    
    # Vectorized or grid-wise linear interpolation for the target isotherm
    for i in range(lat_size):
        for j in range(lon_size):
            profile = t_vals[:, i, j]
            if np.isnan(profile).all():
                continue
            
            # Find where the profile crosses the target temperature
            # We look for sign changes relative to target_temp
            diff = profile - target_temp
            sign_change = np.where(np.diff(np.signbit(diff)))[0]
            
            if len(sign_change) > 0:
                idx = sign_change[0]
                # Linear interpolation between depth[idx] and depth[idx+1]
                t1, t2 = profile[idx], profile[idx+1]
                d1, d2 = depths[idx], depths[idx+1]
                
                if t1 != t2:
                    d20_grid[i, j] = d1 + (target_temp - t1) * (d2 - d1) / (t2 - t1)

    # Wrap the resulting 2D numpy array back into an xarray DataArray to preserve coordinates
    isotherm_da = xr.DataArray(
        d20_grid,
        coords={'lat': ds_current.lat, 'lon': ds_current.lon},
        dims=['lat', 'lon']
    )

    # 4. Generate the Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Isotherm Depth
    # 'coolwarm_r' highlights anomalies: shallow depths (upwelling/cyclonic cores) vs deep depths (anticyclonic cores)
    isotherm_da.plot(
        ax=ax,
        cmap='coolwarm_r',
        add_colorbar=True,
        cbar_kwargs={'label': f'{target_temp}°C Isotherm Depth (meters)'}
    )
    
    # Clean up formatting for the user UI
    actual_date = ds_current.time.dt.strftime('%Y-%m-%d').values
    plt.title(f"Subsurface Eddy Proxy: D{int(target_temp)} Isotherm Depth\nValid for: {actual_date}", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    # 5. Convert Plot to Base64 for Frontend Delivery
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig) # Frees server memory
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"