import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

# 1. Load the dataset globally ONCE when the server starts
# Replace with your actual path
DS_GLOBAL = xr.open_dataset('/content/predicted_subsurface_temp.nc')

def generate_thermocline_image(requested_time: str) -> str:
    """
    Calculates the Thermocline depth for a specific time and returns a Base64 image string.
    
    :param requested_time: ISO format string from frontend (e.g., '2023-04-15')
    :return: Base64 string of the generated PNG
    """
    
    # 2. Select the specific time requested by the user
    try:
        ds_current = DS_GLOBAL.sel(time=requested_time, method='nearest')
    except KeyError:
        return "Error: Requested time is out of dataset bounds."

    # 3. Core Math (Thermocline Depth Calculation)
    temp_3d = ds_current.predicted_temperature
    
    # Differentiate along the 'depth' axis to find the temperature gradient (dT/dz)
    vertical_gradient = temp_3d.differentiate('depth')
    
    # Find the depth where the gradient is most negative (sharpest drop in temperature)
    # .idxmin() automatically returns the coordinate value (the depth in meters)
    thermocline_depth = vertical_gradient.idxmin(dim='depth')

    # 4. Generate the Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the Thermocline Depth
    # 'YlGnBu' (Yellow-Green-Blue) is standard for depth mapping. 
    # Shallow thermoclines will be yellow/light, deep thermoclines will be dark blue.
    thermocline_depth.plot(
        ax=ax,
        cmap='YlGnBu',
        add_colorbar=True,
        cbar_kwargs={'label': 'Thermocline Depth (meters)'}
    )
    
    # Clean up formatting for the user UI
    actual_date = ds_current.time.dt.strftime('%Y-%m-%d').values
    plt.title(f"Ocean Thermocline Depth\nValid for: {actual_date}", fontsize=14)
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.tight_layout()

    # 5. Convert Plot to Base64 for Frontend Delivery
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig) # Crucial: frees memory on the server
    buf.seek(0)
    
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return f"data:image/png;base64,{image_base64}"