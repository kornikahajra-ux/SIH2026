import xarray as xr
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import io
import base64

# 1. Load the dataset globally ONCE when the server starts (saves API response time)
# Replace with your actual path
DS_GLOBAL = xr.open_dataset('ocean_data.nc')

def generate_pfz_image(requested_time: str, front_threshold: float = 0.5) -> str:
    """
    Calculates PFZ for a specific time and returns a Base64 image string for the frontend.
    
    :param requested_time: ISO format string from frontend (e.g., '2023-04-15')
    :param front_threshold: Sensitivity for edge detection
    :return: Base64 string of the generated PNG
    """
    
    # 2. Select the specific time requested by the user
    # method='nearest' prevents crashes if the frontend requests a time slightly off the exact hour/day
    try:
        ds_current = DS_GLOBAL.sel(time=requested_time, method='nearest')
    except KeyError:
        return "Error: Requested time is out of dataset bounds."

    # 3. Core Math (PFZ & Upwelling)
    sst = ds_current.predicted_temperature.sel(depth=0)
    t_30m = ds_current.predicted_temperature.sel(depth=30)
    
    # Upwelling Index
    upwelling_index = sst - t_30m
    
    # Fronts (Edge Detection)
    grad_lat = sst.differentiate('lat')
    grad_lon = sst.differentiate('lon')
    sst_gradient = np.sqrt(grad_lat**2 + grad_lon**2)
    thermal_fronts = sst_gradient.where(sst_gradient > front_threshold)

    # 4. Generate the Visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Upwelling
    upwelling_index.plot(
        ax=ax,
        cmap='Blues_r',
        add_colorbar=True,
        cbar_kwargs={'label': 'Upwelling Index (°C)'}
    )
    
    # Overlay Fronts
    thermal_fronts.plot(
        ax=ax,
        cmap='autumn',
        add_colorbar=True,
        alpha=0.9,
        cbar_kwargs={'label': 'Thermal Front Strength'}
    )
    
    # Clean up formatting for the user UI
    actual_date = ds_current.time.dt.strftime('%Y-%m-%d').values
    plt.title(f"Potential Fishing Zones (PFZ)\nValid for: {actual_date}", fontsize=14)
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

# use this in frontend to render the image 
# <img src="data:image/png;base64,iVBORw0KGgo...[PASTE THE REST OF YOUR STRING HERE]" alt="Matplotlib Plot" />