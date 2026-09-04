# 🌊 OceanEmbed: 3D Ocean Subsurface Temperature Reconstruction

**OceanEmbed** is a deep learning framework designed to reconstruct 3D ocean subsurface temperature profiles (0.5m down to 902.3m across 35 discrete depth levels) directly from 12 multi-modal satellite surface observation channels. Built with PyTorch and integrated into an interactive Streamlit dashboard, OceanEmbed bridges satellite altimetry, thermometry, and surface dynamics with subsurface oceanography.

---

## 📌 Key Features

- **Multi-Modal Satellite Synthesis**: Leverages 12 distinct sea-surface channels (thermal, salinity, height anomaly, geostrophic currents, surface currents, wind vectors, and data density flags).
- **3D Subsurface Reconstruction**: Simultaneously predicts 35 non-linear vertical ocean depth temperature layers (0.5m to 902.3m).
- **`OceanUNet` Architecture**: Custom multi-scale encoder-decoder CNN designed specifically for continuous spatial fields and vertical profile predictions.
- **Interactive Web Dashboard**: Streamlit-based web interface featuring interactive 3D depth slice sliders, cross-sectional temperature transects, performance metrics, and ARGO float validation.
- **In-Situ ARGO Validation**: Direct comparison and validation routines against independent in-situ ARGO float profiles.
- **Physical Coordinate Mapping**: Native tracking and visualization using physical units (meters for depth, °C for temperature, m/s for velocity).

---

## 🏗️ System Architecture & Data Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      12 Surface Input Predictors                        │
│  (SST, SSS, SLA, ADT, u, v, u_g, v_g, u_wnd, v_wnd, ws, n_obs)           │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │ Shape: (Batch, 12, Lat, Lon)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                           OceanUNet Model                                │
│  Multi-Scale Encoder-Decoder CNN with Skip Connections (in=12, out=35)   │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │ Shape: (Batch, 35, Lat, Lon)
                                      ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                35 Subsurface Depth Temperature Layers                    │
│               Physical Depth Grid: 0.5m ───► 902.3m                      │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
┌───────────────────────────────────┐     ┌───────────────────────────────┐
│    Interactive Dashboard (app.py) │     │  Performance & Static          │
│  - Spatial 2D Maps (Plotly)       │     │  Evaluation Visuals            │
│  - Vertical Transect Views        │     │   - metrics_vs_depth.png       │
│  - ARGO Float Comparisons         │     │   - spatial_evaluations.png    │
└───────────────────────────────────┘     └───────────────────────────────┘
```

---

## 📡 Input Channel Specification (12 Surface Features)

The `OceanUNet` encoder takes a 12-channel surface feature tensor `(B, 12, H, W)` defined as follows:

| Channel Index | Variable | Description | Physical Unit |
| :---: | :--- | :--- | :---: |
| **0** | `SST` | Sea Surface Temperature | °C |
| **1** | `SSS` | Sea Surface Salinity | PSU |
| **2** | `SLA` | Sea Level Anomaly | m |
| **3** | `ADT` | Absolute Dynamic Topography | m |
| **4** | `u` | Total Zonal Surface Current (East-West) | m/s |
| **5** | `v` | Total Meridional Surface Current (North-South) | m/s |
| **6** | `u_g` | Geostrophic Zonal Current Component | m/s |
| **7** | `v_g` | Geostrophic Meridional Current Component | m/s |
| **8** | `u_wnd` | Zonal Surface Wind Speed | m/s |
| **9** | `v_wnd` | Meridional Surface Wind Speed | m/s |
| **10** | `ws` | Total Surface Wind Speed Magnitude | m/s |
| **11** | `n_obs` | Observation Count / Quality Density Flag | Count |

---

## 📁 Repository Structure & File Overview

```
SIH_OCEANEMBEDD/
├── checkpoints/
│   └── best_ocean_unet.pth          # Trained OceanUNet PyTorch weights & stats
├── data/                            # Raw and preprocessed input data files
├── venv/                            # Project virtual environment
├── app.py                           # Streamlit interactive web dashboard
├── model.py                         # PyTorch OceanUNet network definition
├── dataset.py                       # Data loading, normalization & PyTorch DataLoaders
├── evaluate.py                      # Quantitative evaluation script (computes RMSE, MAE, Pearson r)
├── predict.py                       # Inference script outputting NetCDF predictions
├── plot_evaluations.py              # Script generating depth profile & spatial evaluation plots
├── plot_data.py                     # Data visualization utilities
├── inspect_data.py                  # Dataset inspection & structural analysis
├── load_data.py                     # Raw dataset loader utilities
├── collect_argo.py                  # In-situ ARGO float data collector
├── collect_cmems.py                 # Copernicus CMEMS satellite data fetcher
├── collect_podaac.py                # NASA PO.DAAC data fetcher
├── validate_argo.py                 # Validation routine comparing model predictions vs ARGO profiles
├── test.py                          # Utility script attaching depth_m coordinates to metrics
├── config.py                        # Project hyperparameters, grid parameters, and paths
├── evaluation_metrics.csv           # Layer-by-layer evaluation metrics with physical depth (depth_m)
├── INPUT_CHANNELS.md                # Detailed specification of input surface tensor channels
├── PROJECT_CONTEXT.md               # Context documentation and system specs
├── README.md                        # Primary project documentation
└── requirements.txt                 # Python dependency requirements list
```

### Detailed Script Descriptions

- **`app.py`**: Main application interface built with Streamlit and Plotly. Features 3 core tabs:
  1. *Subsurface Temperature Explorer*: Interactive 2D spatial map sliders for depth selection and vertical transect depth profile lines.
  2. *ARGO Validation*: In-situ ARGO float profile comparison vs model predictions.
  3. *Architecture & Metrics*: Model overview cards, layer-by-layer metrics table, and error/correlation profile charts.
- **`model.py`**: Defines `OceanUNet`, a custom PyTorch convolutional UNet architecture (`in_channels=12`, `out_channels=35`) utilizing double-convolution blocks, max-pooling encoders, transposed convolution decoders, and residual skip connections.
- **`dataset.py`**: Manages dataset loading from NetCDF/NumPy formats, feature normalization (mean/std), land-mask handling, and PyTorch `DataLoader` generation for training, validation, and testing splits.
- **`evaluate.py`**: Evaluates model performance across all test samples and depth levels, calculating Layer-wise Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Bias, and Pearson Correlation Coefficients (r), saving results to `evaluation_metrics.csv`.
- **`predict.py`**: Executes inference over input domains and exports predicted 3D temperature fields into structured NetCDF files (`predicted_subsurface_temp.nc`).
- **`plot_evaluations.py`**: Automates generation of publication-quality static figures:
  - `metrics_vs_depth.png`: RMSE, MAE, and Pearson r plotted against physical depth (meters).
  - `spatial_evaluations.png`: Spatial target vs prediction vs absolute error slices across Surface, Thermocline, and Deep layers.
- **`validate_argo.py`**: Fetches and aligns real-world ARGO float profiles with model coordinate predictions for independent validation.

---

## 🚀 Getting Started

### 1. Prerequisites & Environment Setup

Ensure Python 3.10+ is installed. Clone the repository and set up the virtual environment:

```bash
# Navigate to project directory
cd SIH_OCEANEMBEDD

# Activate virtual environment
# On Windows (Git Bash):
source venv/Scripts/activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
```

**Note on Dependencies**: If non-fatal `argopy` or `erddapy` import warnings appear in your terminal during execution, you can suppress or resolve them by updating erddapy:

```bash
pip install "erddapy<2.0.0"
```

### 2. Running the Pipeline

**Run Quantitative Evaluation**

Calculate performance metrics across all 35 ocean depth levels:

```bash
python evaluate.py
```

This generates `evaluation_metrics.csv` containing `depth_index`, `depth_m`, `rmse`, `mae`, `bias`, and `pearson_r`.

**Generate Static Evaluation Visuals**

Produce high-resolution evaluation figures for reports or presentations:

```bash
python plot_evaluations.py
```

Outputs saved:
- `metrics_vs_depth.png`
- `spatial_evaluations.png`

**Launch Interactive Streamlit Dashboard**

Start the local dashboard server:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` to explore interactive subsurface ocean predictions.

---

## 📈 Model Performance Highlights

- **Upper Ocean / Surface Layer (0-100m)**: High accuracy with RMSE < 0.80°C and Pearson r > 0.92.
- **Thermocline Region (100-300m)**: Captures complex steep thermal gradients effectively.
- **Deep Ocean (300-900m+)**: Maintains strong performance stability with minimal absolute error as temperature variance decreases with depth.

---

## 📜 License & Acknowledgments

Developed for the Smart India Hackathon (SIH). Data sources include Copernicus Marine Environment Monitoring Service (CMEMS), NASA PO.DAAC, and the International ARGO Program.
