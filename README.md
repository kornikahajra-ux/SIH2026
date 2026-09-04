# 🌊 OceanEmbed: 3D Ocean Subsurface Temperature Reconstruction

**OceanEmbed** is a deep learning framework designed to reconstruct 3D ocean subsurface temperature profiles (0.5m down to 902.3m across 35 discrete native depth levels, or vertically interpolated to the 15 INCOIS standard depths) directly from 12 multi-modal satellite surface observation channels. Built with PyTorch and integrated into an interactive Streamlit dashboard, OceanEmbed bridges satellite altimetry, thermometry, and surface dynamics with subsurface oceanography.

---

## 📌 Key Features

- **Multi-Modal Satellite Synthesis**: Leverages 12 distinct sea-surface channels (thermal, salinity, height anomaly, geostrophic currents, surface currents, wind vectors, and data density flags).
- **3D Subsurface Reconstruction**: Predicts 35 non-linear vertical ocean depth temperature layers (0.5m to 902.3m) natively, with support for 15 INCOIS standard depth benchmark mapping.
- **`OceanUNet` Architecture**: Multi-scale encoder-decoder CNN designed specifically for continuous spatial fields, residual skip connections, and vertical profile predictions.
- **Dual Depth Pipeline**: Supports evaluation and prediction across both 35 Native GLORYS Depth Levels and 15 INCOIS Standard Benchmark Depths via command-line flags.
- **Interactive Web Dashboard**: Streamlit interface featuring interactive 3D depth slice sliders, thermocline gradient estimations, cross-sectional temperature transects, dual-metric visualization, and customizable land-masking themes.
- **In-Situ ARGO Validation**: Direct comparison and validation routines against independent in-situ ARGO float profiles.

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
│                Subsurface Depth Temperature Mapping                      │
│      Native 35 Depths (0.5m - 902.3m) OR 15 INCOIS Benchmark Levels      │
└────────────────────────────────────┬────────────────────────────────────┘
                                      │
                  ┌───────────────────┴───────────────────┐
                  ▼                                       ▼
┌───────────────────────────────────┐     ┌───────────────────────────────┐
│    Interactive Dashboard (app.py) │     │  Performance & Static Visuals  │
│ - Depth Map Slices (Plotly)       │     │   - metrics_vs_depth.png       │
│ - Vertical Profile Explorer       │     │   - spatial_evaluations.png    │
│ - Dual Metric Selection (15 vs 35)│     │   - evaluation_metrics*.csv    │
└───────────────────────────────────┘     └───────────────────────────────┘
```

---

## 📡 Input Channel Specification

The `OceanUNet` encoder processes a 12-channel surface feature tensor `(B, 12, H, W)`:

| Channel | Variable | Description | Unit |
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
| **11** | `n_obs` | Observation Count / Data Density Flag | Count |

---

## 📁 Repository Structure

```
SIH_OCEANEMBEDD/
├── checkpoints/
│   └── best_ocean_unet.pth          # Trained OceanUNet PyTorch weights & stats
├── data/                            # Processed NetCDF inputs and split files
├── app.py                           # Interactive Streamlit dashboard
├── model.py                         # PyTorch OceanUNet network definition
├── dataset.py                       # PyTorch DataLoaders, normalization & land masking
├── evaluate.py                      # Multi-depth metric evaluation script
├── predict.py                       # 3D inference script (Native vs INCOIS modes)
├── plot_evaluations.py              # Publication figure generator (metrics & spatial slices)
├── collect_argo.py                  # In-situ ARGO float data collector
├── validate_argo.py                 # ARGO float validation routines
├── config.py                        # Hyperparameters, depth grids, and spatial bounds
├── evaluation_metrics.csv           # Performance metrics across 35 Native GLORYS depths
├── evaluation_metrics_15incois.csv  # Performance metrics across 15 INCOIS benchmark depths
├── metrics_vs_depth.png             # Vertical metric profile output figure
├── spatial_evaluations.png          # Spatial slice comparison output figure
└── requirements.txt                 # Python dependency configuration
```

---

## 📊 Evaluation & Validation Metrics

Model performance is evaluated vertically across ocean layers using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Bias, and Pearson Correlation (r).

### ARGO Pearson Correlation (r)

The Pearson correlation coefficient quantifies structural alignment and shape reconstruction between predicted temperature profiles and observed in-situ ARGO profiles across depth levels:

```
        Σ (Pᵢ − P̄)(Oᵢ − Ō)
r = ────────────────────────────
     √Σ(Pᵢ − P̄)²  √Σ(Oᵢ − Ō)²
```

Where `Pᵢ` is the predicted temperature at depth step `i`, `Oᵢ` is the observed ground-truth temperature, and `P̄`, `Ō` represent layer mean values. An overall score of **r > 0.92** confirms accurate thermocline gradient slope tracking.

---

## 🚀 Pipeline Execution Order

Execute scripts in exact numerical sequence to populate evaluation CSVs, generate NetCDF prediction files, produce static figures, and run the UI dashboard.

### Step 1: Run Quantitative Evaluation

Calculates metrics across both 35 Native GLORYS levels and 15 INCOIS standard depths:

```bash
python evaluate.py
```

Outputs generated: `evaluation_metrics.csv` and `evaluation_metrics_15incois.csv`.

### Step 2: Generate 3D Subsurface Predictions

Run inference to output vertical predictions. By default, `--mode incois` interpolates predictions to the 15 standard INCOIS depths:

```bash
python predict.py --mode incois
```

(Optional: Run `python predict.py --mode native` to output raw 35-depth physical layers.)

Output generated: `predicted_subsurface_temp.nc`.

### Step 3: Produce Evaluation Plots

Automates static visual rendering of error curves and spatial slices:

```bash
python plot_evaluations.py
```

Outputs generated: `metrics_vs_depth.png` and `spatial_evaluations.png`.

### Step 4: Launch Interactive Streamlit Dashboard

Launch the web interface for 3D exploration and metric toggle views:

```bash
streamlit run app.py
```

Open local browser at `http://localhost:8501`.

---

## 📈 Model Performance Highlights

- **Upper Ocean / Surface Layer (0–100m)**: High profile fidelity with RMSE < 0.80°C and Pearson correlation r > 0.92.
- **Thermocline Region (100–300m)**: Accurately detects maximum temperature gradient change (dT/dz) and tracks steep vertical stratification.
- **Deep Ocean (300–900m+)**: Low absolute error variance, maintaining structural stability in deep water masses.

---

## 📜 Acknowledgments

Developed for the Smart India Hackathon (SIH). Operational dataset integration includes Copernicus Marine Environment Monitoring Service (CMEMS), NASA PO.DAAC, and the International ARGO Program.
