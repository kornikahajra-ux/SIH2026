# 🌊 OceanEmbed: 3D Ocean Subsurface Temperature Reconstruction

**OceanEmbed** is a deep learning framework designed to reconstruct 3D ocean subsurface temperature profiles (0.5m down to 902.3m across 35 discrete native depth levels, or vertically interpolated to the 15 INCOIS standard depths) directly from 12 multi-modal satellite surface observation channels. Built with PyTorch and integrated into an interactive Streamlit dashboard, OceanEmbed bridges satellite altimetry, thermometry, and surface dynamics with subsurface oceanography.

---

## 📌 Key Features

- **Multi-Modal Satellite Synthesis**: Leverages 12 distinct sea-surface channels (thermal, salinity, height anomaly, geostrophic currents, surface currents, wind vectors, and data density flags).
- **3D Subsurface Reconstruction**: Predicts 35 non-linear vertical ocean depth temperature layers (0.5m to 902.3m) natively, with support for 15 INCOIS standard depth benchmark mapping.
- **Split Thermocline / Remaining-Depths Architecture**: Instead of one network predicting all 35 depth levels at once, the depth column is divided into two disjoint bands, each trained by its own independently-checkpointed `OceanUNet` instance — a **thermocline submodel** (depth indices 17–24, 8 levels) and a **remaining-depths submodel** (depth indices 0–16 + 25–34, 27 levels). This isolates the fastest-varying, hardest-to-predict part of the profile so it can be trained/tuned/checkpointed independently of the rest of the column.
- **`OceanUNet` Architecture**: A shared multi-scale encoder-decoder CNN backbone (identical for both submodels, differing only in the final 1×1 projection head's channel count) designed for continuous spatial fields, residual skip connections, and vertical profile predictions.
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
                     ┌────────────────┴────────────────┐
                     ▼                                  ▼
┌───────────────────────────────────┐   ┌───────────────────────────────────┐
│   Thermocline OceanUNet Submodel   │   │  Remaining-Depths OceanUNet       │
│   in=12, out=8  (depths 17–24)     │   │  Submodel · in=12, out=27         │
│                                     │   │  (depths 0–16 + 25–34)            │
└────────────────────────────────────┘   └───────────────────────────────────┘
                     │ (B, 8, Lat, Lon)                 │ (B, 27, Lat, Lon)
                     └────────────────┬─────────────────┘
                                       ▼
                     assemble_full_depth_profile() [model.py]
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

Both submodels share the exact same `OceanUNet` encoder/decoder code and are trained together in a single pass over the dataloader in `train.py` (one optimizer step per batch, with the two masked-MSE losses summed before `backward()`), but they share no parameters and are checkpointed to separate files, so either can later be retrained or swapped independently. `evaluate.py` and `predict.py` load both checkpoints and use `assemble_full_depth_profile()` to reconstruct a single 35-depth column before computing metrics or writing outputs — downstream code never needs to know the model is split in two.

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

## 🌡️ Depth-Band Split

`model.py` defines the split as a set of fixed index constants, checked against each other at import time so the two bands always reconstruct the full 35-level column:

| Submodel | Depth Indices | Levels | Notes |
| :--- | :--- | :---: | :--- |
| **Thermocline** | 17–24 | 8 | Fastest-varying, hardest-to-predict band |
| **Remaining-depths** | 0–16 and 25–34 | 27 | Shallow surface layers + deep, low-variance layers |

`split_full_depth_profile()` slices a full `(B, 35, H, W)` target or mask tensor into the two submodels' inputs, and `assemble_full_depth_profile()` performs the inverse, reassembling both submodels' predictions back into native-depth order. Every training, evaluation, and prediction script calls these two functions rather than re-deriving the index math.

`MaskedMSELoss` (also in `model.py`) is depth-count agnostic — the same loss class is used for the 8-level thermocline tensors, the 27-level remaining-depths tensors, or a full 35-level tensor, and accepts either a legacy 2D `(B, H, W)` mask broadcast across all depth channels, or a depth-aware `(B, Depth, H, W)` mask already sliced to match.

---

## 📁 Repository Structure

```
SIH_OCEANEMBEDD/
├── checkpoints/
│   ├── best_ocean_unet_thermocline.pth   # Thermocline submodel weights & stats (depths 17-24)
│   └── best_ocean_unet_remaining.pth     # Remaining-depths submodel weights & stats (depths 0-16, 25-34)
├── data/                                 # Processed NetCDF inputs and split files
├── app.py                                # Interactive Streamlit dashboard
├── model.py                              # OceanUNet definition, depth-band split/assemble helpers, MaskedMSELoss
├── dataset.py                            # PyTorch DataLoaders, normalization & land masking
├── train.py                              # Joint training loop for both submodels
├── evaluate.py                           # Multi-depth metric evaluation script (loads both checkpoints)
├── predict.py                            # 3D inference script (Native vs INCOIS modes)
├── plot_evaluations.py                   # Publication figure generator (metrics & spatial slices)
├── collect_argo.py                       # In-situ ARGO float data collector
├── validate_argo.py                      # ARGO float validation routines
├── config.py                             # Hyperparameters, depth grids, and spatial bounds
├── training_history.csv                  # Per-epoch train/val loss history (combined + per-submodel)
├── evaluation_metrics.csv                # Performance metrics across 35 Native GLORYS depths
├── evaluation_metrics_15incois.csv       # Performance metrics across 15 INCOIS benchmark depths
├── metrics_vs_depth.png                  # Vertical metric profile output figure
├── spatial_evaluations.png               # Spatial slice comparison output figure
└── requirements.txt                      # Python dependency configuration
```

---

## 📊 Evaluation & Validation Metrics

Model performance is evaluated vertically across ocean layers using Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), Bias, and Pearson Correlation (r), computed only over valid (unmasked) ocean cells at each depth.

### ARGO Pearson Correlation (r)

The Pearson correlation coefficient quantifies structural alignment and shape reconstruction between predicted temperature profiles and observed in-situ ARGO profiles across depth levels:

```
        Σ (Pᵢ − P̄)(Oᵢ − Ō)
r = ────────────────────────────
     √Σ(Pᵢ − P̄)²  √Σ(Oᵢ − Ō)²
```

Where `Pᵢ` is the predicted temperature at depth step `i`, `Oᵢ` is the observed ground-truth temperature, and `P̄`, `Ō` represent layer mean values. An overall score of **r > 0.92** confirms accurate thermocline gradient slope tracking.

### Latest measured results

From the current `evaluation_metrics.csv` / `evaluation_metrics_15incois.csv` (reassembled 35-depth predictions from both submodels):

| Depth Band | RMSE (°C) | MAE (°C) | Pearson r |
| :--- | :---: | :---: | :---: |
| Surface (0.5m native) | ~0.40 | ~0.30 | ~0.97 |
| INCOIS 100m | ~0.94 | ~0.69 | ~0.92 |
| INCOIS 500m | ~0.31 | ~0.22 | ~0.97 |
| INCOIS 700m | ~0.33 | ~0.24 | ~0.97 |

Depths beyond the maximum native GLORYS depth (e.g. the 1000m INCOIS row) are flagged `extrapolated=True` in `evaluation_metrics_15incois.csv` and should be interpreted with additional caution, since they fall outside the native 35-level grid and rely on `scipy.interpolate.interp1d`'s extrapolation.

---

## 🚀 Pipeline Execution Order

Execute scripts in exact numerical sequence to populate evaluation CSVs, generate NetCDF prediction files, produce static figures, and run the UI dashboard.

### Step 1: Train Both Submodels

Trains the thermocline and remaining-depths submodels together in a single loop over the dataloader, saving each to its own checkpoint whenever combined validation loss improves:

```bash
python train.py
python train.py --epochs 150 --lr 5e-4 --patience 10
python train.py --thermocline_checkpoint_name thermo_v2.pth --remaining_checkpoint_name remaining_v2.pth
```

Outputs generated: `checkpoints/best_ocean_unet_thermocline.pth`, `checkpoints/best_ocean_unet_remaining.pth`, `training_history.csv`.

### Step 2: Run Quantitative Evaluation

Loads both submodel checkpoints, reassembles full 35-depth predictions on the test split, and calculates metrics across both the 35 Native GLORYS levels and the 15 INCOIS standard depths:

```bash
python evaluate.py
python evaluate.py --thermocline_checkpoint checkpoints/best_ocean_unet_thermocline.pth \
                    --remaining_checkpoint checkpoints/best_ocean_unet_remaining.pth
```

Outputs generated: `evaluation_metrics.csv` and `evaluation_metrics_15incois.csv`.

### Step 3: Generate 3D Subsurface Predictions

Run inference to output vertical predictions. By default, `--mode incois` interpolates predictions to the 15 standard INCOIS depths:

```bash
python predict.py --mode incois
```

(Optional: Run `python predict.py --mode native` to output raw 35-depth physical layers.)

Output generated: `predicted_subsurface_temp.nc`.

### Step 4: Produce Evaluation Plots

Automates static visual rendering of error curves and spatial slices:

```bash
python plot_evaluations.py
```

Outputs generated: `metrics_vs_depth.png` and `spatial_evaluations.png`.

### Step 5: Launch Interactive Streamlit Dashboard

Launch the web interface for 3D exploration and metric toggle views:

```bash
streamlit run app.py
```

Open local browser at `http://localhost:8501`.

---

## 📈 Model Performance Highlights

- **Upper Ocean / Surface Layer (0–100m)**: High profile fidelity with RMSE < 0.80°C and Pearson correlation r > 0.92.
- **Thermocline Region (100–300m, handled by the dedicated thermocline submodel)**: Accurately detects maximum temperature gradient change (dT/dz) and tracks steep vertical stratification, benefiting from being trained as an independent, focused submodel rather than sharing capacity with the rest of the column.
- **Deep Ocean (300–900m+, handled by the remaining-depths submodel)**: Low absolute error variance, maintaining structural stability in deep water masses.

---

## 📜 Acknowledgments

Developed for the Smart India Hackathon (SIH). Operational dataset integration includes Copernicus Marine Environment Monitoring Service (CMEMS), NASA PO.DAAC, and the International ARGO Program.
