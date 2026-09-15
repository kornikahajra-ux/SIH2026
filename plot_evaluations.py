"""
OceanEmbed INCOIS 15-Depth Evaluation & PPT Visualization Engine.

Evaluates trained split submodels (Thermocline & Remaining depths) on test data,
interpolates native profile outputs to the 15 INCOIS standard depth levels,
and generates publication- and PPT-ready CSV metrics, flowcharts, metric profiles,
residual/scatter error plots, spatial comparison maps, vertical thermocline profiles,
and executive KPI dashboards.

Usage:
    python evaluate.py
    python evaluate.py --output_dir evaluation_outputs
"""

import argparse
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless execution
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import interp1d

from dataset import get_dataloaders
from model import (
    build_thermocline_model,
    build_remaining_depths_model,
    assemble_full_depth_profile,
)
from config import NATIVE_DEPTHS_35, INCOIS_STANDARD_DEPTHS_M, verify_native_depths

BASE_DIR = Path(__file__).parent
DEFAULT_THERMO_CHECKPOINT = BASE_DIR / "checkpoints" / "best_ocean_unet_thermocline.pth"
DEFAULT_REMAINING_CHECKPOINT = BASE_DIR / "checkpoints" / "best_ocean_unet_remaining.pth"
DEFAULT_OUTPUT_DIR = BASE_DIR / "evaluation_outputs"


# ==============================================================================
# OUTPUT DIRECTORY MANAGEMENT
# ==============================================================================

def setup_output_directories(base_output_dir: Path) -> dict:
    """Creates directory structure for organizing evaluation artifacts."""
    subdirs = {
        "root": base_output_dir,
        "csv": base_output_dir / "csv",
        "flowcharts": base_output_dir / "flowcharts",
        "metrics": base_output_dir / "metrics",
        "errors": base_output_dir / "errors",
        "spatial": base_output_dir / "spatial",
        "profiles": base_output_dir / "profiles",
        "dashboards": base_output_dir / "dashboards",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


# ==============================================================================
# FLOWCHARTS & DIAGRAM GENERATORS
# ==============================================================================

def draw_styled_box(ax, x, y, w, h, title, text, bg_color="#EBF5FB", border_color="#2980B9"):
    """Draws a styled rounded rectangle card for pipeline flowcharts."""
    rect = patches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25",
        facecolor=bg_color, edgecolor=border_color, linewidth=2
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center", fontsize=10, fontweight="bold", color="#1A5276")
    ax.text(x + w / 2, y + h / 2 - 0.15, text, ha="center", va="center", fontsize=8.5, color="#2C3E50", multialignment="center")


def draw_styled_arrow(ax, x1, y1, x2, y2, label=""):
    """Draws an annotated directional arrow between flowchart blocks."""
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", lw=2, color="#34495E", mutation_scale=15)
    )
    if label:
        ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", va="bottom", fontsize=8, fontweight="bold", color="#5D6D7E")


def generate_pipeline_flowchart(out_dir: Path):
    """Generates the main end-to-end OceanEmbed evaluation pipeline flowchart."""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)

    draw_styled_box(ax, 0.4, 2.0, 2.0, 2.0, "Satellite Inputs", "Surface Observations\n(B, 12, H, W)\nSST, SSH, Winds", "#E8F8F5", "#16A085")
    draw_styled_box(ax, 2.9, 2.0, 2.0, 2.0, "Input Channels", "12 Channel\nMultispectral Tensor\nStacking & Masking", "#EBF5FB", "#2980B9")
    draw_styled_box(ax, 5.4, 2.0, 2.0, 2.0, "OceanUNet Model", "Split Architectures:\nThermocline + Remaining\nSubmodels", "#FEF9E7", "#F39C12")
    draw_styled_box(ax, 7.9, 2.0, 2.2, 2.0, "3D Subsurface Pred.", "3D Temperature\nProfile Assembly\n(B, 35, H, W)", "#F4ECF7", "#8E44AD")
    draw_styled_box(ax, 10.6, 2.0, 2.3, 2.0, "INCOIS Standard Depth", "15 Standard Depths\nVertical Interpolation\n0m to 1000m", "#EBF5FB", "#2980B9")
    draw_styled_box(ax, 13.4, 2.0, 2.2, 2.0, "Validation & Metrics", "ARGO / Ground Truth\nRMSE | MAE | Bias | r\nPPT Dashboard & Plots", "#EAECEE", "#7F8C8D")

    draw_styled_arrow(ax, 2.4, 3.0, 2.9, 3.0)
    draw_styled_arrow(ax, 4.9, 3.0, 5.4, 3.0)
    draw_styled_arrow(ax, 7.4, 3.0, 7.9, 3.0)
    draw_styled_arrow(ax, 10.1, 3.0, 10.6, 3.0)
    draw_styled_arrow(ax, 12.9, 3.0, 13.4, 3.0)

    plt.title("OceanEmbed End-to-End Subsurface Temperature Prediction Pipeline", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / "pipeline_flowchart.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_workflow_diagrams(out_dir: Path):
    """Generates dedicated model inference and validation workflow diagrams."""
    # 1. Model Inference Workflow
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)

    draw_styled_box(ax, 0.5, 1.5, 2.2, 2.0, "Test Dataset", "GLORYS Reanalysis\nSatellite Feed\nTest Samples", "#E8F8F5", "#16A085")
    draw_styled_box(ax, 3.2, 1.5, 2.2, 2.0, "Load Checkpoints", "Thermocline Ckpt\nRemaining Ckpt\nModel Weights & Stats", "#FEF9E7", "#F39C12")
    draw_styled_box(ax, 5.9, 1.5, 2.2, 2.0, "Model Inference", "GPU Feed Forward\nDual Submodel\nOutput Computation", "#EBF5FB", "#2980B9")
    draw_styled_box(ax, 8.6, 1.5, 2.2, 2.0, "Full Profile Assembly", "assemble_full_depth()\nReconstruct Native\n35-Depth Profile", "#F4ECF7", "#8E44AD")
    draw_styled_box(ax, 11.3, 1.5, 2.2, 2.0, "Un-Normalization", "Z-Score Reversal\nstd * Pred + mean\nDegree Celsius (°C)", "#EAECEE", "#7F8C8D")

    draw_styled_arrow(ax, 2.7, 2.5, 3.2, 2.5)
    draw_styled_arrow(ax, 5.4, 2.5, 5.9, 2.5)
    draw_styled_arrow(ax, 8.1, 2.5, 8.6, 2.5)
    draw_styled_arrow(ax, 10.8, 2.5, 11.3, 2.5)

    plt.title("OceanEmbed Model Inference & Reconstruction Workflow", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / "evaluation_workflow.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Validation & Interpolation Workflow
    fig, ax = plt.subplots(figsize=(14, 5), dpi=300)
    ax.axis("off")
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)

    draw_styled_box(ax, 0.5, 1.5, 2.2, 2.0, "Un-Normalized Profile", "Native 35 Depths\nPredicted & Target\nTensors (°C)", "#E8F8F5", "#16A085")
    draw_styled_box(ax, 3.2, 1.5, 2.4, 2.0, "15 INCOIS Depth Interp", "scipy.interpolate.interp1d\nStandard Levels:\n0m to 1000m", "#FEF9E7", "#F39C12")
    draw_styled_box(ax, 6.1, 1.5, 2.2, 2.0, "Land / Validity Masking", "Apply Land Mask\nNearest Interp Masking\nExclude Land Cells", "#EBF5FB", "#2980B9")
    draw_styled_box(ax, 8.8, 1.5, 2.2, 2.0, "Metric Computation", "RMSE, MAE, Bias\nPearson r\nPer 15 INCOIS Depths", "#F4ECF7", "#8E44AD")
    draw_styled_box(ax, 11.5, 1.5, 2.0, 2.0, "Artifact Exports", "15-Depth CSV\nPPT Visualizations\nKPI Dashboards", "#EAECEE", "#7F8C8D")

    draw_styled_arrow(ax, 2.7, 2.5, 3.2, 2.5)
    draw_styled_arrow(ax, 5.6, 2.5, 6.1, 2.5)
    draw_styled_arrow(ax, 8.3, 2.5, 8.8, 2.5)
    draw_styled_arrow(ax, 11.0, 2.5, 11.5, 2.5)

    plt.title("OceanEmbed INCOIS 15-Depth Mapping & Validation Workflow", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    plt.savefig(out_dir / "validation_workflow.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# METRIC PROFILE PLOTS (15 INCOIS DEPTHS ONLY)
# ==============================================================================

def plot_individual_and_combined_metrics(df_incois: pd.DataFrame, out_dir: Path):
    """Plots RMSE, MAE, Bias, and Pearson r vertical profiles across the 15 INCOIS depths."""
    y_depth = df_incois["depth_m"]

    # 1. RMSE vs Depth
    fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
    ax.plot(df_incois["rmse"], y_depth, "o-", color="#C0392B", lw=2.5, ms=6)
    ax.axhspan(50, 250, color="#FCF3CF", alpha=0.5, label="Thermocline Region (50m-250m)")
    for r_val, d_val in zip(df_incois["rmse"], y_depth):
        ax.text(r_val + 0.01, d_val, f"{d_val:.0f}m ({r_val:.2f}°C)", fontsize=7.5, va="center", color="#2C3E50")
    ax.set_title("RMSE vs. Depth (°C) [15 INCOIS Depths]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Root Mean Squared Error (°C)")
    ax.set_ylabel("Depth (meters)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "rmse_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. MAE vs Depth
    fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
    ax.plot(df_incois["mae"], y_depth, "s-", color="#D35400", lw=2.5, ms=6)
    ax.axhspan(50, 250, color="#FCF3CF", alpha=0.5, label="Thermocline Region (50m-250m)")
    for m_val, d_val in zip(df_incois["mae"], y_depth):
        ax.text(m_val + 0.01, d_val, f"{d_val:.0f}m ({m_val:.2f}°C)", fontsize=7.5, va="center", color="#2C3E50")
    ax.set_title("MAE vs. Depth (°C) [15 INCOIS Depths]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Absolute Error (°C)")
    ax.set_ylabel("Depth (meters)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "mae_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Bias vs Depth
    fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
    ax.plot(df_incois["bias"], y_depth, "d-", color="#8E44AD", lw=2.5, ms=6)
    ax.axvline(0, color="gray", linestyle="--", lw=1.5, label="Zero Bias Reference")
    ax.axhspan(50, 250, color="#FCF3CF", alpha=0.5, label="Thermocline Region")
    ax.set_title("Prediction Bias vs. Depth (°C) [15 INCOIS Depths]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean Bias (°C) [Pred - Ground Truth]")
    ax.set_ylabel("Depth (meters)", fontsize=11, fontweight="bold")
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_dir / "bias_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Pearson Correlation vs Depth
    fig, ax = plt.subplots(figsize=(6, 7), dpi=300)
    ax.plot(df_incois["pearson_r"], y_depth, "^-", color="#16A085", lw=2.5, ms=6)
    ax.axvline(0.90, color="crimson", linestyle="--", lw=1.5, label="Performance Threshold (r = 0.90)")
    ax.set_title("Pearson Correlation (r) vs. Depth [15 INCOIS Depths]", fontsize=12, fontweight="bold")
    ax.set_xlabel("Pearson Correlation Coefficient (r)")
    ax.set_ylabel("Depth (meters)", fontsize=11, fontweight="bold")
    ax.set_xlim(-0.05, 1.02)
    ax.invert_yaxis()
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_dir / "pearson_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Combined Metric Dashboard
    fig, axes = plt.subplots(1, 4, figsize=(18, 7), sharey=True, dpi=300)
    
    # RMSE Subplot
    axes[0].plot(df_incois["rmse"], y_depth, "o-", color="#C0392B", lw=2)
    axes[0].set_title("RMSE (°C)", fontweight="bold")
    axes[0].set_xlabel("RMSE (°C)")
    axes[0].set_ylabel("Depth (meters)", fontweight="bold")
    axes[0].invert_yaxis()
    axes[0].grid(True, linestyle="--", alpha=0.6)

    # MAE Subplot
    axes[1].plot(df_incois["mae"], y_depth, "s-", color="#D35400", lw=2)
    axes[1].set_title("MAE (°C)", fontweight="bold")
    axes[1].set_xlabel("MAE (°C)")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    # Bias Subplot
    axes[2].plot(df_incois["bias"], y_depth, "d-", color="#8E44AD", lw=2)
    axes[2].axvline(0, color="gray", linestyle="--")
    axes[2].set_title("Bias (°C)", fontweight="bold")
    axes[2].set_xlabel("Bias (°C)")
    axes[2].grid(True, linestyle="--", alpha=0.6)

    # Pearson Correlation Subplot
    axes[3].plot(df_incois["pearson_r"], y_depth, "^-", color="#16A085", lw=2)
    axes[3].axvline(0.90, color="crimson", linestyle="--", label="Threshold r=0.90")
    axes[3].set_title("Pearson Correlation (r)", fontweight="bold")
    axes[3].set_xlabel("Correlation (r)")
    axes[3].set_xlim(-0.05, 1.02)
    axes[3].grid(True, linestyle="--", alpha=0.6)
    axes[3].legend(loc="lower left")

    plt.suptitle("OceanEmbed: Comprehensive Performance Profiles Across 15 INCOIS Depths", fontsize=14, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(out_dir / "metrics_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close()


def generate_metric_heatmap_and_summary(df_incois: pd.DataFrame, out_dir: Path):
    """Generates an annotated heatmap matrix of performance metrics for all 15 INCOIS depths."""
    metric_cols = ["rmse", "mae", "bias", "pearson_r"]
    data_matrix = df_incois[metric_cols].values

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 10), dpi=300)
    im = ax.imshow(data_matrix, cmap="YlGnBu", aspect="auto")

    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(df_incois)))
    ax.set_xticklabels(["RMSE (°C)", "MAE (°C)", "Bias (°C)", "Pearson r"], fontweight="bold", fontsize=10)
    ax.set_yticklabels([f"{d:.0f} meters" for d in df_incois["depth_m"]], fontweight="bold")

    for i in range(len(df_incois)):
        for j in range(len(metric_cols)):
            val = data_matrix[i, j]
            color = "white" if (j < 3 and val > 0.8) or (j == 3 and val < 0.5) else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", color=color, fontweight="bold", fontsize=9)

    ax.set_title("INCOIS 15-Depth Performance Metric Matrix Heatmap", fontsize=12, fontweight="bold", pad=12)
    plt.colorbar(im, ax=ax, label="Metric Value Intensity")
    plt.tight_layout()
    plt.savefig(out_dir / "metric_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Metric Summary Bar Graph
    fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
    x = np.arange(len(df_incois))
    width = 0.35

    ax.bar(x - width/2, df_incois["rmse"], width, label="RMSE (°C)", color="#C0392B", alpha=0.85)
    ax.bar(x + width/2, df_incois["mae"], width, label="MAE (°C)", color="#D35400", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0f}m" for d in df_incois["depth_m"]], rotation=45, ha="right")
    ax.set_ylabel("Error (°C)", fontweight="bold")
    ax.set_title("Depth-wise RMSE and MAE Comparison across 15 INCOIS Levels", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "metric_summary.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# ERROR & RESIDUAL ANALYTICS PLOTS
# ==============================================================================

def generate_error_visualizations(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray, df_incois: pd.DataFrame, out_dir: Path):
    """Generates prediction vs observation scatter, residual distributions, and residual vs depth plots."""
    valid_mask = masks == 1.0
    p_flat = preds[valid_mask]
    t_flat = targets[valid_mask]

    # Subsample for lightweight scatter rendering if dataset is large
    if len(p_flat) > 100_000:
        sub_idx = np.random.choice(len(p_flat), size=100_000, replace=False)
        p_sub, t_sub = p_flat[sub_idx], t_flat[sub_idx]
    else:
        p_sub, t_sub = p_flat, t_flat

    residuals = p_sub - t_sub
    abs_errors = np.abs(residuals)

    # 1. Prediction vs Observation Hexbin Scatter
    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    hb = ax.hexbin(t_sub, p_sub, gridsize=70, cmap="Blues", mincnt=1, bins="log")
    min_val, max_val = min(t_sub.min(), p_sub.min()), max(t_sub.max(), p_sub.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal 1:1 Reference Line")

    overall_r = np.corrcoef(t_sub, p_sub)[0, 1]
    overall_rmse = np.sqrt(np.mean((p_sub - t_sub) ** 2))
    overall_mae = np.mean(abs_errors)

    stats_text = (
        f"Valid Samples : {len(p_flat):,}\n"
        f"Pearson r     : {overall_r:.4f}\n"
        f"Overall RMSE  : {overall_rmse:.3f} °C\n"
        f"Overall MAE   : {overall_mae:.3f} °C"
    )
    ax.text(0.05, 0.93, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_title("Target Ground Truth vs. Predicted Subsurface Temperature", fontweight="bold")
    ax.set_xlabel("Observed Ground Truth Temperature (°C)")
    ax.set_ylabel("Predicted Temperature (°C)")
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.colorbar(hb, ax=ax, label="Log10 Sample Density")
    plt.tight_layout()
    plt.savefig(out_dir / "prediction_vs_observation.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Residual Distribution (Pred - Target)
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.hist(residuals, bins=100, color="#2980B9", edgecolor="white", alpha=0.8, density=True)
    ax.axvline(0, color="red", linestyle="--", lw=2, label="Zero Bias Line")
    ax.axvline(np.mean(residuals), color="orange", linestyle="-", lw=2, label=f"Mean Bias ({np.mean(residuals):.3f}°C)")

    res_stats = f"Mean Bias : {np.mean(residuals):.3f} °C\nStd Dev   : {np.std(residuals):.3f} °C\nRMSE      : {overall_rmse:.3f} °C"
    ax.text(0.05, 0.93, res_stats, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_title("Residual Error Distribution (Predicted - Ground Truth)", fontweight="bold")
    ax.set_xlabel("Residual Error (°C)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 3. Residual Error vs Depth Boxplot
    fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
    depth_residuals = []
    for d_idx in range(len(INCOIS_STANDARD_DEPTHS_M)):
        d_mask = masks[:, d_idx, :, :] == 1.0
        d_p = preds[:, d_idx, :, :][d_mask]
        d_t = targets[:, d_idx, :, :][d_mask]
        depth_residuals.append(d_p - d_t)

    bp = ax.boxplot(depth_residuals, patch_artist=True, showfliers=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("#85C1E9")
        patch.set_edgecolor("#1B4F72")

    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.set_xticklabels([f"{d:.0f}m" for d in INCOIS_STANDARD_DEPTHS_M], rotation=45)
    ax.set_title("Residual Prediction Error Distribution Across 15 INCOIS Depth Levels", fontweight="bold")
    ax.set_xlabel("INCOIS Standard Depths")
    ax.set_ylabel("Residual Error (°C) [Pred - Target]")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "residual_vs_depth.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 4. Absolute Error Distribution
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    ax.hist(abs_errors, bins=100, color="#E67E22", edgecolor="white", alpha=0.8, density=True)
    median_err = np.median(abs_errors)
    p90_err = np.percentile(abs_errors, 90)
    p95_err = np.percentile(abs_errors, 95)

    ax.axvline(overall_mae, color="red", linestyle="--", lw=2, label=f"MAE ({overall_mae:.3f}°C)")
    ax.axvline(median_err, color="blue", linestyle="-.", lw=2, label=f"Median ({median_err:.3f}°C)")

    pct_text = f"MAE (Mean) : {overall_mae:.3f} °C\nMedian     : {median_err:.3f} °C\n90th Pct   : {p90_err:.3f} °C\n95th Pct   : {p95_err:.3f} °C"
    ax.text(0.55, 0.93, pct_text, transform=ax.transAxes, fontsize=10,
            verticalalignment="top", bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="gray"))

    ax.set_title("Absolute Prediction Error Distribution", fontweight="bold")
    ax.set_xlabel("Absolute Error (°C)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(out_dir / "absolute_error_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 5. Error Heatmap Matrix
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    err_matrix = np.array([df_incois["rmse"].values, df_incois["mae"].values, np.abs(df_incois["bias"].values)])
    im = ax.imshow(err_matrix, cmap="Reds", aspect="auto")

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["RMSE (°C)", "MAE (°C)", "|Bias| (°C)"], fontweight="bold")
    ax.set_xticks(np.arange(len(INCOIS_STANDARD_DEPTHS_M)))
    ax.set_xticklabels([f"{d:.0f}m" for d in INCOIS_STANDARD_DEPTHS_M], rotation=45)

    for i in range(3):
        for j in range(len(INCOIS_STANDARD_DEPTHS_M)):
            val = err_matrix[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color="black" if val < 0.6 else "white", fontsize=8.5, fontweight="bold")

    ax.set_title("Depth-wise Error Heatmap Summary", fontweight="bold")
    plt.colorbar(im, ax=ax, label="Error Magnitude (°C)")
    plt.tight_layout()
    plt.savefig(out_dir / "error_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# SPATIAL TEMPERATURE COMPARISON PLOTS (15 INCOIS DEPTHS)
# ==============================================================================

def generate_spatial_visualizations(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray, out_dir: Path):
    """Generates spatial comparison heatmaps for representative INCOIS depth layers."""
    sample_p = preds[0].copy()
    sample_t = targets[0].copy()
    sample_m = masks[0]

    # Select representative INCOIS depths: Surface (0m), Upper Thermocline (50m), Lower Thermocline (150m), Deep (500m)
    rep_depths_m = [0.0, 50.0, 150.0, 500.0]
    depth_indices = [int(np.argmin(np.abs(np.array(INCOIS_STANDARD_DEPTHS_M) - d))) for d in rep_depths_m]

    # Apply land masking (NaN)
    for d in range(sample_p.shape[0]):
        m_d = sample_m[d] == 0
        sample_p[d][m_d] = np.nan
        sample_t[d][m_d] = np.nan

    # 1. Target vs Prediction vs Absolute Error Comparison (4 depths x 3 columns)
    fig, axes = plt.subplots(len(depth_indices), 3, figsize=(15, 12), dpi=300)

    for r_idx, d_idx in enumerate(depth_indices):
        t_slice = sample_t[d_idx]
        p_slice = sample_p[d_idx]
        err_slice = np.abs(p_slice - t_slice)

        v_min = np.nanmin(t_slice)
        v_max = np.nanmax(t_slice)
        d_label = f"{INCOIS_STANDARD_DEPTHS_M[d_idx]:.0f}m"

        # Ground Truth Target
        im0 = axes[r_idx, 0].imshow(t_slice, cmap="viridis", vmin=v_min, vmax=v_max, origin="lower")
        axes[r_idx, 0].set_title(f"Target Temp [{d_label}]", fontweight="bold")
        plt.colorbar(im0, ax=axes[r_idx, 0], fraction=0.046, pad=0.04)

        # Model Prediction
        im1 = axes[r_idx, 1].imshow(p_slice, cmap="viridis", vmin=v_min, vmax=v_max, origin="lower")
        axes[r_idx, 1].set_title(f"Predicted Temp [{d_label}]", fontweight="bold")
        plt.colorbar(im1, ax=axes[r_idx, 1], fraction=0.046, pad=0.04)

        # Absolute Error
        im2 = axes[r_idx, 2].imshow(err_slice, cmap="Reds", origin="lower")
        axes[r_idx, 2].set_title(f"Abs Error (°C) [{d_label}]", fontweight="bold")
        plt.colorbar(im2, ax=axes[r_idx, 2], fraction=0.046, pad=0.04)

        for col in range(3):
            axes[r_idx, col].axis("off")

    plt.suptitle("OceanEmbed Spatial Slices: Target vs Prediction vs Absolute Error across INCOIS Depths", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "target_prediction_error_comparison.png", dpi=300, bbox_inches="tight")
    plt.savefig(out_dir / "spatial_temperature_slices.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Standalone Spatial Absolute Error Slices
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=300)
    axes_flat = axes.flatten()

    for idx, d_idx in enumerate(depth_indices):
        t_slice = sample_t[d_idx]
        p_slice = sample_p[d_idx]
        err_slice = np.abs(p_slice - t_slice)
        d_label = f"{INCOIS_STANDARD_DEPTHS_M[d_idx]:.0f}m"

        im = axes_flat[idx].imshow(err_slice, cmap="Reds", origin="lower")
        axes_flat[idx].set_title(f"Absolute Error Map at {d_label}", fontweight="bold")
        axes_flat[idx].axis("off")
        plt.colorbar(im, ax=axes_flat[idx], label="Error (°C)", fraction=0.046, pad=0.04)

    plt.suptitle("Spatial Prediction Absolute Error Distribution Across Selected INCOIS Depths", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "spatial_error_slices.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# VERTICAL TEMPERATURE PROFILES & THERMOCLINE ANALYSIS
# ==============================================================================

def generate_profile_visualizations(preds: np.ndarray, targets: np.ndarray, masks: np.ndarray, out_dir: Path):
    """Generates vertical temperature profiles and thermocline gradient visualizations."""
    sample_p = preds[0]
    sample_t = targets[0]
    sample_m = masks[0]

    # Find valid ocean grid points
    valid_locs = np.argwhere(sample_m[0] == 1)
    if len(valid_locs) < 3:
        valid_locs = np.argwhere(sample_m == 1)

    selected_points = valid_locs[np.linspace(0, len(valid_locs) - 1, 3, dtype=int)]

    # 1. Vertical Temperature Profile Comparison (3 Representative Ocean Locations)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, dpi=300)

    for i, (py, px) in enumerate(selected_points):
        pred_prof = sample_p[:, py, px]
        target_prof = sample_t[:, py, px]

        axes[i].plot(target_prof, INCOIS_STANDARD_DEPTHS_M, "o-", color="#1B4F72", lw=2.5, label="Observed Target")
        axes[i].plot(pred_prof, INCOIS_STANDARD_DEPTHS_M, "s--", color="#E74C3C", lw=2, label="OceanEmbed Predicted")
        axes[i].axhspan(50, 250, color="#FCF3CF", alpha=0.4, label="Thermocline Zone")

        axes[i].set_title(f"Ocean Grid Point #{i+1} (Y={py}, X={px})", fontweight="bold")
        axes[i].set_xlabel("Temperature (°C)")
        axes[i].grid(True, linestyle="--", alpha=0.6)
        axes[i].legend(loc="lower left")

    axes[0].set_ylabel("Depth (meters)", fontweight="bold")
    axes[0].invert_yaxis()

    plt.suptitle("Observed vs. Predicted Vertical Temperature Profiles (15 INCOIS Standard Depths)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "vertical_temperature_profiles.png", dpi=300, bbox_inches="tight")
    plt.close()

    # 2. Thermocline Analysis & Temperature Gradient Profile
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

    py, px = selected_points[0]
    pred_prof = sample_p[:, py, px]
    target_prof = sample_t[:, py, px]

    # Thermal gradient (dT/dz) approximation
    depths = np.array(INCOIS_STANDARD_DEPTHS_M)
    dz = np.gradient(depths)
    dt_target = np.gradient(target_prof, dz)
    dt_pred = np.gradient(pred_prof, dz)

    # Temperature Profile in Thermocline
    axes[0].plot(target_prof, depths, "o-", color="#2E86C1", lw=2.5, label="Observed Target")
    axes[0].plot(pred_prof, depths, "s--", color="#E67E22", lw=2, label="Predicted")
    axes[0].axhspan(50, 250, color="#FCF3CF", alpha=0.5, label="Thermocline (50-250m)")
    axes[0].set_title("Vertical Temperature Profile", fontweight="bold")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Depth (m)")
    axes[0].invert_yaxis()
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    # Vertical Temperature Gradient dT/dz
    axes[1].plot(dt_target, depths, "o-", color="#2E86C1", lw=2.5, label="Observed dT/dz")
    axes[1].plot(dt_pred, depths, "s--", color="#E67E22", lw=2, label="Predicted dT/dz")
    axes[1].axhspan(50, 250, color="#FCF3CF", alpha=0.5, label="Thermocline (50-250m)")
    axes[1].set_title("Vertical Thermal Gradient (dT/dz)", fontweight="bold")
    axes[1].set_xlabel("Temperature Gradient (°C / m)")
    axes[1].invert_yaxis()
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.suptitle("Thermocline Structure & Vertical Temperature Gradient Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_dir / "thermocline_profiles.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# EXECUTIVE PPT SUMMARY DASHBOARD
# ==============================================================================

def generate_ppt_summary_dashboard(df_incois: pd.DataFrame, stats_dict: dict, out_dir: Path):
    """Generates an executive KPI summary graphic formatted for PowerPoint presentations."""
    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#F8F9F9")

    # Title Banner
    plt.text(0.5, 0.93, "OceanEmbed Model Evaluation Executive Dashboard", ha="center", va="center", fontsize=20, fontweight="bold", color="#1B2631")
    plt.text(0.5, 0.88, "Subsurface Temperature Profile Retrieval Performance across 15 INCOIS Standard Depths", ha="center", va="center", fontsize=12, color="#5D6D7E")

    # Helper function for drawing KPI Cards
    def draw_card(x, y, w, h, title, value, unit, color):
        rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02", facecolor="white", edgecolor=color, linewidth=2.5)
        fig.text(x + w / 2, y + h - 0.04, title, ha="center", va="center", fontsize=11, fontweight="bold", color="#2C3E50")
        fig.text(x + w / 2, y + h / 2 - 0.01, f"{value}", ha="center", va="center", fontsize=20, fontweight="bold", color=color)
        fig.text(x + w / 2, y + 0.03, unit, ha="center", va="center", fontsize=9.5, color="#7F8C8D")

    # 4 Executive KPI Summary Cards
    draw_card(0.06, 0.68, 0.20, 0.15, "Overall Test RMSE", f"{stats_dict['overall_mean_rmse']:.3f}", "°C (15 INCOIS Depths)", "#C0392B")
    draw_card(0.29, 0.68, 0.20, 0.15, "Overall Test MAE", f"{stats_dict['overall_mean_mae']:.3f}", "°C (15 INCOIS Depths)", "#D35400")
    draw_card(0.52, 0.68, 0.20, 0.15, "Mean Bias", f"{stats_dict['overall_mean_bias']:.3f}", "°C (Pred - Ground Truth)", "#8E44AD")
    draw_card(0.75, 0.68, 0.20, 0.15, "Pearson Correlation", f"{stats_dict['overall_mean_pearson_r']:.3f}", "Avg. Correlation (r)", "#16A085")

    # Embedded Profiles: RMSE / MAE
    ax1 = fig.add_axes([0.08, 0.12, 0.38, 0.45])
    ax1.plot(df_incois["rmse"], df_incois["depth_m"], "o-", color="#C0392B", lw=2, label="RMSE (°C)")
    ax1.plot(df_incois["mae"], df_incois["depth_m"], "s-", color="#D35400", lw=2, label="MAE (°C)")
    ax1.axhspan(50, 250, color="#FCF3CF", alpha=0.5, label="Thermocline (50-250m)")
    ax1.invert_yaxis()
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.set_title("Depth-wise Vertical Error Profile (°C)", fontweight="bold")
    ax1.set_xlabel("Error Magnitude (°C)")
    ax1.set_ylabel("INCOIS Depth (m)")
    ax1.legend(loc="lower right")

    # Embedded Profiles: Pearson Correlation (r)
    ax2 = fig.add_axes([0.54, 0.12, 0.38, 0.45])
    ax2.plot(df_incois["pearson_r"], df_incois["depth_m"], "^-", color="#16A085", lw=2, label="Pearson Correlation (r)")
    ax2.axvline(0.90, color="crimson", linestyle="--", label="Target Threshold (r=0.90)")
    ax2.axhspan(50, 250, color="#FCF3CF", alpha=0.5)
    ax2.invert_yaxis()
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.set_title("Vertical Pearson Correlation Profile (r)", fontweight="bold")
    ax2.set_xlabel("Pearson r")
    ax2.set_ylabel("INCOIS Depth (m)")
    ax2.legend(loc="lower left")

    plt.savefig(out_dir / "ppt_summary_dashboard.png", dpi=300, bbox_inches="tight")
    plt.close()


# ==============================================================================
# MAIN EVALUATION PIPELINE
# ==============================================================================

def evaluate_model(
    thermo_ckpt_path: str = None,
    remaining_ckpt_path: str = None,
    output_dir: str = None
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Using device for evaluation: {device}")

    verify_native_depths()

    out_base = Path(output_dir) if output_dir else DEFAULT_OUTPUT_DIR
    dirs = setup_output_directories(out_base)

    thermo_ckpt = Path(thermo_ckpt_path) if thermo_ckpt_path else DEFAULT_THERMO_CHECKPOINT
    remaining_ckpt = Path(remaining_ckpt_path) if remaining_ckpt_path else DEFAULT_REMAINING_CHECKPOINT

    if not thermo_ckpt.exists() or not remaining_ckpt.exists():
        raise FileNotFoundError(
            f"Checkpoint files not found at {thermo_ckpt} or {remaining_ckpt}. Ensure model training is complete."
        )

    print(f"[*] Loading thermocline model checkpoint: {thermo_ckpt}")
    ckpt_thermo = torch.load(thermo_ckpt, map_location=device, weights_only=False)

    print(f"[*] Loading remaining-depths model checkpoint: {remaining_ckpt}")
    ckpt_remaining = torch.load(remaining_ckpt, map_location=device, weights_only=False)

    # Instantiate submodels
    model_thermo = build_thermocline_model().to(device)
    model_remaining = build_remaining_depths_model().to(device)

    model_thermo.load_state_dict(ckpt_thermo["model_state_dict"])
    model_remaining.load_state_dict(ckpt_remaining["model_state_dict"])

    model_thermo.eval()
    model_remaining.eval()

    stats = ckpt_thermo.get("stats", {})
    _, _, test_loader, _ = get_dataloaders(batch_size=4)

    all_preds_35, all_targets_35, all_masks_35 = [], [], []

    print("[*] Running model inference on test split...")
    with torch.no_grad():
        for x, y, mask in test_loader:
            x_dev = x.to(device)
            pred_thermo = model_thermo(x_dev)
            pred_remaining = model_remaining(x_dev)

            preds_35 = assemble_full_depth_profile(pred_thermo, pred_remaining)

            all_preds_35.append(preds_35.cpu().numpy())
            all_targets_35.append(y.cpu().numpy())
            all_masks_35.append(mask.cpu().numpy())

    preds_arr_35 = np.concatenate(all_preds_35, axis=0)
    targets_arr_35 = np.concatenate(all_targets_35, axis=0)
    masks_arr_35 = np.concatenate(all_masks_35, axis=0)

    # Un-normalize targets and predictions if normalization stats exist
    if "target_mean" in stats and "target_std" in stats:
        t_mean, t_std = stats["target_mean"], stats["target_std"]
        preds_arr_35 = preds_arr_35 * t_std + t_mean
        targets_arr_35 = targets_arr_35 * t_std + t_mean

    # Expand masks to 4D if necessary
    if masks_arr_35.ndim == 3:
        masks_arr_35 = np.repeat(np.expand_dims(masks_arr_35, axis=1), len(NATIVE_DEPTHS_35), axis=1)

    print("\n[*] Interpolating profiles to 15 INCOIS standard depths...")
    interp_pred_fn = interp1d(NATIVE_DEPTHS_35, preds_arr_35, axis=1, bounds_error=False, fill_value="extrapolate")
    interp_target_fn = interp1d(NATIVE_DEPTHS_35, targets_arr_35, axis=1, bounds_error=False, fill_value="extrapolate")
    interp_mask_fn = interp1d(NATIVE_DEPTHS_35, masks_arr_35.astype(float), axis=1, kind="nearest", bounds_error=False, fill_value="extrapolate")

    preds_15 = interp_pred_fn(INCOIS_STANDARD_DEPTHS_M)
    targets_15 = interp_target_fn(INCOIS_STANDARD_DEPTHS_M)
    masks_15 = interp_mask_fn(INCOIS_STANDARD_DEPTHS_M) >= 0.5

    print("[*] Calculating performance evaluation metrics across 15 INCOIS depths...")
    incois_metrics = []

    for d_idx, depth_m in enumerate(INCOIS_STANDARD_DEPTHS_M):
        valid = masks_15[:, d_idx, :, :] == 1.0
        p_d = preds_15[:, d_idx, :, :][valid]
        t_d = targets_15[:, d_idx, :, :][valid]

        if len(t_d) == 0:
            continue

        rmse = float(np.sqrt(np.mean((p_d - t_d) ** 2)))
        mae = float(np.mean(np.abs(p_d - t_d)))
        bias = float(np.mean(p_d - t_d))
        corr = float(
            np.corrcoef(p_d, t_d)[0, 1]
            if np.std(p_d) > 0 and np.std(t_d) > 0 else 0.0
        )

        incois_metrics.append({
            "Depth": depth_m,
            "Number of valid cells": int(valid.sum()),
            "RMSE": rmse,
            "MAE": mae,
            "Bias": bias,
            "Pearson Correlation (r)": corr,
            # Lowercase keys for internal plotter compatibility
            "depth_m": depth_m,
            "rmse": rmse,
            "mae": mae,
            "bias": bias,
            "pearson_r": corr,
        })

    df_incois = pd.DataFrame(incois_metrics)

    # Save official 15-depth CSV
    csv_out_path = dirs["csv"] / "evaluation_metrics_15incois.csv"
    csv_cols = ["Depth", "Number of valid cells", "RMSE", "MAE", "Bias", "Pearson Correlation (r)"]
    df_incois[csv_cols].to_csv(csv_out_path, index=False)
    print(f"[+] Saved 15 INCOIS depth evaluation CSV to: {csv_out_path}")

    # Calculate summary statistics
    best_rmse_idx = df_incois["rmse"].idxmin()
    worst_rmse_idx = df_incois["rmse"].idxmax()
    best_corr_idx = df_incois["pearson_r"].idxmax()
    worst_corr_idx = df_incois["pearson_r"].idxmin()

    all_valid_cells = int(masks_15.sum())
    max_abs_err = float(np.max(np.abs(preds_15[masks_15] - targets_15[masks_15])))

    summary_stats = {
        "depth_layers_evaluated": len(INCOIS_STANDARD_DEPTHS_M),
        "overall_mean_rmse": float(df_incois["rmse"].mean()),
        "overall_mean_mae": float(df_incois["mae"].mean()),
        "overall_mean_bias": float(df_incois["bias"].mean()),
        "overall_mean_pearson_r": float(df_incois["pearson_r"].mean()),
        "median_rmse": float(df_incois["rmse"].median()),
        "max_rmse": float(df_incois["rmse"].max()),
        "max_absolute_error": max_abs_err,
        "best_performing_depth_m": float(df_incois.loc[best_rmse_idx, "depth_m"]),
        "worst_performing_depth_m": float(df_incois.loc[worst_rmse_idx, "depth_m"]),
        "highest_correlation_depth_m": float(df_incois.loc[best_corr_idx, "depth_m"]),
        "lowest_correlation_depth_m": float(df_incois.loc[worst_corr_idx, "depth_m"]),
        "number_of_valid_cells": all_valid_cells,
    }

    # Save summary stats JSON & TXT
    json_summary_path = dirs["root"] / "summary_metrics.json"
    with open(json_summary_path, "w") as f:
        json.dump(summary_stats, f, indent=4)

    txt_summary_path = dirs["root"] / "summary_metrics.txt"
    with open(txt_summary_path, "w") as f:
        for k, v in summary_stats.items():
            f.write(f"{k}: {v}\n")

    print("\n[*] Generating PPT-ready visual diagrams and graphs...")
    generate_pipeline_flowchart(dirs["flowcharts"])
    generate_workflow_diagrams(dirs["flowcharts"])
    plot_individual_and_combined_metrics(df_incois, dirs["metrics"])
    generate_metric_heatmap_and_summary(df_incois, dirs["metrics"])
    generate_error_visualizations(preds_15, targets_15, masks_15, df_incois, dirs["errors"])
    generate_spatial_visualizations(preds_15, targets_15, masks_15, dirs["spatial"])
    generate_profile_visualizations(preds_15, targets_15, masks_15, dirs["profiles"])
    generate_ppt_summary_dashboard(df_incois, summary_stats, dirs["dashboards"])

    # Final Console Output Banner
    print("\n====================================================")
    print("OceanEmbed INCOIS 15-Depth Evaluation Complete")
    print("====================================================")
    print(f"Depth Layers Evaluated : {summary_stats['depth_layers_evaluated']}")
    print(f"Overall RMSE           : {summary_stats['overall_mean_rmse']:.4f} °C")
    print(f"Overall MAE            : {summary_stats['overall_mean_mae']:.4f} °C")
    print(f"Mean Bias              : {summary_stats['overall_mean_bias']:.4f} °C")
    print(f"Mean Pearson r         : {summary_stats['overall_mean_pearson_r']:.4f}")
    print(f"\nBest Depth             : {summary_stats['best_performing_depth_m']:.0f} m")
    print(f"Worst Depth            : {summary_stats['worst_performing_depth_m']:.0f} m")
    print(f"\nCSV Output             : {dirs['csv'].resolve()}")
    print(f"Visualizations         : {dirs['root'].resolve()}")
    print(f"PPT Dashboard          : {dirs['dashboards'].resolve()}")
    print("====================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OceanEmbed INCOIS 15-Depth Comprehensive Evaluation & Visualization Engine")
    parser.add_argument("--thermocline_checkpoint", type=str, default=str(DEFAULT_THERMO_CHECKPOINT), help="Path to thermocline model checkpoint")
    parser.add_argument("--remaining_checkpoint", type=str, default=str(DEFAULT_REMAINING_CHECKPOINT), help="Path to remaining-depths model checkpoint")
    parser.add_argument("--output_dir", type=str, default=str(DEFAULT_OUTPUT_DIR), help="Directory where CSVs, diagrams, and figures will be saved")
    args = parser.parse_args()

    evaluate_model(
        thermo_ckpt_path=args.thermocline_checkpoint,
        remaining_ckpt_path=args.remaining_checkpoint,
        output_dir=args.output_dir
    )
