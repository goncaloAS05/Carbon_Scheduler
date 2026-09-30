"""
analysis.py
───────────
Post-benchmark analysis producing the benchmark summary plots, quadrant
scatter plots, and optional workflow-order permutation analysis.

Usage:
    # From the repository root, analyse benchmark_results_total.csv:
    python airflow/config/benchmark_results/analysis.py

    # Choose a different CSV or output directory:
    python airflow/config/benchmark_results/analysis.py \
        --results airflow/config/benchmark_results/benchmark_results_total.csv \
        --out-dir airflow/config/benchmark_results/analysis

    # Also run the optional permutation analysis:
    python airflow/config/benchmark_results/analysis.py \
        --permute --workloads airflow/config/benchmark_workloads.json \
        --permute-max 5

By default, input and output paths are resolved beside this script, so the
first command works from any current directory.

Run from the repository root:
    python airflow/config/benchmark_results/analysis.py --help
    python airflow/config/benchmark_results/analysis.py \
        --results airflow/config/benchmark_results/benchmark_results_total.csv \
        --out-dir airflow/config/benchmark_results/analysis

Include the optional Oracle permutation analysis:
    python airflow/config/benchmark_results/analysis.py --permute \
        --workloads airflow/config/data/benchmark_workloads_alibaba_factor100.json \
        --permute-max 5
"""

import os
import sys
import json
import math
import argparse
import itertools
import datetime as dt
import types
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats as scipy_stats

# ── Import scheduler functions for permutation analysis ──────────────────────
# The script may live in a subdirectory (e.g. benchmark_results/) while
# airflow_local_settings.py lives one level up in config/. Add both.
_HERE   = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in [_HERE, _PARENT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

if "airflow" not in sys.modules:
    airflow_module = types.ModuleType("airflow")
    airflow_exceptions = types.ModuleType("airflow.exceptions")
    class AirflowRescheduleException(Exception):
        pass
    airflow_exceptions.AirflowRescheduleException = AirflowRescheduleException
    airflow_module.exceptions = airflow_exceptions
    sys.modules["airflow"] = airflow_module
    sys.modules["airflow.exceptions"] = airflow_exceptions

from airflow_local_settings import (
    plan_workflow_alg3,
    load_carbon_data,
    load_cluster_state,
    save_cluster_state,
    calculate_standardized_stats,
    DATA_DIR,
    REGIONS,
    DATA_SIZE_GB,
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

ALG_COLORS = {
    'Alg3_RLE_Spatio':       '#7F77DD',
    'Alg4_Batch_EDF':        '#D85A30',
    'Alg4_Batch_LWF':        '#D85A30',
    'Alg4_Batch':            '#D85A30',
    'BaselineA_Earliest':    '#888780',
    'BaselineB_LocalFirst':  '#BA7517',
    'Oracle':                '#1A1A2E',
    'base_temp_lwf':         "#BA4217",
    'base_spat_lwf':         "#17BA3A",
}

ALG_MARKERS = {
    'Alg3_RLE_Spatio':       'D',
    'Alg4_Batch_EDF':        '^',
    'Alg4_Batch_LWF':        'P',
    'Alg4_Batch':            '^',
    'BaselineA_Earliest':    'v',
    'BaselineB_LocalFirst':  'P',
    'Oracle':                '*',
}

ALG_DISPLAY_NAMES = {
    'Alg4_Batch_LWF': 'Spatio-Temporal_Batch_LWF',
    'BaselineA_Earliest': 'Baseline_Earliest_Region',
    'BaselineB_LocalFirst': 'Baseline_Local_First',
}


def algorithm_display_name(algorithm: str) -> str:
    """Return the readable name used in plots without changing CSV keys."""
    return ALG_DISPLAY_NAMES.get(algorithm, algorithm)


def plot_average_carbon_by_region(out_dir: str, data_dir: str = DATA_DIR):
    """Plot average carbon intensity for every regional carbon CSV.

    The 123 regions are split across three panels only to keep the labels
    readable. All panels use the same color scale, so colors are comparable.
    """
    records = []
    for path in sorted(Path(data_dir).glob("log_*.csv")):
        region = path.stem.removeprefix("log_")
        average_carbon = np.nan
        try:
            data = pd.read_csv(path, usecols=["Carbon intensity gCO₂eq/kWh (Life cycle)"])
            values = pd.to_numeric(data.iloc[:, 0], errors="coerce").dropna()
        except (OSError, ValueError, KeyError) as exc:
            print(f"[PLOTS] Skipping {path.name}: {exc}")
            continue
        if not values.empty:
            average_carbon = values.mean()
        records.append({"region": region, "average_carbon": average_carbon})

    if not records:
        print(f"[PLOTS] No regional carbon CSVs found in {data_dir}.")
        return pd.DataFrame(columns=["region", "average_carbon"])

    averages = pd.DataFrame(records).sort_values("region").reset_index(drop=True)
    panel_regions = np.array_split(averages["region"].tolist(), 3)
    valid_averages = averages["average_carbon"].dropna()
    vmin = valid_averages.min()
    vmax = valid_averages.max()

    fig, axes = plt.subplots(1, 3, figsize=(21, max(16, len(averages) / 3 * 0.44)),
                             sharex=True)
    for panel_index, region_chunk in enumerate(panel_regions):
        panel = averages[averages["region"].isin(region_chunk)].set_index("region")
        panel = panel.reindex(region_chunk)
        heatmap = sns.heatmap(
            panel[["average_carbon"]],
            ax=axes[panel_index],
            cmap="RdYlGn_r",
            vmin=vmin,
            vmax=vmax,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 7},
            linewidths=0.3,
            cbar=False,
        )
        first_region = region_chunk[0]
        last_region = region_chunk[-1]
        axes[panel_index].set_title(
            f"Regions {panel_index + 1}/3: {first_region} to {last_region}",
            fontsize=12,
            fontweight="bold",
        )
        axes[panel_index].set_xlabel("")
        axes[panel_index].set_ylabel("Region")
        axes[panel_index].set_xticklabels(["Average gCO₂eq/kWh"])
        axes[panel_index].set_yticks(np.arange(len(panel)) + 0.5)
        axes[panel_index].set_yticklabels(panel.index, rotation=0, fontsize=8)
        axes[panel_index].tick_params(axis="y", labelleft=True)

    colorbar_axis = fig.add_axes([0.92, 0.12, 0.015, 0.78])
    colorbar = fig.colorbar(axes[0].collections[0], cax=colorbar_axis)
    colorbar.set_label("Average carbon intensity (gCO₂eq/kWh)")

    fig.suptitle("Average Carbon Intensity Across Regional Data", fontsize=16,
                 fontweight="bold")
    fig.subplots_adjust(left=0.08, right=0.89, top=0.94, bottom=0.04, wspace=0.25)
    output_path = os.path.join(out_dir, "plot_average_carbon_by_region.png")
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    missing_count = averages["average_carbon"].isna().sum()
    suffix = f"; {missing_count} missing/empty" if missing_count else ""
    print(f"[PLOTS] Average regional carbon heatmap saved ({len(averages)} regions{suffix}): {output_path}")
    return averages

plt.rcParams.update({
    'font.family':    'sans-serif',
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.grid':          True,
    'grid.alpha':         0.25,
    'grid.linestyle':     '--',
})


# ─────────────────────────────────────────────────────────────────────────────
# 1. Quadrant plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_quadrants(df: pd.DataFrame, out_dir: str):
    """
    Produce quadrant scatter plots comparing algorithms across key
    trade-off dimensions. Each point is the mean for one algorithm across
    all runs; error bars show ±1 std. The quadrant dividers are drawn at
    the grand mean of each axis so it's always clear which quadrant is
    'desirable' vs 'undesirable'.
    """
    df_ok = df[df["plan_produced"] == True].copy()
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    if "time_to_schedule_s" not in df_ok.columns and "decision_time_s" in df_ok.columns:
        df_ok["time_to_schedule_s"] = df_ok["decision_time_s"]
    if "decision_time_s" not in df_ok.columns and "time_to_schedule_s" in df_ok.columns:
        df_ok["decision_time_s"] = df_ok["time_to_schedule_s"]
    if "time_to_schedule_s" not in df_ok.columns and "decision_time_s" not in df_ok.columns:
        df_ok["time_to_schedule_s"] = np.nan
        df_ok["decision_time_s"] = np.nan

    algs = sorted(df_ok["algorithm"].unique())

    def alg_stats(metric):
        return df_ok.groupby("algorithm")[metric].agg(['mean', 'std']).reindex(algs)

    def draw_quadrant(ax, x_means, y_means, x_label, y_label,
                      x_low_is_good=True, y_low_is_good=True):
        """
        Draw the four quadrant regions with background shading and label them.
        Green = desirable, red = undesirable, yellow = mixed.
        """

        # Median dividers are more representative than means for this data:
        # makespan and decision time contain substantial high-value outliers.
        x_mid = np.median(x_means)
        y_mid = np.median(y_means)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()

        # Determine which quadrant is 'best' based on axis directions
        # x_low_is_good = lower x is better (e.g. carbon, makespan)
        # y_low_is_good = lower y is better
        q_colors = {}
        q_colors['ll'] = '#C8E6C9' if (x_low_is_good and y_low_is_good) else \
                         '#FFCDD2' if (not x_low_is_good and not y_low_is_good) else '#FFF9C4'
        q_colors['lr'] = '#C8E6C9' if (not x_low_is_good and y_low_is_good) else \
                         '#FFCDD2' if (x_low_is_good and not y_low_is_good) else '#FFF9C4'
        q_colors['ul'] = '#C8E6C9' if (x_low_is_good and not y_low_is_good) else \
                         '#FFCDD2' if (not x_low_is_good and y_low_is_good) else '#FFF9C4'
        q_colors['ur'] = '#C8E6C9' if (not x_low_is_good and not y_low_is_good) else \
                         '#FFCDD2' if (x_low_is_good and y_low_is_good) else '#FFF9C4'

        for (x0, x1, y0, y1, qk) in [
            (xmin, x_mid, ymin, y_mid, 'll'),
            (x_mid, xmax, ymin, y_mid, 'lr'),
            (xmin, x_mid, y_mid, ymax, 'ul'),
            (x_mid, xmax, y_mid, ymax, 'ur'),
        ]:
            ax.fill_between([x0, x1], [y0, y0], [y1, y1],
                            color=q_colors[qk], alpha=0.18, zorder=0)

        ax.axvline(x_mid, color='#999', lw=0.8, ls='--', zorder=1)
        ax.axhline(y_mid, color='#999', lw=0.8, ls='--', zorder=1)

    def scatter_plot(x_metric, y_metric, x_label, y_label, fname,
                     x_low_is_good=True, y_low_is_good=True, title=None):
        x_s = alg_stats(x_metric)
        y_s = alg_stats(y_metric)

        fig, ax = plt.subplots(figsize=(9, 7))

        x_vals = []
        y_vals = []

        for alg in algs:
            if alg not in x_s.index or alg not in y_s.index:
                continue
            xv = x_s.loc[alg, 'mean']
            yv = y_s.loc[alg, 'mean']
            xe = x_s.loc[alg, 'std']
            ye = y_s.loc[alg, 'std']

            if pd.isna(xv) or pd.isna(yv):
                continue

            x_vals.append(xv)
            y_vals.append(yv)

            color  = ALG_COLORS.get(alg, '#555')
            marker = ALG_MARKERS.get(alg, 'o')

            ax.errorbar(xv, yv, xerr=xe, yerr=ye,
                        fmt=marker, color=color, markersize=10,
                        elinewidth=1.2, capsize=4, alpha=0.9, zorder=3)
            ax.annotate(algorithm_display_name(alg), (xv, yv),
                        textcoords='offset points', xytext=(8, 4),
                        fontsize=8, color=color, fontweight='bold', zorder=4)

        if not x_vals or not y_vals:
            ax.text(0.5, 0.5, 'No comparable data available',
                    ha='center', va='center', fontsize=10, color='#666')
            ax.set_axis_off()
        else:
            # Keep zero visible while allowing the long-tailed metrics to fit.
            max_x = max(x_vals) * 1.5 if x_vals else 1
            max_y = max(y_vals) * 1.5 if y_vals else 1
            uses_oracle = df_ok["algorithm"].astype(str).str.contains(
                "oracle", case=False, na=False
            ).any()
            time_metrics = {"decision_time_s", "time_to_schedule_s"}
            x_is_log_time = uses_oracle and x_metric in time_metrics and min(x_vals) > 0
            y_is_log_time = uses_oracle and y_metric in time_metrics and min(y_vals) > 0

            ax.set_xlim(min(x_vals) * 0.8 if x_is_log_time else 0, max_x)
            ax.set_ylim(min(y_vals) * 0.8 if y_is_log_time else 0, max_y)

            # Oracle decision times can be orders of magnitude apart from
            # other algorithms, so show their time axis logarithmically.
            if x_is_log_time:
                ax.set_xscale('log')
            else:
                ax.set_xscale('symlog', linthresh=1.0)
            if y_is_log_time:
                ax.set_yscale('log')
            else:
                ax.set_yscale('symlog', linthresh=1.0)

            draw_quadrant(ax, np.array(x_vals), np.array(y_vals),
                          x_label, y_label, x_low_is_good, y_low_is_good)

            ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
            ax.set_title(title or f"{y_label} vs {x_label}", fontsize=13, pad=14)

            x_direction = 'lower is better' if x_low_is_good else 'higher is better'
            y_direction = 'lower is better' if y_low_is_good else 'higher is better'
            ax.set_xlabel(f'{x_label} ({x_direction})', fontsize=11, fontweight='bold')
            ax.set_ylabel(f'{y_label} ({y_direction})', fontsize=11, fontweight='bold')
            ax.text(0.01, 0.01, 'Best direction shown in axis labels',
                    transform=ax.transAxes, fontsize=8, color='#555', style='italic')

        fig.tight_layout()
        path = os.path.join(out_dir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"[QUADRANT] Saved {fname}")

    # ── Q1: Carbon vs Makespan ────────────────────────────────────────────────
    scatter_plot(
        'makespan_hours', 'total_carbon_with_transfer',
        'Avg Makespan (hours)', 'Avg Total Carbon incl. Transfer (gCO₂eq)',
        'quadrant_carbon_vs_makespan.png',
        x_low_is_good=True, y_low_is_good=True,
        title='Carbon vs Makespan — Algorithm Trade-off Space'
    )

    # ── Q2: Carbon vs Success Rate ────────────────────────────────────────────
    scatter_plot(
        'total_carbon_with_transfer', 'success_rate',
        'Avg Total Carbon incl. Transfer (gCO₂eq)', 'Avg Success Rate (%)',
        'quadrant_carbon_vs_success.png',
        x_low_is_good=True, y_low_is_good=False,
        title='Carbon vs Success Rate — Efficiency vs Reliability'
    )

    # ── Q3: Transfer Carbon vs Execution Carbon ───────────────────────────────
    scatter_plot(
        'total_carbon', 'total_transfer_carbon',
        'Avg Execution Carbon (gCO₂eq)', 'Avg Transfer Carbon (gCO₂eq)',
        'quadrant_exec_vs_transfer_carbon.png',
        x_low_is_good=True, y_low_is_good=True,
        title='Execution Carbon vs Transfer Carbon — Cross-Region Cost Breakdown'
    )

    # ── Q4: Carbon vs decision time ───────────────────────────────────────
    scatter_plot(
        'decision_time_s', 'total_carbon_with_transfer',
        'Avg Decision Time (s)', 'Avg Total Carbon incl. Transfer (gCO₂eq)',
        'quadrant_carbon_vs_decision_time.png',
        x_low_is_good=True, y_low_is_good=True,
        title='Carbon vs Decision Time — Emissions vs Planning Speed'
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Runner summary plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_runner_results(df: pd.DataFrame, out_dir: str):
    """Recreate the summary plots produced by ``benchmark_runner``."""
    df_ok = df[df["plan_produced"]].copy()
    if df_ok.empty:
        print("[PLOTS] No valid runs produced a plan. Skipping runner plots.")
        return

    alg_order = sorted(df_ok["algorithm"].dropna().unique())
    colors = plt.cm.Set2(np.linspace(0, 1, len(alg_order)))
    color_map = dict(zip(alg_order, colors))

    data_by_alg = [
        df_ok.loc[
            (df_ok["algorithm"] == alg) & (df_ok["total_carbon"] > 0),
            "total_carbon",
        ].dropna().values
        for alg in alg_order
    ]
    nonempty = [(alg, values) for alg, values in zip(alg_order, data_by_alg) if len(values)]
    if nonempty:
        alg_order, data_by_alg = zip(*nonempty)
        alg_order = list(alg_order)
        fig, ax = plt.subplots(figsize=(10, 5))
        boxplot = ax.boxplot(data_by_alg, patch_artist=True, notch=False)
        for patch, alg in zip(boxplot["boxes"], alg_order):
            patch.set_facecolor(color_map[alg])
        ax.set_xticks(
            range(1, len(alg_order) + 1),
            [algorithm_display_name(alg) for alg in alg_order],
            rotation=15,
            ha="right",
        )
        ax.set_yscale("log")
        ax.set_ylabel("Total Carbon (gCO₂eq, log scale)")
        ax.set_title("Carbon Footprint Distribution by Algorithm (Log Scale)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda value, _: f"{value:,.0f}"))
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plot_carbon_boxplot.png"), dpi=150)
        plt.close(fig)

    if "success_rate" in df_ok:
        fig, ax = plt.subplots(figsize=(10, 4))
        success_rate = df_ok.groupby("algorithm")["success_rate"].mean().reindex(alg_order)
        bars = ax.bar([algorithm_display_name(alg) for alg in alg_order], success_rate.values,
                      color=[color_map[alg] for alg in alg_order], edgecolor="white")
        ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
        ax.set_ylim(0, 110)
        ax.set_ylabel("Avg Success Rate (%)")
        ax.set_title("Average Task Success Rate by Algorithm")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plot_success_rate.png"), dpi=150)
        plt.close(fig)

    if {"deadline_hours", "makespan_hours"}.issubset(df_ok.columns):
        fig, ax = plt.subplots(figsize=(12, 6))
        
        df_makespan = df_ok.copy()
        
        # 1. Agrupar a variável contínua em intervalos (bins) definidos e limpos
        bins = [0, 24, 48, 100, 250, 500, 1000, 2500, 5000]
        labels = ["<24h", "24-48h", "48-100h", "100-250h", "250-500h", "500-1000h", "1000-2500h", ">2500h"]
        df_makespan["deadline_bin"] = pd.cut(df_makespan["deadline_hours"], bins=bins, labels=labels)
        
        # Mapeamento seguro dos nomes dos algoritmos
        if callable(algorithm_display_name):
            df_makespan["algorithm_display"] = df_makespan["algorithm"].apply(algorithm_display_name)
        elif isinstance(algorithm_display_name, dict):
            df_makespan["algorithm_display"] = df_makespan["algorithm"].map(algorithm_display_name).fillna(df_makespan["algorithm"])
        else:
            df_makespan["algorithm_display"] = df_makespan["algorithm"]
        
        # 2. Desenhar o boxplot com os intervalos agrupados no eixo X
        sns.boxplot(
            data=df_makespan,
            x="deadline_bin",
            y="makespan_hours",
            hue="algorithm_display",
            palette="Set2",
            ax=ax,
            showfliers=False
        )

        # 3. Escala logarítmica no eixo Y
        ax.set_yscale("log")

        # Formatação dos rótulos e títulos
        ax.set_xlabel("Deadline Window (hours)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Makespan (hours, log scale)", fontsize=11, fontweight="bold")
        ax.set_title("Makespan Distribution vs Deadline Pressure by Algorithm", fontsize=13, fontweight="bold")
        
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:g}"))
        ax.legend(title="Algorithm", loc="upper left", fontsize=9)
        ax.tick_params(axis="x", labelrotation=0)  # Rótulos horizontais limpos
        
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "plot_makespan_vs_deadline_boxplot.png"), dpi=300)
        plt.close(fig)
        
        print("[PLOTS] Boxplot de Makespan vs Deadline atualizado com intervalos limpos no eixo X.")

    if {"algorithm", "load_scenario", "total_carbon"}.issubset(df_ok.columns):
        load_order = [load for load in ["empty", "light", "medium", "heavy"]
                      if load in df_ok["load_scenario"].unique()]
        if len(load_order) > 1:
            pivot = (df_ok.groupby(["algorithm", "load_scenario"])["total_carbon"]
                     .mean().unstack(fill_value=np.nan).reindex(index=alg_order,
                                                                columns=load_order))
            fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.5), 5))
            sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r",
                        linewidths=0.5, ax=ax,
                        cbar_kws={"label": "Avg gCO₂eq"})
            ax.set_yticklabels([algorithm_display_name(label)
                                for label in pivot.index], rotation=0)
            ax.set_xlabel("Background load scenario")
            ax.set_ylabel("Algorithm")
            ax.set_title("Average Carbon by Algorithm × Background Load")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "plot_carbon_heatmap_load.png"), dpi=150)
            plt.close(fig)
        else:
            print("[PLOTS] Skipping heatmap: only one background-load scenario is present.")

    print("[PLOTS] Runner summary plots saved.")


def load_workflow_task_counts(workloads_path: str) -> dict[str, dict[str, float]]:
    """Return task counts and independent-task percentage by workflow ID."""
    with open(workloads_path, encoding="utf-8") as workloads_file:
        workflows = json.load(workloads_file)

    return {
        workflow["run_id"]: {
            "total_tasks": len(tasks),
            "dependent_tasks": sum(bool(task.get("depends_on")) for task in tasks),
            "independent_tasks": sum(not task.get("depends_on") for task in tasks),
            "critical_path_hours": workflow.get("critical_path_hours", np.nan),
            "independent_task_percentage": (
                sum(not task.get("depends_on") for task in tasks) / len(tasks) * 100
                if tasks else np.nan
            ),
        }
        for workflow in workflows
        if workflow.get("run_id")
        for tasks in [workflow.get("tasks", [])]
    }


def plot_dependent_task_count_and_decision_time(
        df: pd.DataFrame, out_dir: str, workloads_path: str,
        task_kind: str = "dependent"):
    """Plot a JSON-derived task metric against scheduling decision time."""
    metric_config = {
        "dependent": ("dependent_tasks", "dependent tasks", "Dependent Tasks", "dependent_task_count"),
        "independent": ("independent_tasks", "independent tasks", "Independent Tasks", "independent_task_count"),
        "independent_percentage": (
            "independent_task_percentage", "percentage of independent tasks (%)",
            "Independent Task Percentage", "independent_task_percentage",
        ),
        "critical_duration": (
            "critical_path_hours", "critical workflow duration (hours)",
            "Critical Workflow Duration", "critical_duration",
        ),
    }
    task_count_key, task_label, title_label, output_prefix = metric_config[task_kind]
    required = {"run_id", "algorithm"}
    if not required.issubset(df.columns):
        print(f"[PLOTS] Skipping {task_label} plots: required columns are missing.")
        return

    decision_metric = next(
        (metric for metric in ["decision_time_s", "time_to_schedule_s"] if metric in df.columns),
        None,
    )
    if decision_metric is None:
        print(f"[PLOTS] Skipping {task_label} plots: no decision-time column found.")
        return

    task_counts = load_workflow_task_counts(workloads_path)
    matched = df["run_id"].isin(task_counts)
    unmatched_ids = sorted(df.loc[~matched, "run_id"].dropna().unique())
    if unmatched_ids:
        print(f"[PLOTS] No workload JSON match for {len(unmatched_ids)} run_id(s): "
              f"{unmatched_ids[:5]}")

    columns = ["run_id", "algorithm", decision_metric]
    plot_data = df.loc[df["plan_produced"], columns].copy()
    plot_data["task_count"] = plot_data["run_id"].map(task_counts).map(
        lambda counts: counts[task_count_key] if isinstance(counts, dict) else np.nan
    )
    plot_data["task_count"] = pd.to_numeric(plot_data["task_count"], errors="coerce")
    plot_data[decision_metric] = pd.to_numeric(plot_data[decision_metric], errors="coerce")
    plot_data = plot_data.dropna(subset=["algorithm", "task_count", decision_metric])
    plot_data = plot_data[(plot_data["task_count"] >= 0) & (plot_data[decision_metric] >= 0)]

    if plot_data.empty:
        print(f"[PLOTS] Skipping {task_label} plots: no usable rows found.")
        return

    alg_order = sorted(plot_data["algorithm"].unique())
    use_log_time = (
        plot_data["algorithm"].astype(str).str.contains("oracle", case=False, na=False).any()
        and (plot_data[decision_metric] > 0).all()
    )
    palette = dict(zip(alg_order, sns.color_palette("Set2", n_colors=len(alg_order))))

    # Keep workload-size aggregation for the separate scaling plot.
    line_data = (plot_data.groupby(["algorithm", "task_count"], as_index=False)[decision_metric]
                 .mean().sort_values(["algorithm", "task_count"]))
    fig, ax = plt.subplots(figsize=(10, 7))
    x_values = line_data["task_count"].to_numpy()
    y_values = line_data[decision_metric].to_numpy()
    x_mid = np.median(x_values)
    y_mid = np.median(y_values)
    x_max = max(x_values) * 1.2 if len(x_values) else 1
    y_max = max(y_values) * 1.2 if len(y_values) else 1
    ax.fill_between([0, x_mid], 0, y_mid, color="#C8E6C9", alpha=0.35)
    ax.fill_between([x_mid, x_max], y_mid, y_max, color="#FFCDD2", alpha=0.35)
    ax.fill_between([0, x_mid], y_mid, y_max, color="#FFF9C4", alpha=0.35)
    ax.fill_between([x_mid, x_max], 0, y_mid, color="#FFF9C4", alpha=0.35)
    ax.axvline(x_mid, color="#888", linestyle="--", linewidth=1)
    ax.axhline(y_mid, color="#888", linestyle="--", linewidth=1)

    for algorithm in alg_order:
        algorithm_data = line_data[line_data["algorithm"] == algorithm]
        ax.plot(
            algorithm_data["task_count"], algorithm_data[decision_metric],
            color=palette[algorithm], linewidth=1.5, alpha=0.75, zorder=2,
        )
        ax.scatter(
            algorithm_data["task_count"], algorithm_data[decision_metric],
            color=palette[algorithm], s=45, alpha=0.85,
            label=algorithm_display_name(algorithm), zorder=3,
        )

    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(f"Number of {task_label} (lower is simpler)")
    ax.set_ylabel(
        "Mean decision time (s, log scale, lower is faster)"
        if use_log_time else "Mean decision time (s, lower is faster)"
    )
    if use_log_time:
        ax.set_yscale("log")
    ax.set_title(f"{title_label} vs Scheduling Decision Time Quadrant")
    ax.legend(title="Algorithm", loc="best")
    ax.text(0.02, 0.96, "Few + slow", transform=ax.transAxes,
            fontsize=9, color="#8A6D1D", fontweight="bold", va="top")
    ax.text(0.98, 0.96, "Large + slow", transform=ax.transAxes,
            fontsize=9, color="#C62828", fontweight="bold", va="top", ha="right")
    ax.text(0.02, 0.05, "Few + fast", transform=ax.transAxes,
            fontsize=9, color="#2E7D32", fontweight="bold")
    ax.text(0.98, 0.05, "Large + fast", transform=ax.transAxes,
            fontsize=9, color="#8A6D1D", fontweight="bold", ha="right")
    ax.text(0.02, 0.02, f"Lower-left: fewer {task_label} and faster decisions",
            transform=ax.transAxes, fontsize=8, color="#555", style="italic")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"plot_{output_prefix}_decision_time_quadrant.png"), dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 6))
    for algorithm in alg_order:
        algorithm_data = line_data[line_data["algorithm"] == algorithm]
        ax.plot(
            algorithm_data["task_count"], algorithm_data[decision_metric],
            marker="o", linewidth=2, markersize=5, color=palette[algorithm],
            label=algorithm_display_name(algorithm),
        )
    ax.set_xlabel(f"Number of {task_label} in workflow")
    ax.set_ylabel("Mean decision time (s, log scale)" if use_log_time else "Mean decision time (s)")
    ax.set_title("Decision-Time Scaling by Algorithm")
    if use_log_time:
        ax.set_yscale("log")
    ax.legend(title="Algorithm", loc="best")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"plot_decision_time_by_{output_prefix}.png"), dpi=200)
    plt.close(fig)

    print(f"[PLOTS] {title_label} quadrant and {decision_metric} scaling plots saved.")


def plot_runner_grouped_workload_bars(df: pd.DataFrame, out_dir: str):
    """Plot mean carbon by origin region, split across 3 separate readable 1x3 figures."""
    required = {"origin_region", "algorithm", "total_carbon"}
    df_ok = df[df["plan_produced"]].copy()
    if df_ok.empty or not required.issubset(df_ok.columns):
        return

    grouped_data = (df_ok.groupby(["origin_region", "algorithm"])["total_carbon"]
                    .mean().reset_index())
    regions = sorted(grouped_data["origin_region"].unique())
    
    # Split all regions into 9 total chunks
    region_chunks = np.array_split(regions, 9)
    y_max = grouped_data["total_carbon"].max() * 1.12
    unique_algs = sorted(grouped_data["algorithm"].unique())

    # Divide the 9 chunks into 3 parts (3 chunks per figure)
    part_chunks = [region_chunks[0:3], region_chunks[3:6], region_chunks[6:9]]

    for part_idx, chunks_in_part in enumerate(part_chunks, start=1):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        axes = axes.ravel()
        
        for i, region_chunk in enumerate(chunks_in_part):
            chart_regions = list(region_chunk)
            chart_data = grouped_data[grouped_data["origin_region"].isin(chart_regions)]
            ax = axes[i]
            
            sns.barplot(
                data=chart_data, x="origin_region", y="total_carbon",
                hue="algorithm", hue_order=unique_algs, palette="Set2",
                edgecolor="black", ax=ax
            )
            
            ax.set_ylim(0, y_max)
            region_num = (part_idx - 1) * 3 + i + 1
            ax.set_title(f"Region Set {region_num}", fontsize=11, fontweight="bold")
            ax.set_xlabel("Origin Region", fontsize=10, fontweight="bold")
            
            if i == 0:
                ax.set_ylabel("Mean Carbon (gCO₂eq)", fontsize=11, fontweight="bold")
            else:
                ax.set_ylabel("")
            
            ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda val, _: f"{val:,.0f}"))
            ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8.5)
            
            # Manage legend: show once on the last subplot of each figure
            if i < 2:
                if ax.get_legend():
                    ax.get_legend().remove()
            else:
                handles, labels = ax.get_legend_handles_labels()
                if callable(algorithm_display_name):
                    labels = [algorithm_display_name(lbl) for lbl in labels]
                ax.legend(handles, labels, title="Algorithms", loc="upper right", fontsize=8)

        fig.suptitle(f"Mean Carbon Footprint by Origin Region (Part {part_idx} of 3)", fontsize=13, fontweight="bold")
        fig.tight_layout()
        
        file_path = os.path.join(out_dir, f"plot_workload_grouped_bars_part{part_idx}.jpg")
        fig.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    print("[PLOTS] Saved 3 separate regional carbon figures (Part 1, 2, and 3).")

def print_summary_table(df: pd.DataFrame):
    """Print the same per-algorithm metric summary as the benchmark runner."""
    df_ok = df[df["plan_produced"]]
    metrics = [metric for metric in [
        "success_rate", "total_carbon", "total_transfer_carbon",
        "total_carbon_with_transfer", "total_energy", "makespan_hours",
        "waiting_time_hours", "deadline_fulfillment_pct",
        "avg_task_execution_time_hours", "time_to_schedule_s", "decision_time_s",
    ] if metric in df_ok.columns]
    if df_ok.empty or not metrics:
        return
    print("\nBENCHMARK SUMMARY (mean across all workflows × scenarios)")
    summary = (df_ok.groupby("algorithm")[metrics].mean().round(2)
               .sort_values("total_carbon"))
    summary.index = [algorithm_display_name(algorithm) for algorithm in summary.index]
    print(summary.to_string())


# 3. Permutation analysis
# ─────────────────────────────────────────────────────────────────────────────

def run_permutation_analysis(workflows: list, submission: dt.datetime,
                              deadline: dt.datetime, out_dir: str,
                              max_workflows: int = 5):
    """
    Takes a list of workflows, generates ALL possible orderings (permutations),
    runs alg3 on each ordering, and measures how much the scheduling order
    affects carbon, makespan, and success rate.

    This answers: "How sensitive is carbon efficiency to the order in which
    we schedule workflows?" — and sets an upper/lower bound on what optimal
    ordering (the oracle's job) can achieve vs worst ordering.

    Parameters
    ----------
    max_workflows : cap on how many workflows to permute (n! grows fast —
                   5! = 120 runs, 6! = 720, 7! = 5040)
    """
    wfs = workflows[:max_workflows]
    n   = len(wfs)
    n_perms = math.factorial(n)

    print(f"[PERMUTE] {n} workflows → {n_perms} permutations")
    print(f"[PERMUTE] Submission: {submission} | Deadline: {deadline}")

    all_traces = {reg: load_carbon_data(reg, submission, deadline) for reg in REGIONS}
    sub_naive  = submission.replace(tzinfo=None)

    results = []

    for perm_idx, perm in enumerate(itertools.permutations(wfs)):
        save_cluster_state({})   # fresh cluster for each permutation
        perm_carbon   = 0.0
        perm_transfer = 0.0
        perm_success  = 0
        perm_tasks    = 0
        perm_ends     = []

        for wf in perm:
            edges = [
                (dep, t["id"])
                for t in wf["tasks"]
                for dep in t.get("depends_on", [])
            ]
            run_id = f"perm_{perm_idx:04d}_{wf['run_id']}"

            plan = plan_workflow_alg3(
                run_id, submission, deadline,
                wf["tasks"], edges, wf.get("sla_level", 95),
                in_memory_state=load_cluster_state()
            )

            if plan:
                stats = calculate_standardized_stats(
                    plan, all_traces, sub_naive,
                    total_requested=wf["n_tasks"], deadline=deadline
                )
                perm_carbon   += stats.get('total_carbon', 0)
                perm_transfer += stats.get('total_transfer_carbon', 0)
                perm_success  += len(plan)
            perm_tasks += wf["n_tasks"]

        total_carbon_with_transfer = perm_carbon + perm_transfer
        success_rate = round(perm_success / perm_tasks * 100, 1) if perm_tasks > 0 else 0

        results.append({
            "perm_idx":                   perm_idx,
            "order":                      [wf["run_id"] for wf in perm],
            "total_carbon":               round(perm_carbon, 2),
            "total_transfer_carbon":      round(perm_transfer, 2),
            "total_carbon_with_transfer": round(total_carbon_with_transfer, 2),
            "success_rate":               success_rate,
        })

        if (perm_idx + 1) % max(1, n_perms // 10) == 0:
            pct = (perm_idx + 1) / n_perms * 100
            print(f"[PERMUTE] {pct:.0f}% ({perm_idx + 1}/{n_perms})")

    save_cluster_state({})

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "permutation_analysis.csv")
    df.to_csv(csv_path, index=False)
    print(f"[PERMUTE] Results saved → {csv_path}")

    _plot_permutation_results(df, out_dir)
    _print_permutation_stats(df)

    return df


def _print_permutation_stats(df: pd.DataFrame):
    """Print statistical summary of permutation results."""
    metric = 'total_carbon_with_transfer'
    vals   = df[metric].dropna()

    best_row  = df.loc[df[metric].idxmin()]
    worst_row = df.loc[df[metric].idxmax()]

    print("\n" + "═" * 70)
    print("PERMUTATION ANALYSIS — CARBON (incl. transfer)")
    print("═" * 70)
    print(f"  Permutations tested : {len(df)}")
    print(f"  Min  (best order)   : {vals.min():.2f} gCO₂eq")
    print(f"  Max  (worst order)  : {vals.max():.2f} gCO₂eq")
    print(f"  Mean                : {vals.mean():.2f} gCO₂eq")
    print(f"  Std deviation       : {vals.std():.2f} gCO₂eq")
    print(f"  Variance            : {vals.var():.2f}")
    print(f"  P25                 : {np.percentile(vals, 25):.2f} gCO₂eq")
    print(f"  P50 (median)        : {np.percentile(vals, 50):.2f} gCO₂eq")
    print(f"  P75                 : {np.percentile(vals, 75):.2f} gCO₂eq")
    print(f"  P95                 : {np.percentile(vals, 95):.2f} gCO₂eq")
    savings_vs_worst = (vals.max() - vals.min()) / vals.max() * 100
    print(f"  Best vs worst order : {savings_vs_worst:.1f}% carbon savings potential")
    print(f"\n  Best ordering  : {best_row['order']}")
    print(f"  Worst ordering : {worst_row['order']}")
    print("═" * 70)


def _plot_permutation_results(df: pd.DataFrame, out_dir: str):
    """Three plots for permutation analysis."""
    metric     = 'total_carbon_with_transfer'
    vals       = df[metric].dropna().sort_values().values
    n_perms    = len(vals)

    # ── Plot 1: Distribution histogram with percentile markers ───────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(vals, bins=min(40, n_perms // 2 + 1),
            color='#7F77DD', edgecolor='white', alpha=0.85)

    for p, label, color in [
        (25, 'P25', '#1D9E75'), (50, 'P50', '#378ADD'),
        (75, 'P75', '#BA7517'), (95, 'P95', '#D85A30')
    ]:
        pv = np.percentile(vals, p)
        ax.axvline(pv, color=color, lw=1.5, ls='--',
                   label=f'{label}: {pv:,.0f}')

    ax.axvline(vals.min(), color='#1A1A2E', lw=2, ls='-',
               label=f'Best: {vals.min():,.0f}')
    ax.axvline(vals.max(), color='#D4537E', lw=2, ls='-',
               label=f'Worst: {vals.max():,.0f}')

    ax.set_xlabel('Total Carbon incl. Transfer (gCO₂eq)', fontsize=11)
    ax.set_ylabel('Number of orderings', fontsize=11)
    ax.set_title(f'Carbon Distribution Across All {n_perms} Workflow Orderings',
                 fontsize=13, pad=12)
    ax.legend(fontsize=9, ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'permutation_carbon_distribution.png'), dpi=150)
    plt.close(fig)

    # ── Plot 2: Sorted carbon values — shows spread and ordering sensitivity ─
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_perms), vals, color='#7F77DD', lw=1.5, alpha=0.8)
    ax.fill_between(range(n_perms), vals, vals.min(),
                    color='#7F77DD', alpha=0.12)
    ax.axhline(vals.mean(), color='#BA7517', lw=1.2, ls='--',
               label=f'Mean: {vals.mean():,.0f}')
    ax.set_xlabel('Permutation rank (sorted by carbon)', fontsize=11)
    ax.set_ylabel('Total Carbon incl. Transfer (gCO₂eq)', fontsize=11)
    ax.set_title('Carbon by Permutation Rank — Ordering Sensitivity', fontsize=13, pad=12)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'permutation_carbon_sorted.png'), dpi=150)
    plt.close(fig)

    # ── Plot 3: Carbon vs success rate scatter across permutations ────────────
    fig, ax = plt.subplots(figsize=(8, 6))
    sc = ax.scatter(df[metric], df['success_rate'],
                    c=range(len(df)), cmap='viridis',
                    s=30, alpha=0.6, edgecolors='none')
    plt.colorbar(sc, ax=ax, label='Permutation index')
    ax.set_xlabel('Total Carbon incl. Transfer (gCO₂eq)', fontsize=11)
    ax.set_ylabel('Success Rate (%)', fontsize=11)
    ax.set_title('Carbon vs Success Rate — All Permutations', fontsize=13, pad=12)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, 'permutation_carbon_vs_success.png'), dpi=150)
    plt.close(fig)

    print("[PERMUTE] Plots saved.")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carbon_Scheduler post-benchmark analysis")
    parser.add_argument("--results",      default=os.path.join(_HERE, "benchmark_results_total.csv"),
                        help="Path to the benchmark CSV (default: benchmark_results_total.csv beside this script)")
    parser.add_argument("--out-dir",      default=os.path.join(_HERE, "analysis"),
                        help="Directory for generated plots (default: analysis beside this script)")
    parser.add_argument(
        "--workloads",
        default=os.path.join(_PARENT, "data", "benchmark_workloads_alibaba_factor100.json"),
        help="Path to the workload JSON used for dependency counts and permutation analysis",
    )
    parser.add_argument("--permute",      action="store_true",
                        help="Run permutation analysis")
    parser.add_argument("--permute-max",  type=int, default=5,
                        help="Max workflows to permute (default 5 = 120 permutations)")
    parser.add_argument("--permute-deadline", type=int, default=24,
                        help="Deadline window in hours for permutation runs (default 24)")
    args = parser.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Quadrant plots from existing CSV ─────────────────────────────────────
    #the csv is in config/benchmark_results/benchmark_results.csv

    print(f"[ANALYSIS] Loading results from {args.results} ...")
    df = pd.read_csv(args.results)
    if "plan_produced" not in df.columns:
        raise ValueError("Results CSV is missing the required 'plan_produced' column.")
    df["plan_produced"] = df["plan_produced"].astype(str).str.lower().isin(["true", "1", "yes"])

    print(f"[ANALYSIS] {len(df)} rows ready. Generating quadrant plots...")
    print_summary_table(df)
    plot_runner_results(df, args.out_dir)
    plot_dependent_task_count_and_decision_time(df, args.out_dir, args.workloads)
    plot_dependent_task_count_and_decision_time(
        df, args.out_dir, args.workloads, task_kind="independent")
    plot_dependent_task_count_and_decision_time(
        df, args.out_dir, args.workloads, task_kind="independent_percentage")
    plot_dependent_task_count_and_decision_time(
        df, args.out_dir, args.workloads, task_kind="critical_duration")
    plot_runner_grouped_workload_bars(df, args.out_dir)
    plot_average_carbon_by_region(args.out_dir)
    plot_quadrants(df, args.out_dir)

    # ── Permutation analysis ──────────────────────────────────────────────────
    if args.permute:
        if not args.workloads:
            print("[ERROR] --workloads is required for permutation analysis.")
            sys.exit(1)

        with open(args.workloads) as f:
            all_workflows = json.load(f)

        # Pick workflows matching the deadline window
        candidate_wfs = [wf for wf in all_workflows
                         if wf.get("deadline_hours", 24) == args.permute_deadline]

        if not candidate_wfs:
            print(f"[ERROR] No workflows with deadline_hours={args.permute_deadline} found.")
            sys.exit(1)

        wfs_to_permute = candidate_wfs[:args.permute_max]
        print(f"[ANALYSIS] Permutation analysis: {len(wfs_to_permute)} workflows "
              f"({math.factorial(len(wfs_to_permute))} permutations)")

        submission = dt.datetime(2025, 6, 11, 13, 0, 0, tzinfo=dt.timezone.utc)
        deadline   = submission + dt.timedelta(hours=args.permute_deadline)

        run_permutation_analysis(
            wfs_to_permute, submission, deadline,
            out_dir=args.out_dir,
            max_workflows=args.permute_max
        )

    print(f"\n[ANALYSIS] Done. Outputs in {args.out_dir}")