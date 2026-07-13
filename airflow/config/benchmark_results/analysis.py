"""
analysis.py
───────────
Post-benchmark analysis producing:
    1. Quadrant scatter plots (carbon vs makespan, carbon vs success rate, etc.)
    2. Permutation analysis — all orderings of a workflow trace, measuring
       max/min/percentile/variance of carbon and makespan across orderings

Usage:
    python analysis.py --results benchmark_results.csv
    python analysis.py --results benchmark_results.csv --permute-max 6
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
    REGIONS,
    DATA_SIZE_GB,
)

# ─────────────────────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────────────────────

ALG_COLORS = {
    'Alg1_GlobalWindow':     '#378ADD',
    'Alg2_TransferAware':    '#1D9E75',
    'Alg3_RLE_Spatio':       '#7F77DD',
    'Alg4_Batch_EDF':        '#D85A30',
    'Alg4_Batch_LWF':        '#D85A30',
    'Alg4_Batch':            '#D85A30',
    'BaselineA_Earliest':    '#888780',
    'BaselineB_LocalFirst':  '#BA7517',
    'BaselineC_Atomic':      '#D4537E',
    'Oracle':                '#1A1A2E',
}

ALG_MARKERS = {
    'Alg1_GlobalWindow':     'o',
    'Alg2_TransferAware':    's',
    'Alg3_RLE_Spatio':       'D',
    'Alg4_Batch_EDF':        '^',
    'Alg4_Batch_LWF':        'P',
    'Alg4_Batch':            '^',
    'BaselineA_Earliest':    'v',
    'BaselineB_LocalFirst':  'P',
    'BaselineC_Atomic':      'X',
    'Oracle':                '*',
}

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
        x_mid = x_means.mean()
        y_mid = y_means.mean()
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
            ax.annotate(alg.replace('_', ' '), (xv, yv),
                        textcoords='offset points', xytext=(8, 4),
                        fontsize=8, color=color, fontweight='bold', zorder=4)

        if not x_vals or not y_vals:
            ax.text(0.5, 0.5, 'No comparable data available',
                    ha='center', va='center', fontsize=10, color='#666')
            ax.set_axis_off()
        else:
            # Draw quadrant after setting limits so they're stable
            pad_x = (max(x_vals) - min(x_vals)) * 0.15 if x_vals else 1
            pad_y = (max(y_vals) - min(y_vals)) * 0.15 if y_vals else 1
            ax.set_xlim(min(x_vals) - pad_x, max(x_vals) + pad_x)
            ax.set_ylim(min(y_vals) - pad_y, max(y_vals) + pad_y)

            draw_quadrant(ax, np.array(x_vals), np.array(y_vals),
                          x_label, y_label, x_low_is_good, y_low_is_good)

            ax.set_xlabel(x_label, fontsize=11, fontweight='bold')
            ax.set_ylabel(y_label, fontsize=11, fontweight='bold')
            ax.set_title(title or f"{y_label} vs {x_label}", fontsize=13, pad=14)

            # Axis direction arrows hinting at 'better' direction
            x_arrow = '← better' if x_low_is_good else 'better →'
            y_arrow = '↓ better' if y_low_is_good else '↑ better'
            ax.annotate(x_arrow, xy=(0.99, 0.01), xycoords='axes fraction',
                        ha='right', va='bottom', fontsize=8, color='#555', style='italic')
            ax.annotate(y_arrow, xy=(0.01, 0.99), xycoords='axes fraction',
                        ha='left', va='top', fontsize=8, color='#555', style='italic')

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
# 2. Permutation analysis
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
    parser.add_argument("--results",      default="benchmark_results/benchmark_results.csv",
                        help="Path to benchmark_results.csv")
    parser.add_argument("--out-dir",      default="benchmark_results/analysis",
                        help="Directory for analysis plots")
    parser.add_argument("--workloads",    default=None,
                        help="Path to benchmark_workloads.json (needed for permutation analysis)")
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
    df["plan_produced"] = df["plan_produced"].astype(str).str.lower().isin(['true', '1'])
    print(f"[ANALYSIS] {len(df)} rows loaded. Generating quadrant plots...")
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