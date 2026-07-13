#!/usr/bin/env python3
"""
Compare Alg3_RLE_Spatio against baseline algorithm(s) on the same workloads,
and report which workloads Alg3 performed worse on.

Assumes each workload is tested under multiple algorithms and that rows for
the same workload share a `run_id` (falls back to a composite structural key
of shape/n_tasks/deadline/scenario if run_id doesn't line up across algorithms).

Usage:
    python compare_alg3_vs_baselines.py results.csv
    python compare_alg3_vs_baselines.py results.csv --target Alg3_RLE_Spatio --out worse_for_alg3.csv
"""
import argparse
import sys

import pandas as pd

# Metrics where a HIGHER value is better for Alg3
HIGHER_IS_BETTER = {"success_rate", "deadline_fulfillment_pct"}

# Metrics where a LOWER value is better for Alg3
LOWER_IS_BETTER = {
    "total_carbon_with_transfer",
    "total_carbon",
    "total_transfer_carbon",
    "makespan_hours",
    "waiting_time_hours",
    "total_execution_time_hours",
    "total_energy",
    "total_hours",
}

FALLBACK_KEY = [
    "shape", "n_tasks", "total_core_hours", "critical_path_lb",
    "deadline_hours", "carbon_scenario", "origin_region", "load_scenario",
]

CONTEXT_COLS = [
    "shape", "n_tasks", "deadline_hours", "carbon_scenario",
    "origin_region", "load_scenario",
]


def load_data(path):
    df = pd.read_csv(path)
    if "algorithm" not in df.columns:
        sys.exit("CSV is missing the 'algorithm' column.")
    return df


def pick_join_key(df, target_algo):
    """Prefer run_id if it actually matches target rows to baseline rows;
    otherwise fall back to a composite structural key."""
    target_ids = set(df.loc[df["algorithm"] == target_algo, "run_id"])
    baseline_ids = set(df.loc[df["algorithm"] != target_algo, "run_id"])
    overlap = target_ids & baseline_ids

    if len(overlap) >= 0.5 * len(target_ids):
        return ["run_id"]

    missing = [c for c in FALLBACK_KEY if c not in df.columns]
    if missing:
        sys.exit(
            "run_id doesn't match across algorithms, and fallback key columns "
            f"are missing: {missing}. Can't join Alg3 rows to baseline rows."
        )
    print("Note: run_id didn't line up across algorithms — falling back to "
          "a composite workload key (shape/n_tasks/deadline/scenario/...).\n")
    return FALLBACK_KEY


def compare(df, target_algo, key_cols):
    target = df[df["algorithm"] == target_algo]
    baselines = df[df["algorithm"] != target_algo]

    if target.empty:
        sys.exit(f"No rows found for target algorithm '{target_algo}'.")
    if baselines.empty:
        sys.exit("No baseline rows found to compare against.")

    metrics = [m for m in (HIGHER_IS_BETTER | LOWER_IS_BETTER) if m in df.columns]
    rows = []

    for _, t_row in target.iterrows():
        match = pd.Series(True, index=baselines.index)
        for k in key_cols:
            match &= baselines[k] == t_row[k]
        b_rows = baselines[match]
        if b_rows.empty:
            continue  # no baseline run for this exact workload

        record = {"run_id": t_row.get("run_id")}
        record.update({c: t_row.get(c) for c in CONTEXT_COLS if c in df.columns})
        record["baseline_algorithms"] = ", ".join(sorted(b_rows["algorithm"].unique()))

        worse_metrics = []
        for m in metrics:
            alg3_val = t_row[m]
            baseline_avg = b_rows[m].mean()
            is_worse = alg3_val < baseline_avg if m in HIGHER_IS_BETTER else alg3_val > baseline_avg
            delta = alg3_val - baseline_avg  # negative = alg3 better for HIGHER_IS_BETTER metrics

            record[f"{m}_alg3"] = alg3_val
            record[f"{m}_baseline_avg"] = round(baseline_avg, 3)
            record[f"{m}_delta"] = round(delta, 3)
            if is_worse:
                worse_metrics.append(m)

        record["worse_on"] = ", ".join(worse_metrics)
        record["n_worse_metrics"] = len(worse_metrics)
        rows.append(record)

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path", help="Path to the benchmark results CSV")
    parser.add_argument("--target", default="Alg3_RLE_Spatio",
                         help="Algorithm name to evaluate (default: Alg3_RLE_Spatio)")
    parser.add_argument("--out", default="worse_for_alg3.csv",
                         help="Output CSV path for the full comparison report")
    args = parser.parse_args()

    df = load_data(args.csv_path)
    key_cols = pick_join_key(df, args.target)
    report = compare(df, args.target, key_cols)

    if "n_worse_metrics" not in report.columns or report.empty:
        print("No comparable workloads found between target and baselines.")
        return

    worse = report[report["n_worse_metrics"] > 0].sort_values(
        ["n_worse_metrics"], ascending=False
    )

    print(f"Workloads where {args.target} was worse than its baseline(s) on at "
          f"least one metric: {len(worse)} / {len(report)}\n")

    if not worse.empty:
        display_cols = [c for c in ["run_id"] + CONTEXT_COLS + ["baseline_algorithms", "worse_on"]
                         if c in worse.columns]
        print(worse[display_cols].to_string(index=False))

    report.to_csv(args.out, index=False)
    print(f"\nFull per-workload comparison (all metrics, all deltas) written to: {args.out}")


if __name__ == "__main__":
    main()