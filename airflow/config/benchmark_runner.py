"""
benchmark_runner.py
───────────────────
Runs all Carbon_Scheduler algorithms against a benchmark workload suite and
produces a structured results CSV + comparison plots.

Workload Profiles:
    - Standard (default): 6 shapes × 3 deadlines × 1 rep = 18 workflows, tasks 1-6h
    - Short tasks: 6 shapes × 3 deadlines × 2 reps = 36 workflows, tasks 0.25-2h
      To generate: python workload_generator.py --profile short --reps 2 --out benchmark_workloads_short.json
      Then run: python benchmark_runner.py --workloads benchmark_workloads_short.json
      Short-task variants make scheduling delays more relevant in batch operations.

Algorithms benchmarked:
    1  — Whole-workflow window search (single region)
    2  — Whole-workflow + network transfer penalties
    3  — Spatio-temporal task-level shifting with RLE (ACTIVE)
    4  — Alg4_Batch_EDF: Multi-workflow batch packing (Earliest Deadline First)
    42 — Alg4_Batch_LWF: Multi-workflow batch packing (Least Waste First)
    5  — Baseline A: Earliest start across all regions
    6  — Baseline B: Local-first (Portugal)
    7  — Baseline Atomic: Black-box continuous block
    8  — Oracle: Theoretical Best 
    9  — Atomic_Oracle: Theoretical Best for Atomic Continuous
    10 — Baseline Temporal: Single-Workflow Task-Level Temporal-Only
    11 — Baseline Spatial: Single-Workflow Task-Level Spatial-Only
    13 — Base_Temp_Batch_LWF: Batch Task-Level Temporal-Only (LWF)
    15 — Base_Spat_Batch_LWF: Batch Task-Level Spatial-Only (LWF)
"""

import os
import random
import sys
import json
import copy
import time
import argparse
import datetime as dt
import traceback
import types
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Import Carbon_Scheduler planning functions ────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

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
    plan_workflow_alg1,
    plan_workflow_alg2,
    plan_workflow_alg3,
    plan_batch_alg4,
    plan_baseline_A,
    plan_baseline_B,
    plan_atomic,
    plan_atomic_oracle,
    plan_oracle,
    plan_baseline_task_temporal_only,
    plan_baseline_task_spatial_only,
    plan_batch_baseline,
    load_carbon_data,
    load_cluster_state,
    save_cluster_state,
    calculate_standardized_stats,
    REGIONS,
    DATA_SIZE_GB,
    PLAN_DIR,
    WAITING_ROOM_DIR,
    TOTAL_CORES_PER_REGION,
)

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

ALGORITHM_MAP = {
    1: ("Alg1_GlobalWindow",     plan_workflow_alg1),
    2: ("Alg2_TransferAware",    plan_workflow_alg2),
    3: ("Alg3_RLE_Spatio",       plan_workflow_alg3),
    4: ("Alg4_Batch_EDF",        plan_batch_alg4),
    42:("Alg4_Batch_LWF",        plan_batch_alg4),
    5: ("BaselineA_Earliest",    plan_baseline_A),
    6: ("BaselineB_LocalFirst",  plan_baseline_B),
    7: ("BaselineC_Atomic",      plan_atomic),
    8: ("Oracle",                plan_oracle),
    9: ("Atomic_Oracle",         plan_atomic_oracle),
    10:("Base_Temporal",         plan_baseline_task_temporal_only),
    11:("Base_Spatial",          plan_baseline_task_spatial_only),
    13:("Base_Temp_Batch_LWF",   plan_batch_baseline),
    15:("Base_Spat_Batch_LWF",   plan_batch_baseline)
}

METRICS = [
    "success_rate",
    "total_carbon",
    "total_transfer_carbon",
    "total_carbon_with_transfer",
    "total_energy",
    "makespan_hours",
    "waiting_time_hours",
    "deadline_fulfillment_pct",
    "avg_task_execution_time_hours",
    "time_to_schedule_s",
    "decision_time_s",
]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _call_algorithm(alg_id: int, workflow: dict,
                    submission: dt.datetime, deadline: dt.datetime,
                    scenario_label: str, source_region: str = None) -> dict | None:
    """
    Dispatch to the correct planning function, normalising argument differences
    between algorithm signatures and supplying source_region context information.
    """
    safe_scenario = scenario_label.replace(" ", "_")
    run_id = f"{workflow['run_id']}_alg{alg_id}_{safe_scenario}"

    tasks = workflow["tasks"]
    sla   = workflow.get("sla_level", 95)

    edges = [
        (dep, t["id"])
        for t in tasks
        for dep in t.get("depends_on", [])
    ]

    try:
        if alg_id == 1:
            plan = plan_workflow_alg1(run_id, submission, deadline, tasks)
        elif alg_id == 2:
            plan = plan_workflow_alg2(run_id, submission, deadline, tasks, DATA_SIZE_GB,
                                      in_memory_state=load_cluster_state(), source_region=source_region)
        elif alg_id == 3:
            plan = plan_workflow_alg3(run_id, submission, deadline, tasks, edges, sla,
                                      in_memory_state=load_cluster_state(), source_region=source_region)
        elif alg_id == 5:
            plan = plan_baseline_A(run_id, submission, deadline, tasks, edges,
                                   load_cluster_state(), source_region=source_region)
        elif alg_id == 6:
            plan = plan_baseline_B(run_id, submission, deadline, tasks, edges,
                                   load_cluster_state(), source_region=source_region)
        elif alg_id == 7:
            plan = plan_atomic(run_id, submission, deadline, tasks, edges,
                               load_cluster_state(), source_region=source_region)
        elif alg_id == 10:
            plan = plan_baseline_task_temporal_only(run_id, submission, deadline, tasks, DATA_SIZE_GB,
                                                    in_memory_state=load_cluster_state(), source_region=source_region)
        elif alg_id == 11:
            plan = plan_baseline_task_spatial_only(run_id, submission, deadline, tasks, DATA_SIZE_GB,
                                                   in_memory_state=load_cluster_state(), source_region=source_region)
        else:
            plan = None
    except Exception as e:
        print(f"    [ERROR] alg{alg_id} / {run_id}: {e}")
        traceback.print_exc()
        plan = None

    return plan

# ─────────────────────────────────────────────────────────────────────────────
# Carbon scenario generation
# ─────────────────────────────────────────────────────────────────────────────

def _build_submission_regions(deadline_hours: int) -> list[dict]:
    origin_regions = REGIONS
    base_submission = dt.datetime(2026, 6, 11, 13, 0, 0, tzinfo=dt.timezone.utc)
    deadline_time = base_submission + dt.timedelta(hours=deadline_hours)
    
    windows = []
    for region in origin_regions:
        windows.append({
            "label": f"Origin_{region}",
            "origin_region": region,
            "submission": base_submission,
            "deadline": deadline_time
        })
    return windows

# ─────────────────────────────────────────────────────────────────────────────
# Background load generation
# ─────────────────────────────────────────────────────────────────────────────

LOAD_SCENARIOS = {
    "empty":   0.00,   
    "light":   0.25,   
    "medium":  0.50,   
    "heavy":   0.75,   
}

def generate_background_load(submission: dt.datetime, deadline: dt.datetime,
                             load_factor: float, seed: int = 0) -> dict:
    if load_factor == 0.0:
        return {}

    rng = random.Random(seed)
    state = {r: {} for r in REGIONS}
    cores_to_book = max(1, int(TOTAL_CORES_PER_REGION * load_factor))

    window_start = submission.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    window_end   = deadline.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    total_hours  = int((window_end - window_start).total_seconds() / 3600)

    for r in REGIONS:
        for h in range(total_hours):
            slot = (window_start + dt.timedelta(hours=h)).isoformat()
            jitter = rng.randint(-1, 1)
            booked = max(0, min(TOTAL_CORES_PER_REGION, cores_to_book + jitter))
            if booked > 0:
                state[r][slot] = booked

    return state

def run_batch_engine(workloads: list, base_submission: dt.datetime, deadline_h: int,
                     scenario_label: str, origin_region: str, alg_id: int) -> dict:
    """
    Writes workflow payloads to WAITING_ROOM_DIR and executes the target 
    batched routing routine, accounting for staggered arrival times.
    """
    merged_plan = {}
    written = []
    
    # Assign specific internal nomenclature tags based on algorithm ID
    if alg_id == 4:
        alg_suffix = "alg4_edf"
    elif alg_id == 42:
        alg_suffix = "alg4_lwf"
    elif alg_id == 8:
        alg_suffix = "oracle"
    elif alg_id == 9:
        alg_suffix = "atomic_oracle"
    elif alg_id == 13:
        alg_suffix = "base_temp_lwf"
    elif alg_id == 15:
        alg_suffix = "base_spat_lwf"
    else:
        alg_suffix = f"batch_{alg_id}"

    for wf in workloads:
        run_id_for_batch = wf["run_id"] + f"_{alg_suffix}_{scenario_label}"
        
        # Apply the staggered arrival time offset
        offset_seconds = wf.get("submit_time_relative_seconds", 0)
        wf_submission = base_submission + dt.timedelta(seconds=offset_seconds)
        wf_deadline = wf_submission + dt.timedelta(hours=deadline_h)

        payload = {
            "run_id":      run_id_for_batch,
            "submission":  wf_submission.isoformat(),
            "deadline":    wf_deadline.isoformat(),
            "tasks":       wf["tasks"],
            "edges": [
                (dep, t["id"])
                for t in wf["tasks"]
                for dep in t.get("depends_on", [])
            ],
            "sla": wf.get("sla_level", 95),
        }
        fname = os.path.join(WAITING_ROOM_DIR, f"waiting_{payload['run_id']}.json")
        with open(fname, "w") as f:
            json.dump(payload, f)
        written.append(fname)

    batch_data = []
    for fname in written:
        with open(fname) as f:
            batch_data.append(json.load(f))

    try:
        if alg_id == 4:
            batch_plan = plan_batch_alg4(batch_data, heuristic="EDF", source_region=origin_region)
        elif alg_id == 42:
            batch_plan = plan_batch_alg4(batch_data, heuristic="LWF", source_region=origin_region)
        elif alg_id == 8:
            batch_plan = plan_oracle(batch_data, source_region=origin_region)
        elif alg_id == 9:
            batch_plan = plan_atomic_oracle(batch_data, source_region=origin_region)
        elif alg_id == 13:
            batch_plan = plan_batch_baseline(batch_data, baseline_type="temporal", heuristic="LWF", source_region=origin_region)
        elif alg_id == 15:
            batch_plan = plan_batch_baseline(batch_data, baseline_type="spatial", heuristic="LWF", source_region=origin_region)
        else:
            batch_plan = {}

        if batch_plan:
            merged_plan.update(batch_plan)
    except Exception as e:
        print(f"    [ERROR] batch execution error for alg {alg_id}: {e}")
        traceback.print_exc()
    finally:
        for fname in written:
            try:
                os.remove(fname)
            except FileNotFoundError:
                pass

    return merged_plan

def run_benchmark(workloads: list, algorithms: list[int],
                  out_dir: str, max_workflows: int = None,
                  load_scenarios: list = None) -> pd.DataFrame:
    if load_scenarios is None:
        load_scenarios = ["empty", "medium"]

    # Create explicit folder directory to store snapshot states
    states_dir = os.path.join(out_dir, "states")
    Path(states_dir).mkdir(parents=True, exist_ok=True)

    if max_workflows:
        workloads = workloads[:max_workflows]

    rows = []
    combo_counter = 0
    out_csv = os.path.join(out_dir, "benchmark_results.csv")
    last_flushed_index = 0

    def _flush_results():
        nonlocal last_flushed_index
        if last_flushed_index >= len(rows):
            return

        chunk = rows[last_flushed_index:]
        file_exists = os.path.isfile(out_csv) and os.path.getsize(out_csv) > 0
        pd.DataFrame(chunk).to_csv(
            out_csv,
            mode="a" if file_exists else "w",
            header=not file_exists,
            index=False,
        )
        last_flushed_index = len(rows)

    # Isolate non-batch engines from the batch evaluation algorithms
    batch_alg_ids = [a for a in algorithms if a in [4, 42, 8, 9, 12, 13, 14, 15]]
    single_algs = [a for a in algorithms if a not in batch_alg_ids]
    BATCH_SIZE  = 6  

    # ── Single-workflow algorithms ──
    if single_algs:
        seen_combos = set()
        combos = []
        for wf in workloads:
            deadline_h = wf.get("deadline_hours", 24)
            for scenario in _build_submission_regions(deadline_h):
                for load_label in load_scenarios:
                    key = (deadline_h, scenario["label"], load_label)
                    if key not in seen_combos:
                        seen_combos.add(key)
                        combos.append((deadline_h, scenario, load_label))

        print(f"[RUNNER] Single-workflow evaluation will run {len(single_algs)} algorithms over {len(combos)} scenario/load combinations.")
        for alg_id in single_algs:
            alg_label = ALGORITHM_MAP[alg_id][0]
            print(f"[RUNNER] Starting sequential simulation block for {alg_label} ...")
            print(f"[RUNNER] {alg_label} will evaluate {len(combos)} scenario/load combos in up to {len(workloads)} workflows.")
            alg_start = time.perf_counter()

            for deadline_h, scenario, load_label in combos:
                batch_wfs = [wf for wf in workloads if wf.get("deadline_hours", 24) == deadline_h]
                if not batch_wfs:
                    continue

                submission = scenario["submission"]
                deadline   = scenario["deadline"]
                origin_reg = scenario["origin_region"]
                all_traces = {reg: load_carbon_data(reg, submission, deadline) for reg in REGIONS}
                total_batches = (len(batch_wfs) + BATCH_SIZE - 1) // BATCH_SIZE

                for i in range(0, len(batch_wfs), BATCH_SIZE):
                    sub_batch = batch_wfs[i:i + BATCH_SIZE]
                    batch_num = i // BATCH_SIZE + 1
                    print(f"  [{alg_label}] scenario={scenario['label']} load={load_label} batch={batch_num}/{total_batches} size={len(sub_batch)}")

                    load_factor = LOAD_SCENARIOS[load_label]
                    bg_seed = hash((deadline_h, load_label, i)) % 10000
                    bg_state = generate_background_load(submission, deadline, load_factor, seed=bg_seed)
                    save_cluster_state(bg_state)

                    for wf in sub_batch:
                        base_name = f"{wf['run_id']}_{alg_label}_{scenario['label']}_{load_label}"
                        
                        # --- NEW: Calculate the exact staggered times ---
                        offset_seconds = wf.get("submit_time_relative_seconds", 0)
                        actual_submission = submission + dt.timedelta(seconds=offset_seconds)
                        actual_deadline = actual_submission + dt.timedelta(hours=deadline_h)
                        # ------------------------------------------------
                        
                        pre_state = load_cluster_state()
                        with open(os.path.join(states_dir, f"{base_name}_before.json"), "w") as f:
                            json.dump(pre_state, f, indent=4)

                        t0 = time.perf_counter()
                        # Pass the actual staggered times to the algorithm
                        plan = _call_algorithm(
                            alg_id, wf, actual_submission, actual_deadline,
                            scenario["label"] + f"_{load_label}",
                            source_region=origin_reg
                        )
                        time_to_schedule_s = time.perf_counter() - t0

                        post_state = load_cluster_state()
                        with open(os.path.join(states_dir, f"{base_name}_after.json"), "w") as f:
                            json.dump(post_state, f, indent=4)

                        plan_is_empty = plan is not None and len(plan) == 0
                        plan_produced = plan is not None and len(plan) > 0

                        print(f"    [TIMING] {alg_label} {wf['run_id']} {scenario['label']} {load_label} -> {time_to_schedule_s:.3f}s")

                        row = {
                            "run_id":           wf["run_id"],
                            "algorithm":        alg_label,
                            "alg_id":           alg_id,
                            "shape":            wf.get("shape", "unknown"),
                            "n_tasks":          wf.get("n_tasks", len(wf["tasks"])),
                            "total_core_hours": wf.get("total_core_hours", 0),
                            "critical_path_lb": wf.get("critical_path_lb", 0),
                            "deadline_hours":   deadline_h,
                            "carbon_scenario":  scenario["label"],  
                            "origin_region":    origin_reg,         
                            "load_scenario":    load_label,
                            "plan_produced":    plan_produced or plan_is_empty, 
                            "time_to_schedule_s": round(time_to_schedule_s, 4),
                            "decision_time_s": round(time_to_schedule_s, 4),
                        }

                        if plan_produced:
                            stats = calculate_standardized_stats(
                                plan, all_traces, actual_submission.replace(tzinfo=None),
                                total_requested=wf["n_tasks"], deadline=actual_deadline
                            )
                            row.update(stats)
                        elif plan_is_empty:
                            stats = calculate_standardized_stats(
                                {}, all_traces, actual_submission.replace(tzinfo=None),
                                total_requested=wf["n_tasks"], deadline=actual_deadline
                            )
                            row.update(stats)
                        else:
                            for m in METRICS:
                                if m == "time_to_schedule_s":
                                    continue
                                row[m] = None

                        rows.append(row)
                        _flush_results()

                    combo_counter += 1
                    save_cluster_state({})

    # ── Multi-Workflow Batch Execution Algorithms (4, 42, 8, 9, 13, 15) ──────────────────
    for alg_id in batch_alg_ids:
        alg_label = ALGORITHM_MAP[alg_id][0]
        print(f"[RUNNER] Starting {alg_label} (batched across workloads)...")
        batch_start = time.perf_counter()

        if alg_id == 4:
            alg_suffix = "alg4_edf"
        elif alg_id == 42:
            alg_suffix = "alg4_lwf"
        elif alg_id == 9:
            alg_suffix = "atomic_oracle"
        elif alg_id == 13:
            alg_suffix = "base_temp_lwf"
        elif alg_id == 15:
            alg_suffix = "base_spat_lwf"
        else:
            alg_suffix = "oracle"

        # Sort strictly by Alibaba arrival order
        workloads_sorted = sorted(workloads, key=lambda x: x.get("submit_time_relative_seconds", 0))

        alg_total_time = 0.0
        alg_combo_count = 0

        # Generate scenario windows (origin regions)
        scenarios = _build_submission_regions(24)

        for scenario in scenarios:
            for load_label in load_scenarios:
                submission = scenario["submission"]
                origin_reg = scenario["origin_region"]

                total_batches = (len(workloads_sorted) + BATCH_SIZE - 1) // BATCH_SIZE

                for i in range(0, len(workloads_sorted), BATCH_SIZE):
                    # Slice contiguous sequential batches of size BATCH_SIZE (6)
                    sub_batch = workloads_sorted[i:i + BATCH_SIZE]
                    batch_num = i // BATCH_SIZE + 1

                    # Dynamically calculate the extended deadline window for carbon traces
                    max_offset_s = max(wf.get("submit_time_relative_seconds", 0) for wf in sub_batch)
                    max_dl_h = max(wf.get("deadline_hours", 24) for wf in sub_batch)
                    batch_trace_end = submission + dt.timedelta(seconds=max_offset_s) + dt.timedelta(hours=max_dl_h + 12)

                    all_traces = {reg: load_carbon_data(reg, submission, batch_trace_end) for reg in REGIONS}

                    print(f"  [{alg_label}] scenario={scenario['label']} load={load_label} batch={batch_num}/{total_batches} size={len(sub_batch)}")

                    load_factor  = LOAD_SCENARIOS[load_label]
                    bg_seed      = hash((load_label, alg_id, i)) % 10000
                    bg_state     = generate_background_load(submission, submission + dt.timedelta(hours=max_dl_h), load_factor, seed=bg_seed)
                    save_cluster_state(bg_state)

                    batch_base_name = f"batch_{alg_suffix}_b{i//BATCH_SIZE}_{scenario['label']}_{load_label}"

                    pre_batch_state = load_cluster_state()
                    with open(os.path.join(states_dir, f"{batch_base_name}_before.json"), "w") as f:
                        json.dump(pre_batch_state, f, indent=4)

                    t0 = time.perf_counter()
                    batch_plan = run_batch_engine(
                        sub_batch, 
                        submission, 
                        int(max_dl_h),
                        f"{scenario['label']}_{load_label}_b{i//BATCH_SIZE}",
                        origin_region=origin_reg,
                        alg_id=alg_id
                    )
                    
                    batch_time_to_schedule_s = time.perf_counter() - t0
                    alg_combo_count += 1
                    alg_total_time += batch_time_to_schedule_s
                    avg_time = alg_total_time / alg_combo_count
                    combo_counter += 1

                    print(f"    [PROGRESS] combo={combo_counter} alg={alg_label} elapsed={batch_time_to_schedule_s:.1f}s avg={avg_time:.1f}s")

                    post_batch_state = load_cluster_state()
                    with open(os.path.join(states_dir, f"{batch_base_name}_after.json"), "w") as f:
                        json.dump(post_batch_state, f, indent=4)

                    save_cluster_state(pre_batch_state)

                    for wf in sub_batch:
                        wf_run_id = wf["run_id"] + f"_{alg_suffix}_{scenario['label']}_{load_label}_b{i//BATCH_SIZE}"
                        wf_plan = batch_plan.get(wf_run_id, {})

                        row = {
                            "run_id":           wf["run_id"],
                            "algorithm":        alg_label,
                            "alg_id":           alg_id,
                            "shape":            wf.get("shape", "unknown"),
                            "n_tasks":          wf.get("n_tasks", len(wf["tasks"])),
                            "total_core_hours": wf.get("total_core_hours", 0),
                            "critical_path_lb": wf.get("critical_path_lb", wf.get("critical_path_hours", 0)),
                            "deadline_hours":   wf.get("deadline_hours", 24),
                            "carbon_scenario":  scenario["label"],
                            "origin_region":    origin_reg,
                            "load_scenario":    load_label,
                            "plan_produced":    wf_plan is not None and len(wf_plan) > 0,
                            "time_to_schedule_s": round(batch_time_to_schedule_s, 4),
                            "decision_time_s": round(batch_time_to_schedule_s, 4),
                        }

                        if row["plan_produced"]:
                            offset_seconds = wf.get("submit_time_relative_seconds", 0)
                            actual_submission = submission + dt.timedelta(seconds=offset_seconds)
                            actual_deadline = actual_submission + dt.timedelta(hours=wf.get("deadline_hours", 24))
                            
                            stats = calculate_standardized_stats(
                                wf_plan, all_traces, actual_submission.replace(tzinfo=None),
                                total_requested=wf["n_tasks"], deadline=actual_deadline
                            )
                            row.update(stats)
                        else:
                            for m in METRICS:
                                if m == "time_to_schedule_s":
                                    continue
                                row[m] = None

                        rows.append(row)
                        _flush_results()

        print(f"[RUNNER] Completed {alg_label} in {time.perf_counter() - batch_start:.2f}s over {alg_combo_count} batched combos.")

    df = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "benchmark_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"[RUNNER] Wrote benchmark results to {out_csv}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Analysis & plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_results(df: pd.DataFrame, out_dir: str):
    df_ok = df[df["plan_produced"] == True].copy()
    if df_ok.empty:
        print("[PLOTS] Warning: No valid runs produced a plan. Skipping plots.")
        return

    alg_order = sorted(df_ok["algorithm"].unique())
    colors    = plt.cm.Set2(np.linspace(0, 1, len(alg_order)))
    color_map = dict(zip(alg_order, colors))

    # 1. Total carbon box plot
    fig, ax = plt.subplots(figsize=(10, 5))
    data_by_alg = [df_ok[df_ok["algorithm"] == a]["total_carbon"].dropna().values for a in alg_order]
    bp = ax.boxplot(data_by_alg, patch_artist=True, notch=False)
    for patch, alg in zip(bp["boxes"], alg_order):
        patch.set_facecolor(color_map[alg])
    ax.set_xticklabels(alg_order, rotation=15, ha="right")
    ax.set_ylabel("Total Carbon (gCO₂eq)")
    ax.set_title("Carbon Footprint Distribution by Algorithm")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_carbon_boxplot.png"), dpi=150)
    plt.close(fig)

    # 2. Success rate bar chart
    fig, ax = plt.subplots(figsize=(10, 4))
    sr = df_ok.groupby("algorithm")["success_rate"].mean().reindex(alg_order)
    bars = ax.bar(alg_order, sr.values, color=[color_map[a] for a in alg_order], edgecolor="white")
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("Avg Success Rate (%)")
    ax.set_title("Average Task Success Rate by Algorithm")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_success_rate.png"), dpi=150)
    plt.close(fig)

    # 3. Makespan vs deadline pressure
    fig, ax = plt.subplots(figsize=(10, 5))
    for alg, color in color_map.items():
        subset = df_ok[df_ok["algorithm"] == alg]
        grouped = subset.groupby("deadline_hours")["makespan_hours"].mean()
        ax.plot(grouped.index, grouped.values, marker="o", label=alg, color=color)
    ax.set_xlabel("Deadline Window (hours)")
    ax.set_ylabel("Avg Makespan (hours)")
    ax.set_title("Makespan vs Deadline Pressure by Algorithm")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_makespan_vs_deadline.png"), dpi=150)
    plt.close(fig)

    # 4. Carbon by DAG shape heatmap
    pivot = df_ok.groupby(["algorithm", "shape"])["total_carbon"].mean().unstack(fill_value=np.nan).reindex(alg_order)
    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.2), 5))
    sns.heatmap(pivot, annot=True, fmt=".0f", cmap="RdYlGn_r", linewidths=0.5, ax=ax, cbar_kws={"label": "Avg gCO₂eq"})
    ax.set_title("Average Carbon by Algorithm × DAG Shape")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "plot_carbon_heatmap_shape.png"), dpi=150)
    plt.close(fig)


def plot_grouped_workload_bars(df: pd.DataFrame, out_dir: str):
    df_ok = df[df["plan_produced"] == True].copy()
    if df_ok.empty:
        return

    grouped_data = df_ok.groupby(["origin_region", "algorithm"])["total_carbon"].mean().reset_index()

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        data=grouped_data,
        x="origin_region",
        y="total_carbon",
        hue="algorithm",
        palette="Set2",
        edgecolor="black"
    )

    ax.set_ylabel("Mean Carbon Footprint (gCO₂eq)", fontsize=11, fontweight='bold')
    ax.set_xlabel("Workflow Submission Origin Region", fontsize=11, fontweight='bold')
    ax.set_title("Carbon Footprint Performance by Workflow Origin and Optimization Policy", fontsize=13, pad=15, fontweight='bold')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

    plt.xticks(rotation=0)
    plt.legend(title="Scheduling Engines", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()

    save_path = os.path.join(out_dir, "plot_workload_grouped_bars.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[PLOTS] Grouped workload bar chart generated successfully -> {save_path}")


def print_summary_table(df: pd.DataFrame):
    df_ok = df[df["plan_produced"] == True]
    if df_ok.empty:
        return
    summary = df_ok.groupby("algorithm")[METRICS].mean().round(2).sort_values("total_carbon")

    print("\n" + "═" * 90)
    print("BENCHMARK SUMMARY  (mean across all workflows × scenarios)")
    print("═" * 90)
    print(summary.to_string())
    print("═" * 90)

    failed = df[df["plan_produced"] == False]
    if not failed.empty:
        print(f"\n[WARNING] {len(failed)} runs produced no plan:")
        print(failed[["run_id", "algorithm", "carbon_scenario"]].to_string(index=False))

# ─────────────────────────────────────────────────────────────────────────────
# CLI Execution Engine
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Carbon_Scheduler benchmark runner")
    parser.add_argument("--workloads",   default="benchmark_workloads.json",
                        help="Path to workloads JSON dataset")
    parser.add_argument("--out-dir",     default="benchmark_results",
                        help="Directory for output metric arrays and charts")
    parser.add_argument("--algorithms",  nargs="+", type=int,
                        default=[8, 9, 42, 13, 15],  
                        help="Target scheduling algorithm identifiers")
    parser.add_argument("--max",         type=int, default=None,
                        help="Cap evaluation matrix sequence run counts")
    parser.add_argument("--load",        nargs="+", default=["empty", "medium"],
                        choices=list(LOAD_SCENARIOS.keys()),
                        help="Background thread core block allocations")
    args = parser.parse_args()

    print(f"[RUNNER] Loading workloads from {args.workloads} ...")
    with open(args.workloads) as f:
        workloads = json.load(f)
    print(f"[RUNNER] {len(workloads)} workflows loaded.")
    print(f"[RUNNER] Target evaluation engines: {args.algorithms}")
    print(f"[RUNNER] Destination storage path: {args.out_dir}\n")

    df = run_benchmark(
        workloads=workloads,
        algorithms=args.algorithms,
        out_dir=args.out_dir,
        max_workflows=args.max,
        load_scenarios=args.load,
    )

    print_summary_table(df)
    plot_results(df, args.out_dir)
    plot_grouped_workload_bars(df, args.out_dir)
    print("\n[RUNNER] Evaluation sequence execution finished.")