"""
alibaba_workload_generator.py
────────────────────────────────
Parses DAG jobs from Alibaba Cluster Trace v2018 (batch_task.csv).
- Uses raw 'cores' instead of core_hours.
- Uses multiplier 'N' to synthesize deadlines.
- Extracts and normalizes workflow arrival times for realistic simulation.
- Supports duration scaling and minimum duration thresholds for hourly scheduling.

Usage from the repository root:
    python airflow/config/data/alibaba_workload_generator.py --help
    python airflow/config/data/alibaba_workload_generator.py \
        --csv airflow/config/data/batch_task.csv \
        --out airflow/config/data/benchmark_workloads_alibaba_factor100.json \
        --max 50 --N 10 --scale 100 --min-dur 0.25

The generated JSON can be passed to benchmark_runner.py with its
`--workloads` option. Use `--max` to limit workflows, `--N` to control the
deadline multiplier, and `--scale` to scale task durations into simulation hours.
"""

import pandas as pd
import json
import argparse
import os
import re

def parse_alibaba_dag_task(task_name: str):
    """
    Validates Alibaba DAG task naming convention (e.g., M1, M2_1, R4_2).
    """
    match = re.match(r'^[A-Za-z](\d+)((?:_\d+)*)$', str(task_name).strip())
    if not match:
        return None, None
    
    parts = str(task_name).strip().split('_')
    task_id = f"task_{parts[0][1:]}"
    depends_on = [f"task_{d}" for d in parts[1:]]
    return task_id, depends_on

def calculate_critical_path(tasks: list) -> float:
    """
    Computes the critical path (longest execution path) in hours.
    """
    task_map = {t["id"]: t for t in tasks}
    durations = {t["id"]: t["dur"] for t in tasks}
    memo = {}

    def get_max_path_to(task_id):
        if task_id in memo:
            return memo[task_id]
        
        task = task_map[task_id]
        valid_deps = [d for d in task["depends_on"] if d in task_map]
        
        max_pred = max([get_max_path_to(d) for d in valid_deps], default=0.0) if valid_deps else 0.0
            
        memo[task_id] = max_pred + durations[task_id]
        return memo[task_id]

    return max([get_max_path_to(t["id"]) for t in tasks], default=0.0)

def parse_alibaba_trace(
    csv_path: str, 
    output_json: str, 
    max_workflows: int = 50, 
    N_multiplier: float = 2.0,
    scale_factor: float = 10.0,
    min_dur_hours: float = 0.25
):
    if not os.path.exists(csv_path):
        print(f"[ERROR] Could not find {csv_path}.")
        return

    names = [
        "task_name", "instance_num", "job_name", "task_type", 
        "status", "start_time", "end_time", "plan_cpu", "plan_mem"
    ]
    
    print(f"Loading Alibaba trace from {csv_path}...")
    df = pd.read_csv(csv_path, names=names, nrows=2000000) 
    df = df[df['status'] == 'Terminated'].dropna(subset=['start_time', 'end_time'])
    
    workflows = []
    grouped = df.groupby('job_name')
    
    for job_name, group in grouped:
        if len(group) < 3 or len(group) > 15:
            continue
            
        tasks_payload = []
        valid_dag_job = True
        job_start_time = float('inf')
        
        for _, row in group.iterrows():
            task_id, depends_on = parse_alibaba_dag_task(row['task_name'])
            
            if task_id is None:
                valid_dag_job = False
                break
                
            duration_sec = row['end_time'] - row['start_time']
            if duration_sec <= 0:
                valid_dag_job = False
                break
                
            # Convert to base hours, scale up by scale_factor, and enforce min_dur_hours threshold
            raw_hours = duration_sec / 3600.0
            duration_hours = raw_hours * scale_factor
            
            # Use raw cores directly
            cores_needed = row['plan_cpu'] if pd.notnull(row['plan_cpu']) and row['plan_cpu'] > 0 else 1.0
            
            # Track the earliest start time of any task to represent the job's arrival
            if row['start_time'] < job_start_time:
                job_start_time = row['start_time']
            
            tasks_payload.append({
                "id": task_id,
                "depends_on": depends_on,
                "dur": round(duration_hours, 4),
                "cores": cores_needed / 100
            })
            
        if not valid_dag_job or not tasks_payload:
            continue

        all_task_ids = {t["id"] for t in tasks_payload}
        for t in tasks_payload:
            t["depends_on"] = [d for d in t["depends_on"] if d in all_task_ids]

        cp_hours = calculate_critical_path(tasks_payload)
        deadline_hours = max(2.0, round(cp_hours * N_multiplier, 2))

        workflows.append({
            "run_id": f"alibaba_{job_name}",
            "raw_start_time": job_start_time,
            "critical_path_hours": round(cp_hours, 4),
            "deadline_multiplier_N": N_multiplier,
            "deadline_hours": deadline_hours,
            "sla_level": 95, 
            "n_tasks": len(tasks_payload),
            "shape": "alibaba_real_dag",
            "tasks": tasks_payload
        })
        
        if len(workflows) >= max_workflows:
            break

    # Normalize submission times so the simulation starts at T=0
    if workflows:
        global_min_start = min(wf["raw_start_time"] for wf in workflows)
        
        # Sort workflows by arrival time to make it sequential for the runner
        workflows.sort(key=lambda x: x["raw_start_time"])
        
        for wf in workflows:
            wf["submit_time_relative_seconds"] = int(wf["raw_start_time"] - global_min_start)
            del wf["raw_start_time"]
            
    with open(output_json, 'w') as f:
        json.dump(workflows, f, indent=4)
        
    print(f"[SUCCESS] Extracted {len(workflows)} valid DAG workflows.")
    print(f"Applied duration scale multiplier: {scale_factor}x (min threshold: {min_dur_hours}h)")
    print(f"Used N={N_multiplier} to calculate deadlines.")
    print("Workflows are sorted by arrival time (submit_time_relative_seconds).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="batch_task.csv")
    parser.add_argument("--out", default="benchmark_workloads_alibaba.json")
    parser.add_argument("--max", type=int, default=50)
    parser.add_argument("--N", type=float, default=10.0, help="Deadline multiplier")
    parser.add_argument("--scale", type=float, default=100.0, help="Multiplier to scale task durations up into full hours")
    parser.add_argument("--min-dur", type=float, default=0.25, help="Minimum duration floor in hours per task")
    args = parser.parse_args()

    parse_alibaba_trace(
        csv_path=args.csv, 
        output_json=args.out, 
        max_workflows=args.max, 
        N_multiplier=args.N,
        scale_factor=args.scale,
        min_dur_hours=args.min_dur
    )