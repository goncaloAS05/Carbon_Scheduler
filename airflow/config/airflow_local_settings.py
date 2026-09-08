import os
import json
import fcntl
import copy
import logging
import datetime as dt
import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from airflow.exceptions import AirflowRescheduleException

AIRFLOW_BASE_DIR = os.path.expanduser("~/Carbon_Scheduler/airflow")
CLUSTER_FILE     = os.path.join(AIRFLOW_BASE_DIR, "cluster_state.json")
DATA_DIR         = os.path.join(AIRFLOW_BASE_DIR, "plugins", "data", "")
PLAN_DIR         = os.path.join(AIRFLOW_BASE_DIR, "plans", "")
ORDERS_DIR       = os.path.join(AIRFLOW_BASE_DIR, "orders", "")
HISTORY_FILE     = os.path.join(AIRFLOW_BASE_DIR, "task_history.json")
WAITING_ROOM_DIR = os.path.join(AIRFLOW_BASE_DIR, "waiting_room", "")
LOG_DIR          = os.path.join(AIRFLOW_BASE_DIR, "logs")
CARBON_LOG_FILE  = os.path.join(LOG_DIR, "carbon_scheduler.log")

os.makedirs(LOG_DIR, exist_ok=True)

carbon_logger = logging.getLogger("carbon_scheduler")
if not carbon_logger.handlers:
    file_handler = logging.FileHandler(CARBON_LOG_FILE, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    carbon_logger.addHandler(file_handler)
carbon_logger.setLevel(logging.INFO)
carbon_logger.propagate = False


def _log(message, level=logging.INFO):
    """Write scheduler diagnostics to the shared file logger."""
    carbon_logger.log(level, message)

DEBUG = True
ACTIVE_ALGORITHM = 15
BATCH_SIZE = 3
TOTAL_CORES_PER_REGION = 6
CORES_LIMIT = TOTAL_CORES_PER_REGION
REGIONS_FILE = os.path.join(AIRFLOW_BASE_DIR, "plugins", "data", "regions_123.txt")


def _load_regions_from_file(path):
    """Read the configured region codes from disk or fall back to the default set."""
    if not os.path.exists(path):
        return ["DE", "PL", "PT", "ES"]

    regions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            region = line.strip()
            if not region or region.startswith("#"):
                continue
            regions.append(region)

    return regions or ["DE", "PL", "PT", "ES"]

REGIONS = _load_regions_from_file(REGIONS_FILE)

DATA_SIZE_GB = 1
TRANSFER_SPEED_GBPS = 400 / 8
KWH_PER_GB = 0.001875
DEADLINE_PENALTY_MULTIPLIER = 1.5

os.makedirs(PLAN_DIR, exist_ok=True)
os.makedirs(ORDERS_DIR, exist_ok=True)
os.makedirs(WAITING_ROOM_DIR, exist_ok=True)

def plot_optimization_window_heatmap(best_manifest, all_regions_data, submission_time, deadline, plot_name="default", stats=None):
    """Render a carbon heatmap for a workflow window with task placements overlaid."""
    if not DEBUG:
        return
    sub_naive = submission_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    dl_naive = deadline.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    total_window_hours = int((dl_naive - sub_naive).total_seconds() // 3600) + 1
    
    time_slots = [(sub_naive + dt.timedelta(hours=i)) for i in range(total_window_hours)]
    time_labels = [t.strftime("%H:00\n%d/%m") for t in time_slots]
    
    regions = list(all_regions_data.keys())
    heatmap_matrix = np.zeros((len(regions), len(time_slots)))
    
    for r_idx, reg in enumerate(regions):
        reg_intensities = all_regions_data[reg]
        for t_idx, t_slot in enumerate(time_slots):
            t_lookup = t_slot.replace(minute=0, second=0, microsecond=0)
            if t_lookup in reg_intensities:
                heatmap_matrix[r_idx, t_idx] = reg_intensities[t_lookup]
            elif reg_intensities:
                closest_t = min(reg_intensities.keys(), key=lambda d: abs((d - t_lookup).total_seconds()))
                heatmap_matrix[r_idx, t_idx] = reg_intensities[closest_t]
            else:
                heatmap_matrix[r_idx, t_idx] = 450

    plt.figure(figsize=(max(14, total_window_hours * 0.6), 8.5))
    df = pd.DataFrame(heatmap_matrix, index=regions, columns=time_labels)
    
    ax = sns.heatmap(df, cmap="RdYlGn_r", annot=True, fmt=".0f", annot_kws={"size": 7},
                     cbar_kws={'label': 'gCO2eq/kWh'})

    label_collision_counter = {}

    for task_id, data in best_manifest.items():
        if data['region'] in regions:
            reg_row = regions.index(data['region'])
            t_start = dt.datetime.fromisoformat(data['start']).replace(tzinfo=None)
            duration = data.get('dur', 1) 

            hour_idx = int((t_start - sub_naive).total_seconds() // 3600)
            
            if 0 <= hour_idx < len(time_slots):
                ax.add_patch(plt.Rectangle((hour_idx, reg_row), duration, 1, 
                                           fill=False, edgecolor='blue', lw=3, zorder=10, alpha=0.6))
                
                cell_key = (hour_idx, reg_row)
                offset_idx = label_collision_counter.get(cell_key, 0)
                label_collision_counter[cell_key] = offset_idx + 1
                
                y_pos = reg_row + 0.2 + (offset_idx * 0.15)
                display_id = str(task_id).split('__')[-1]
                
                plt.text(hour_idx + 0.1, y_pos, display_id, 
                         color='blue', weight='bold', ha='left', va='center', 
                         fontsize=8, zorder=11, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0))

    if stats:
        stats_str = (
            f"Success Rate: {stats['success_rate']}%   |   "
            f"Execution Time: {stats.get('total_execution_time_hours', 0)}h   |   "
            f"Makespan: {stats.get('makespan_hours', 0)}h   |   "
            f"Waiting Time: {stats.get('waiting_time_hours', 0)}h\n"
            f"Avg Task Dur: {stats.get('avg_task_execution_time_hours', 0)}h   |   "
            f"Deadline Fulfilled: {stats.get('deadline_fulfillment_pct', 100.0)}%   |   "
            f"Energy: {stats['total_energy']} kWh   |   "
            f"Execution Carbon: {stats['total_carbon']} gCO2\n"
            f"Transfer Carbon: {stats.get('total_transfer_carbon', 0.0)} gCO2   |   "
            f"Total Carbon (incl. transfer): {stats.get('total_carbon_with_transfer', stats['total_carbon'])} gCO2"
        )
        
        plt.gcf().text(0.5, 0.03, stats_str, fontsize=10.5,
                       ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='#f9f9f9', alpha=1.0, edgecolor='gray'))

    plt.title(f"Carbon Opportunity Window: {total_window_hours} Hours Total", fontsize=14, pad=25)
    plt.xticks(rotation=0)
    
    plt.tight_layout(rect=[0, 0.15, 1, 0.95])

    save_path = os.path.join(PLAN_DIR, f"heatmap_{plot_name}.png")
    plt.savefig(save_path) 
    plt.close()

    
def calculate_standardized_stats(plan, all_traces, window_start, total_requested, deadline=None):
    """Aggregate execution, carbon, deadline, and timing metrics for a manifest."""
    total_energy = 0.0
    total_carbon = 0.0
    total_transfer_carbon = 0.0
    planned_starts = []
    planned_ends = []
    task_durations = []
    active_hours = set()
    POWER_FACTOR = 0.2

    successful_tasks = 0
    naive_deadline = deadline.replace(tzinfo=None) if deadline else None

    for tid, tdata in plan.items():
        task_energy = tdata['cores'] * tdata['dur'] * POWER_FACTOR
        total_energy += task_energy

        start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
        end_dt = start_dt + dt.timedelta(hours=tdata['dur'])
        
        planned_starts.append(start_dt)
        task_durations.append(tdata['dur'])
        reg = tdata['region']

        for h in range(int(tdata['dur'])):
            hour_obj = start_dt + dt.timedelta(hours=h)
            intensity = all_traces.get(reg, {}).get(hour_obj, 400)
            total_carbon += (tdata['cores'] * POWER_FACTOR) * intensity
            active_hours.add(hour_obj.isoformat())

        planned_ends.append(end_dt)
        total_transfer_carbon += tdata.get('transfer_carbon_g', 0.0)

        if naive_deadline and end_dt > naive_deadline:
            pass
        else:
            successful_tasks += 1

    success_rate = round((successful_tasks / total_requested) * 100, 1) if total_requested > 0 else 0

    if plan and planned_starts and planned_ends:
        first_start = min(planned_starts)
        last_end = max(planned_ends)
        total_execution_time = len(active_hours)
        makespan = round((last_end - first_start).total_seconds() / 3600, 2)
        waiting_time = round((first_start - window_start).total_seconds() / 3600, 2)
        avg_task_execution = round(sum(task_durations) / len(task_durations), 2)
        
        deadline_fulfilled_pct = 100.0 if (naive_deadline and last_end <= naive_deadline) else 0.0
        total_hours = round((last_end - window_start).total_seconds() / 3600, 2)
    else:
        total_execution_time = 0
        makespan = 0
        waiting_time = 0
        avg_task_execution = 0
        deadline_fulfilled_pct = 0.0
        total_hours = 0

    total_carbon_with_transfer = total_carbon + total_transfer_carbon
    total_core_hours = total_energy / POWER_FACTOR if POWER_FACTOR > 0 else 0.0
    
    carbon_per_core_hour = round(total_carbon_with_transfer / total_core_hours, 4) if total_core_hours > 0 else 0.0
    carbon_per_success_pct = round(total_carbon_with_transfer / success_rate, 4) if success_rate > 0 else 0.0
    carbon_per_makespan_hour = round(total_carbon_with_transfer / makespan, 4) if makespan > 0 else 0.0
    makespan_hours_per_carbon = round(makespan / total_carbon_with_transfer, 6) if total_carbon_with_transfer > 0 else 0.0

    return {
        "success_rate": success_rate,
        "total_hours": total_hours,
        "total_energy": round(total_energy, 2),
        "total_carbon": round(total_carbon, 2),
        "total_transfer_carbon": round(total_transfer_carbon, 2),
        "total_carbon_with_transfer": round(total_carbon_with_transfer, 2), 
        "total_execution_time_hours": total_execution_time,
        "makespan_hours": makespan,
        "waiting_time_hours": waiting_time,
        "avg_task_execution_time_hours": avg_task_execution,
        "deadline_fulfillment_pct": deadline_fulfilled_pct,
        "carbon_per_core_hour": carbon_per_core_hour,
        "carbon_per_success_pct": carbon_per_success_pct,
        "carbon_per_makespan_hour": carbon_per_makespan_hour,
        "makespan_hours_per_carbon": makespan_hours_per_carbon
    }

def find_earliest_slot(region, requested_cores, state, not_before=None, deadline=None):
    """Return the earliest feasible hourly slot for a task in a region."""
    anchor = not_before if not_before else dt.datetime.now()
    anchor = anchor.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    if deadline:
        deadline = deadline.replace(minute=0, second=0, microsecond=0, tzinfo=None)

    for h in range(240):
        slot_dt = anchor + dt.timedelta(hours=h)
        
        if deadline and slot_dt > deadline:
            break
            
        slot = slot_dt.isoformat()
        used = state.get(region, {}).get(slot, 0)
        if (used + requested_cores) <= CORES_LIMIT:
            return slot_dt
            
    return None



def save_chosen_path(batch_label, algorithm, order_run_ids, task_regions,
                      exec_carbon_g, transfer_carbon_g, extra=None):
    """Persist one batch schedule so it can be compared against other algorithms."""
    record = {
        "algorithm":          algorithm,
        "batch_label":        batch_label,
        "workflow_order":     order_run_ids,
        "task_regions":       task_regions,
        "exec_carbon_g":      round(exec_carbon_g, 2),
        "transfer_carbon_g":  round(transfer_carbon_g, 2),
        "total_carbon_g":     round(exec_carbon_g + transfer_carbon_g, 2),
    }
    if extra:
        record.update(extra)

    safe_label = "".join(c if c.isalnum() or c in "._-" else "_" for c in batch_label)
    out_path = os.path.join(ORDERS_DIR, f"order_{safe_label}_{algorithm}_{int(dt.datetime.now().timestamp())}.json")
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return out_path


def update_scratchpad(state, region, start_dt, duration, cores):
    """Reserve hourly CPU capacity for a task in the in-memory cluster state."""
    if region not in state:
        state[region] = {}
    
    t_lookup = start_dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    for h in range(int(duration)):
        ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
        
        current_usage = state[region].get(ts, 0)
        state[region][ts] = current_usage + cores
        
    return state



def calculate_transfer_penalty(target_region, data_size_gb, source_region=REGIONS[0]):
    """Estimate the transfer latency and carbon cost for moving data between regions."""
    if target_region == source_region:
        return 0, 0

    transfer_speed_gbps = 400 / 8
    transfer_seconds = data_size_gb / transfer_speed_gbps
    transfer_time_h = transfer_seconds / 3600

    transfer_carbon_cost = data_size_gb * KWH_PER_GB * 250

    return transfer_time_h, transfer_carbon_cost

def load_carbon_data(region, start_limit, end_limit):
    """Load hourly carbon-intensity values for a region and remap them to the planner timeline."""
    df_path = os.path.join(DATA_DIR, f"log_{region}.csv")
    if not os.path.exists(df_path):
        return {}

    df = pd.read_csv(df_path)
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)']).dt.tz_localize(None)

    target_start = start_limit.replace(year=2025, tzinfo=None)
    target_end = end_limit.replace(year=2025, tzinfo=None)

    search_start = target_start
    search_end = target_end

    mask = (df['Datetime (UTC)'] >= search_start) & (df['Datetime (UTC)'] <= search_end)
    mask_results = df[mask]

    return {row['Datetime (UTC)'].replace(year=2026): row['Carbon intensity gCO₂eq/kWh (Life cycle)']
            for _, row in mask_results.iterrows()}

def load_cluster_state(file_path=CLUSTER_FILE):
    """Reads the current core usage from the JSON file."""
    if not os.path.exists(file_path):
        return {}
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}

def save_cluster_state(state_dict, file_path=CLUSTER_FILE):
    """Persists the updated core usage back to the JSON file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(state_dict, f, indent=4)
    except IOError as e:
        _log(f"Error saving cluster state: {e}", logging.ERROR)

def get_dynamic_critical_path(tasks_metadata, edges=None, sla_level=None):
    """Compute the workflow critical-path length from the task dependency graph."""
    if not tasks_metadata:
        return 0.0

    durations = {t['id']: t.get('dur', 1.0) for t in tasks_metadata}
    
    graph = {t['id']: [] for t in tasks_metadata}
    in_degree = {t['id']: 0 for t in tasks_metadata}
    
    for t in tasks_metadata:
        child_id = t['id']
        parents = t.get('depends_on', [])
        
        for parent_id in parents:
            if parent_id in graph:
                graph[parent_id].append(child_id)
                in_degree[child_id] += 1

    earliest_start = {t['id']: 0 for t in tasks_metadata}
    
    queue = [t['id'] for t in tasks_metadata if in_degree[t['id']] == 0]
    
    while queue:
        u = queue.pop(0)
        
        for v in graph[u]:
            earliest_start[v] = max(earliest_start[v], earliest_start[u] + durations[u])
            
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    cp_length = max(earliest_start[n] + durations[n] for n in earliest_start)
    return cp_length

def _apply_late_carbon_penalty(plan, all_traces, stats, multiplier):
    """Apply a deadline penalty to late tasks after the base manifest is built."""
    POWER_FACTOR = 0.2
    penalized_carbon = 0.0

    for tid, tdata in plan.items():
        is_late = tdata.get('scheduled_late', False)
        mult = multiplier if is_late else 1.0
        start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
        reg = tdata['region']

        for h in range(int(tdata['dur'])):
            hour_obj = start_dt + dt.timedelta(hours=h)
            intensity = all_traces.get(reg, {}).get(hour_obj, 400)
            penalized_carbon += (tdata['cores'] * POWER_FACTOR) * intensity * mult

    stats['total_carbon'] = round(penalized_carbon, 2)
    stats['total_carbon_with_transfer'] = round(
        penalized_carbon + stats.get('total_transfer_carbon', 0.0), 2
    )
    return stats

def plan_baseline_A(run_id, submission, deadline, tasks, edges, in_memory_state, source_region=None):
    """Baseline A: Immediate Schedule (Earliest Start across all regions)"""
    if source_region is None:
        _log(f"[WARNING] No source region provided for {run_id}. Defaulting to {REGIONS[0]}.", logging.WARNING)
        source_region = REGIONS[0]

    _log(f"[BASELINE A] Planning {run_id} originating from {source_region}.")
    plan = {}

    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(submission)
    dl_dt  = deadline   if isinstance(deadline,  dt.datetime) else dt.datetime.fromisoformat(deadline)
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt  = dl_dt.replace(tzinfo=None)

    

    task_map      = {t['id']: t for t in tasks}
    finish_times  = {}
    max_end_time  = dl_dt

    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in REGIONS}

    for task in tasks:
        parent_finish = max(
            (finish_times[dep] for dep in task.get("depends_on", []) if dep in finish_times),
            default=sub_dt
        )

        data_size = sum(
            task_map[p_id].get('output_size_gb', DATA_SIZE_GB)
            for p_id in task.get('depends_on', [])
            if p_id in task_map
        ) or DATA_SIZE_GB

        best_start  = None
        best_region = None

        for reg in REGIONS:
            time_penalty, _ = calculate_transfer_penalty(reg, data_size, source_region=source_region)
            adjusted_ready_time = parent_finish + dt.timedelta(hours=time_penalty)

            slot = find_earliest_slot(reg, task["cores"], in_memory_state,
                                      not_before=adjusted_ready_time, deadline=dl_dt)
            if slot is not None:
                if best_start is None or slot < best_start:
                    best_start  = slot
                    best_region = reg

        is_late = False

        if best_start is None:
            _log(f"[BASELINE A] Task {task['id']} over core limit inside deadline. Scheduling late...")
            is_late = True
            
            extended_deadline = parent_finish + dt.timedelta(hours=720)
            for reg in REGIONS:
                time_penalty, _ = calculate_transfer_penalty(reg, data_size, source_region=source_region)
                adjusted_ready_time = parent_finish + dt.timedelta(hours=time_penalty)

                slot = find_earliest_slot(reg, task["cores"], in_memory_state,
                                          not_before=adjusted_ready_time, deadline=extended_deadline)
                if slot is not None:
                    if best_start is None or slot < best_start:
                        best_start  = slot
                        best_region = reg

            if best_start is None:
                best_start  = parent_finish
                best_region = source_region

        finish_times[task["id"]] = best_start + dt.timedelta(hours=task["dur"])
        max_end_time = max(max_end_time, finish_times[task["id"]])

        _, task_transfer_carbon = calculate_transfer_penalty(best_region, data_size, source_region=source_region)

        plan[task["id"]] = {
            "region":            best_region,
            "start":             best_start.isoformat(),
            "dur":               task["dur"],
            "cores":             task["cores"],
            "transfer_carbon_g": round(task_transfer_carbon, 2),
            "scheduled_late":    is_late,
            "carbon_penalty_multiplier": DEADLINE_PENALTY_MULTIPLIER if is_late else 1.0,
        }
        in_memory_state = update_scratchpad(
            in_memory_state, best_region, best_start, task["dur"], task["cores"]
        )

    save_cluster_state(in_memory_state)

    stats_traces = all_traces
    if max_end_time > dl_dt:
        stats_traces = {reg: load_carbon_data(reg, sub_dt, max_end_time) for reg in REGIONS}

    stats = calculate_standardized_stats(plan, stats_traces, sub_dt, len(tasks), deadline=dl_dt)
    stats = _apply_late_carbon_penalty(plan, stats_traces, stats, DEADLINE_PENALTY_MULTIPLIER)

    plot_optimization_window_heatmap(
        plan, stats_traces, sub_dt, max_end_time,
        plot_name=f"baseline_A_{run_id}",
        stats=stats
    )

    plan_path = os.path.join(PLAN_DIR, f"plan_{run_id}.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f)

    return plan

def plan_baseline_B(run_id, submission, deadline, tasks, edges, in_memory_state, source_region=None):
    """Baseline B: Local-First (Default Origin Location)"""
    if source_region is None:
        _log(f"[WARNING] No source region provided for {run_id}. Defaulting to {REGIONS[0]}.", logging.WARNING)
        source_region = REGIONS[0]

    _log(f"[BASELINE B] Planning {run_id} local to region {source_region}...")
    plan = {}

    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(str(submission))
    dl_dt  = deadline   if isinstance(deadline,  dt.datetime) else dt.datetime.fromisoformat(str(deadline))
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt  = dl_dt.replace(tzinfo=None)

    DEADLINE_PENALTY_MULTIPLIER = 1.5
    task_map = {t['id']: t for t in tasks}
    finish_times = {}
    max_end_time = dl_dt

    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in REGIONS}

    for task in tasks:
        parent_finish = max(
            (finish_times[dep] for dep in task.get("depends_on", []) if dep in finish_times),
            default=sub_dt
        )

        data_size = sum(
            task_map[p_id].get('output_size_gb', DATA_SIZE_GB)
            for p_id in task.get('depends_on', [])
            if p_id in task_map
        ) or DATA_SIZE_GB

        best_start = None
        chosen_reg = None

        slot_local = find_earliest_slot(source_region, task["cores"], in_memory_state,
                                        not_before=parent_finish, deadline=dl_dt)
        if slot_local is not None:
            best_start = slot_local
            chosen_reg = source_region
        else:
            for reg in REGIONS:
                if reg == source_region:
                    continue
                time_penalty, _ = calculate_transfer_penalty(reg, data_size, source_region=source_region)
                adjusted_ready_time = parent_finish + dt.timedelta(hours=time_penalty)
                earliest_elsewhere = find_earliest_slot(reg, task["cores"], in_memory_state,
                                                        not_before=adjusted_ready_time, deadline=dl_dt)
                if earliest_elsewhere is not None:
                    if best_start is None or earliest_elsewhere < best_start:
                        best_start = earliest_elsewhere
                        chosen_reg = reg

        is_late = False

        if best_start is None:
            _log(f"[BASELINE B] Task {task['id']} over core limit inside deadline. Scheduling late...")
            is_late = True
            extended_deadline = parent_finish + dt.timedelta(hours=720)
            
            for reg in REGIONS:
                time_penalty, _ = calculate_transfer_penalty(reg, data_size, source_region=source_region)
                adjusted_ready_time = parent_finish + dt.timedelta(hours=time_penalty)
                earliest_elsewhere = find_earliest_slot(reg, task["cores"], in_memory_state,
                                                        not_before=adjusted_ready_time, deadline=extended_deadline)
                if earliest_elsewhere is not None:
                    if best_start is None or earliest_elsewhere < best_start:
                        best_start = earliest_elsewhere
                        chosen_reg = reg

            if best_start is None:
                best_start = parent_finish
                chosen_reg = source_region

        finish_times[task["id"]] = best_start + dt.timedelta(hours=task["dur"])
        max_end_time = max(max_end_time, finish_times[task["id"]])

        _, task_transfer_carbon = calculate_transfer_penalty(chosen_reg, data_size, source_region=source_region)

        plan[task["id"]] = {
            "region":            chosen_reg,
            "start":             best_start.isoformat(),
            "dur":               task["dur"],
            "cores":             task["cores"],
            "transfer_carbon_g": round(task_transfer_carbon, 2),
            "scheduled_late":    is_late,
            "carbon_penalty_multiplier": DEADLINE_PENALTY_MULTIPLIER if is_late else 1.0,
        }
        in_memory_state = update_scratchpad(
            in_memory_state, chosen_reg, best_start, task["dur"], task["cores"]
        )

    save_cluster_state(in_memory_state)

    stats_traces = all_traces
    if max_end_time > dl_dt:
        stats_traces = {reg: load_carbon_data(reg, sub_dt, max_end_time) for reg in REGIONS}

    stats = calculate_standardized_stats(plan, stats_traces, sub_dt, len(tasks), deadline=dl_dt)
    stats = _apply_late_carbon_penalty(plan, stats_traces, stats, DEADLINE_PENALTY_MULTIPLIER)

    plot_optimization_window_heatmap(
        plan, stats_traces, sub_dt, max_end_time,
        plot_name=f"baseline_B_{run_id}",
        stats=stats
    )

    plan_path = os.path.join(PLAN_DIR, f"plan_{run_id}.json")
    with open(plan_path, "w") as f:
        json.dump(plan, f)

    return plan

def plan_baseline_task_temporal_only(run_id, submission_time, deadline_date, tasks_metadata, data_size_gb, in_memory_state=None, source_region=REGIONS[0]):
    """
    BASELINE 3: Task-Level Temporal-Only Shifting
    - Tasks are pinned to the source_region (0 network transfer overhead).
    - Each task independently searches for the greenest time slot between its arrival and the deadline.
    - Incorporates exact capacity checking (CORES_LIMIT).
    """
    _log(f"[BASELINE TEMPORAL] Planning with high-fidelity tracking | Workload: {run_id}")
    plan_path = get_safe_plan_path(f"base_temp_{run_id}")
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    if isinstance(submission_time, str): submission_time = dt.datetime.fromisoformat(submission_time)
    if submission_time.tzinfo is None: submission_time = submission_time.replace(tzinfo=dt.timezone.utc)
        
    if isinstance(deadline_date, str): deadline_date = dt.datetime.fromisoformat(deadline_date)
    deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)

    source_traces = load_carbon_data(source_region, submission_time, deadline_date)
    if not source_traces: return {}
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    source_base_ci = source_traces.get(sub_naive, source_traces[min(source_traces.keys(), key=lambda d: abs(d-sub_naive))])

    if in_memory_state is None:
        in_memory_state = load_cluster_state()
        
    manifest = {}
    current_ready_time = submission_time
    temp_search_state = copy.deepcopy(in_memory_state)

    for t in tasks_metadata:
        best_task_impact = float('inf')
        best_task_start = None
        best_task_ci = None
        
        window_hours = int((deadline_date - current_ready_time).total_seconds() // 3600)
        
        for offset in range(max(1, int(window_hours - t['dur'] + 1))):
            cand_start = current_ready_time + dt.timedelta(hours=offset)
            t_lookup = cand_start.replace(tzinfo=None, minute=0, second=0, microsecond=0)
            
            can_reserve = True
            for h in range(int(t['dur'])):
                ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                usage = temp_search_state.get(source_region, {}).get(ts, 0)
                if usage + t['cores'] > CORES_LIMIT:
                    can_reserve = False
                    break
                    
            if not can_reserve: continue
                
            t_ci = source_traces.get(min(source_traces.keys(), key=lambda d: abs(d - t_lookup)), 400)
            t_aware_carbon = t_ci * (t['dur'] * t['cores'])
            
            if t_aware_carbon < best_task_impact:
                best_task_impact = t_aware_carbon
                best_task_start = t_lookup
                best_task_ci = t_ci

        if not best_task_start:
            best_task_start = current_ready_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
            best_task_ci = source_traces.get(min(source_traces.keys(), key=lambda d: abs(d - best_task_start)), 400)
            
        t_baseline_carbon = source_base_ci * (t['dur'] * t['cores'])
        
        for h in range(int(t['dur'])):
            ts = (best_task_start + dt.timedelta(hours=h)).isoformat()
            if source_region not in temp_search_state: temp_search_state[source_region] = {}
            temp_search_state[source_region][ts] = temp_search_state[source_region].get(ts, 0) + t['cores']

        manifest[t['id']] = {
            "start": best_task_start.isoformat(), 
            "dur": t['dur'], 
            "region": source_region, 
            "cores": t['cores'],
            "pt_ci": round(source_base_ci, 2),
            "target_ci": round(best_task_ci, 2),
            "saved_g": round(t_baseline_carbon - best_task_impact, 2), 
            "pct": round(((t_baseline_carbon - best_task_impact) / t_baseline_carbon * 100), 1) if t_baseline_carbon > 0 else 0
        }
        
        current_ready_time = best_task_start.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=t['dur'])

    _log(f"\n[BASELINE TEMPORAL REPORT] Workflow: {run_id}")
    save_cluster_state(temp_search_state)
    with open(plan_path, 'w') as f: json.dump(manifest, f, indent=4)
    return manifest

def plan_baseline_task_spatial_only(run_id, submission_time, deadline_date, tasks_metadata, data_size_gb, in_memory_state=None, source_region=REGIONS[0]):
    """
    BASELINE 4: Task-Level Spatial-Only Shifting
    - Tasks execute as early as possible (0 temporal shifting offset).
    - Evaluates carbon intensity strictly at task/data arrival time in each region.
    - Respects DAG dependencies (`depends_on`) and cross-region data transfers.
    """
    _log(f"[BASELINE SPATIAL] Planning with high-fidelity tracking | Workload: {run_id}")
    plan_path = get_safe_plan_path(f"base_spat_{run_id}")
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    if isinstance(submission_time, str): submission_time = dt.datetime.fromisoformat(submission_time)
    if submission_time.tzinfo is None: submission_time = submission_time.replace(tzinfo=dt.timezone.utc)

    if isinstance(deadline_date, str): deadline_date = dt.datetime.fromisoformat(deadline_date)
    deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)

    source_traces = load_carbon_data(source_region, submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    source_base_ci = source_traces.get(sub_naive, source_traces[min(source_traces.keys(), key=lambda d: abs(d-sub_naive))] if source_traces else 450)

    if in_memory_state is None: in_memory_state = load_cluster_state()
        
    manifest = {}
    temp_search_state = copy.deepcopy(in_memory_state)
    
    task_end_times = {}
    task_regions = {}
    data_per_task = data_size_gb / len(tasks_metadata) if tasks_metadata else 0

    for t in tasks_metadata:
        parents = t.get('depends_on', [])
        
        best_task_impact = float('inf')
        best_task_start = None
        best_task_ci = None
        best_task_reg = None
        best_transfer_carbon = 0.0
        
        for region in REGIONS:
            max_parent_ready = submission_time
            total_transfer_carbon_g = 0.0
            
            if parents:
                for p_id in parents:
                    p_end = task_end_times.get(p_id, submission_time)
                    p_reg = task_regions.get(p_id, source_region)
                    
                    t_hours, t_carbon = calculate_transfer_penalty(region, data_per_task, p_reg)
                    
                    p_ready_in_region = p_end + dt.timedelta(hours=t_hours)
                    if p_ready_in_region > max_parent_ready:
                        max_parent_ready = p_ready_in_region
                    
                    total_transfer_carbon_g += t_carbon
            else:
                t_hours, t_carbon = calculate_transfer_penalty(region, data_per_task, source_region)
                max_parent_ready = submission_time + dt.timedelta(hours=t_hours)
                total_transfer_carbon_g = t_carbon

            cand_start = max_parent_ready
            cand_start_naive = cand_start.replace(tzinfo=None, minute=0, second=0, microsecond=0)
            
            intensities = load_carbon_data(region, cand_start, deadline_date)
            if not intensities: continue

            spatial_ci = intensities.get(
                min(intensities.keys(), key=lambda d: abs(d - cand_start_naive)), 400
            )

            can_reserve = False
            queue_offset = 0
            while not can_reserve and queue_offset < 72:
                can_reserve = True
                test_start = cand_start_naive + dt.timedelta(hours=queue_offset)
                for h in range(int(t['dur'])):
                    ts = (test_start + dt.timedelta(hours=h)).isoformat()
                    usage = temp_search_state.get(region, {}).get(ts, 0)
                    if usage + t['cores'] > CORES_LIMIT:
                        can_reserve = False
                        queue_offset += 1
                        break
            
            if not can_reserve: continue
                
            actual_start = cand_start_naive + dt.timedelta(hours=queue_offset)
            
            t_aware_carbon = (spatial_ci * (t['dur'] * t['cores'])) + total_transfer_carbon_g
            
            if t_aware_carbon < best_task_impact:
                best_task_impact = t_aware_carbon
                best_task_start = actual_start
                best_task_ci = spatial_ci
                best_task_reg = region
                best_transfer_carbon = total_transfer_carbon_g

        if not best_task_reg: return {}

        t_baseline_carbon = source_base_ci * (t['dur'] * t['cores'])
        
        for h in range(int(t['dur'])):
            ts = (best_task_start + dt.timedelta(hours=h)).isoformat()
            if best_task_reg not in temp_search_state: temp_search_state[best_task_reg] = {}
            temp_search_state[best_task_reg][ts] = temp_search_state[best_task_reg].get(ts, 0) + t['cores']

        manifest[t['id']] = {
            "start": best_task_start.isoformat(), 
            "dur": t['dur'], 
            "region": best_task_reg, 
            "cores": t['cores'],
            "pt_ci": round(source_base_ci, 2),
            "target_ci": round(best_task_ci, 2),
            "transfer_carbon_g": round(best_transfer_carbon, 2), 
            "saved_g": round(t_baseline_carbon - best_task_impact, 2), 
            "pct": round(((t_baseline_carbon - best_task_impact) / t_baseline_carbon * 100), 1) if t_baseline_carbon > 0 else 0
        }
        
        task_regions[t['id']] = best_task_reg
        task_end_times[t['id']] = best_task_start.replace(tzinfo=dt.timezone.utc) + dt.timedelta(hours=t['dur'])

    _log(f"\n[BASELINE SPATIAL REPORT] Workflow: {run_id}")

    sub_naive = submission_time.replace(tzinfo=None)
    stats_traces = {reg: load_carbon_data(reg, submission_time, deadline_date) for reg in REGIONS}
    stats = calculate_standardized_stats(manifest, stats_traces, sub_naive, len(tasks_metadata), deadline=deadline_date)

    plot_optimization_window_heatmap(
        best_manifest=manifest,
        all_regions_data=stats_traces,
        submission_time=submission_time,
        deadline=deadline_date,
        plot_name=f"base_spat_{run_id}",
        stats=stats
    )

    save_cluster_state(temp_search_state)
    with open(plan_path, 'w') as f: json.dump(manifest, f, indent=4)
    return manifest


def plan_atomic(run_id, submission, deadline, tasks, edges, in_memory_state, source_region=None):
    if source_region is None:
        _log(f"[WARNING] No source region provided for {run_id}. Defaulting to {REGIONS[0]}.", logging.WARNING)
        source_region = REGIONS[0]

    _log(f"[BASELINE ATOMIC] Planning continuous block for {run_id} from {source_region}...")
    plan = {}

    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(str(submission))
    dl_dt  = deadline   if isinstance(deadline,  dt.datetime) else dt.datetime.fromisoformat(str(deadline))
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt  = dl_dt.replace(tzinfo=None)

    total_duration   = sum(t['dur'] for t in tasks)
    max_cores_needed = max(t['cores'] for t in tasks)
    task_map         = {t['id']: t for t in tasks}

    DEADLINE_PENALTY_MULTIPLIER = 1.5

    best_start            = None
    best_region           = None
    lowest_average_carbon = float('inf')
    best_is_late          = False

    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in REGIONS}

    for reg in REGIONS:
        time_penalty, _ = calculate_transfer_penalty(reg, DATA_SIZE_GB, source_region=source_region)
        adjusted_sub_time = sub_dt + dt.timedelta(hours=time_penalty)

        if adjusted_sub_time >= dl_dt:
            continue

        earliest_possible_start = find_earliest_slot(
            reg, max_cores_needed, in_memory_state,
            not_before=adjusted_sub_time, deadline=dl_dt
        )
        if earliest_possible_start is None:
            continue

        available_hours = int((dl_dt - earliest_possible_start).total_seconds() / 3600)

        for offset in range(max(0, int(available_hours - total_duration + 1))):
            candidate_start = earliest_possible_start + dt.timedelta(hours=offset)

            if candidate_start + dt.timedelta(hours=total_duration) > dl_dt:
                break

            block_is_valid       = True
            current_block_carbon = 0.0
            current_task_start   = candidate_start

            for task in tasks:
                task_transfer_carbon = 0.0
                for p_id in task.get('depends_on', []):
                    parent_task = task_map.get(p_id)
                    if parent_task:
                        data_size = parent_task.get('output_size_gb', DATA_SIZE_GB)
                        _, t_carbon = calculate_transfer_penalty(
                            target_region=reg, source_region=source_region, data_size_gb=data_size
                        )
                        task_transfer_carbon += t_carbon
                current_block_carbon += task_transfer_carbon

                for h in range(int(task['dur'])):
                    hour_obj = current_task_start + dt.timedelta(hours=h)
                    hour_str = hour_obj.isoformat()
                    cores_used_already = in_memory_state.get(reg, {}).get(hour_str, 0)
                    if cores_used_already + task['cores'] > CORES_LIMIT:
                        block_is_valid = False
                        break
                    intensity = all_traces.get(reg, {}).get(hour_obj, 400)
                    current_block_carbon += (task['cores'] * 0.2) * intensity

                if not block_is_valid:
                    break
                current_task_start += dt.timedelta(hours=task['dur'])

            if block_is_valid and current_block_carbon < lowest_average_carbon:
                lowest_average_carbon = current_block_carbon
                best_start   = candidate_start
                best_region  = reg
                best_is_late = False

    if best_start is None:
        _log(f"[ATOMIC] No valid block within deadline for {run_id}. "
              f"Staying in {source_region} and scheduling late (no region change — "
              f"transfer time would make it even later).")

        earliest_possible_start = find_earliest_slot(
            source_region, max_cores_needed, in_memory_state,
            not_before=sub_dt, deadline=None
        )

        if earliest_possible_start is not None:
            candidate_start = earliest_possible_start
            late_window_end = candidate_start + dt.timedelta(hours=total_duration + 1)
            late_traces = load_carbon_data(source_region, candidate_start, late_window_end)

            block_is_valid = True
            current_block_carbon = 0.0
            current_task_start = candidate_start

            for task in tasks:
                for h in range(int(task['dur'])):
                    hour_obj = current_task_start + dt.timedelta(hours=h)
                    hour_str = hour_obj.isoformat()
                    cores_used_already = in_memory_state.get(source_region, {}).get(hour_str, 0)
                    if cores_used_already + task['cores'] > CORES_LIMIT:
                        block_is_valid = False
                        break
                    intensity = late_traces.get(hour_obj, 400)
                    current_block_carbon += (task['cores'] * 0.2) * intensity * DEADLINE_PENALTY_MULTIPLIER
                if not block_is_valid:
                    break
                current_task_start += dt.timedelta(hours=task['dur'])

            if block_is_valid:
                best_start  = candidate_start
                best_region = source_region
                best_is_late = True

        if best_start is None:
            _log(f"[ATOMIC] Last resort: forcing {run_id} to submission time in {source_region}.")
            best_region  = source_region
            best_start   = sub_dt
            best_is_late = True

    current_task_start = best_start
    for task in tasks:
        task_transfer_carbon = 0.0
        for p_id in task.get('depends_on', []):
            parent_task = task_map.get(p_id)
            if parent_task:
                data_size = parent_task.get('output_size_gb', DATA_SIZE_GB)
                _, t_carbon = calculate_transfer_penalty(
                    target_region=best_region, source_region=source_region, data_size_gb=data_size
                )
                task_transfer_carbon += t_carbon

        plan[task['id']] = {
            "region":            best_region,
            "start":             current_task_start.isoformat(),
            "dur":               task['dur'],
            "cores":             task['cores'],
            "transfer_carbon_g": round(task_transfer_carbon, 2),
            "scheduled_late":    best_is_late,
        }
        in_memory_state = update_scratchpad(
            in_memory_state, best_region, current_task_start, task['dur'], task['cores']
        )
        current_task_start += dt.timedelta(hours=task['dur'])

    save_cluster_state(in_memory_state)

    stats_window_end = max(dl_dt, current_task_start)
    stats_traces = {reg: load_carbon_data(reg, sub_dt, stats_window_end) for reg in REGIONS}
    stats = calculate_standardized_stats(plan, stats_traces, sub_dt, len(tasks), deadline=dl_dt)

    plot_optimization_window_heatmap(
        plan, stats_traces, sub_dt, stats_window_end,
        plot_name=f"carbon_atomic_{run_id}", stats=stats
    )

    plan_path = get_safe_plan_path(run_id)
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=4)

    return plan

def plan_atomic_oracle(dag_runs_metadata, source_region=REGIONS[0], batch_label=None):
    """
    Atomic ORACLE SCHEDULER (Strict Carbon Minimization)
    """
    initial_cluster_state = load_cluster_state()

    all_subs, all_dls, total_requested_tasks = [], [], 0
    for wf in dag_runs_metadata:
        s, d = wf['submission'], wf['deadline']
        dl_dt = dt.datetime.fromisoformat(d) if isinstance(d, str) else d
        all_dls.append(dl_dt.replace(tzinfo=None))
        sub_dt = dt.datetime.fromisoformat(s) if isinstance(s, str) else s
        all_subs.append(sub_dt.replace(tzinfo=None))
        total_requested_tasks += len(wf['tasks'])

    batch_start_bound = min(all_subs)
    batch_end_bound = max(all_dls)

    total_perms = math.factorial(len(dag_runs_metadata))
    _log(f"\n[Atomic ORACLE] Batch Window: {batch_start_bound} to {batch_end_bound} | "
          f"Total Requested Tasks: {total_requested_tasks}")
    _log(f"[Atomic ORACLE] Testing all {total_perms} possible orderings for lowest carbon...")

    best_carbon      = float('inf')
    best_results     = {}
    best_manifest    = {}
    best_order       = []
    best_stats       = {}
    best_scratchpad  = None

    for queue in itertools.permutations(dag_runs_metadata):

        scratchpad     = copy.deepcopy(initial_cluster_state)
        batch_results  = {}
        total_manifest = {}

        for workflow in queue:
            wf_submission = workflow['submission']
            wf_submission = (dt.datetime.fromisoformat(wf_submission)
                              if isinstance(wf_submission, str) else wf_submission)

            plan = plan_atomic(
                workflow['run_id'],
                wf_submission,
                workflow['deadline'],
                workflow['tasks'],
                workflow['edges'],
                in_memory_state=scratchpad,
                source_region=source_region
            )

            if not plan:
                batch_results[workflow['run_id']] = {}
                continue

            batch_results[workflow['run_id']] = plan

            for tid, tdata in plan.items():
                unique_key = f"{workflow['run_id']}_{tid}"
                total_manifest[unique_key] = tdata

                start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
                reg = tdata['region']
                scratchpad = update_scratchpad(scratchpad, reg, start_dt, tdata['dur'], tdata['cores'])

        if not total_manifest:
            continue

        all_traces = {reg: load_carbon_data(reg, batch_start_bound, batch_end_bound) for reg in REGIONS}
        start_naive = batch_start_bound.replace(tzinfo=None)
        stats = calculate_standardized_stats(total_manifest, all_traces, start_naive, total_requested_tasks)

        ordering_carbon = stats.get(
            'total_carbon_with_transfer',
            stats.get('total_carbon', float('inf')),
        )
        if ordering_carbon < best_carbon:
            best_carbon     = ordering_carbon
            best_results    = batch_results
            best_manifest   = total_manifest
            best_order      = [wf['run_id'] for wf in queue]
            best_stats      = stats
            best_scratchpad = scratchpad

    if not best_results:
        _log("[Atomic ORACLE] WARNING: No valid ordering found across all permutations.", logging.WARNING)
        return {}

    _log(f"[Atomic ORACLE] Evaluation complete. Winner ({total_perms} orderings tested): "
          f"{round(best_carbon, 2)}gCO2eq (execution + transfer)")

    save_cluster_state(best_scratchpad)

    batch_id = f"atomic_oracle_{dt.datetime.now().strftime('%H%M%S')}"
    plot_optimization_window_heatmap(
        best_manifest,
        {reg: load_carbon_data(reg, batch_start_bound, batch_end_bound) for reg in REGIONS},
        batch_start_bound,
        batch_end_bound,
        batch_id,
        stats=best_stats
    )

    save_chosen_path(
        batch_label=batch_label or "atomic_oracle_unlabeled",
        algorithm="Atomic_Oracle",
        order_run_ids=best_order,
        task_regions={
            f"{run_id}::{tid}": tdata['region']
            for run_id, plan in best_results.items()
            for tid, tdata in (plan or {}).items()
        },
        exec_carbon_g=best_stats.get('total_carbon', 0.0),
        transfer_carbon_g=best_stats.get('total_transfer_carbon', 0.0),
        extra={"permutations_tested": total_perms},
    )

    return best_results

def preprocess_carbon_to_rle_buckets(all_regions_intensities, REGIONS, max_window_hours):
    all_active_values = []
    for r in REGIONS:
        intensities = all_regions_intensities.get(r, {})
        if not intensities:
            continue
        timeline = [intensities[ts] for ts in sorted(intensities.keys())]
        active = timeline[:max_window_hours] if len(timeline) >= max_window_hours else timeline
        all_active_values.extend(active)

    if not all_active_values:
        return {r: {"DARK_GREEN": [], "LIGHT_GREEN": [], "YELLOW": [], "DIRTY_RED": []} for r in REGIONS}

    p25 = np.percentile(all_active_values, 20)
    p50 = np.percentile(all_active_values, 40)
    p75 = np.percentile(all_active_values, 75)

    def classify(ci_val):
        if ci_val <= p25:
            return "DARK_GREEN"
        elif ci_val <= p50:
            return "LIGHT_GREEN"
        elif ci_val <= p75:
            return "YELLOW"
        else:
            return "DIRTY_RED"

    rle_index = {r: {"DARK_GREEN": [], "LIGHT_GREEN": [], "YELLOW": [], "DIRTY_RED": []} for r in REGIONS}

    for r in REGIONS:
        intensities = all_regions_intensities.get(r, {})
        if not intensities:
            continue

        timeline = [intensities[ts] for ts in sorted(intensities.keys())]
        active_values = timeline[:max_window_hours] if len(timeline) >= max_window_hours else timeline
        if not active_values:
            continue

        current_bucket = None
        current_start = 0
        current_length = 0

        for hour_offset in range(max_window_hours):
            ci_val = timeline[hour_offset] if hour_offset < len(timeline) else timeline[-1]
            bucket_type = classify(ci_val)

            if current_bucket is None:
                current_bucket = bucket_type
                current_start = hour_offset
                current_length = 1
            elif bucket_type == current_bucket:
                current_length += 1
            else:
                rle_index[r][current_bucket].append({"start": current_start, "length": current_length})
                current_bucket = bucket_type
                current_start = hour_offset
                current_length = 1

        if current_bucket is not None:
            rle_index[r][current_bucket].append({"start": current_start, "length": current_length})

        for bucket, intervals in rle_index[r].items():
            total_hours = sum(i["length"] for i in intervals)

    return rle_index

def get_incoming_transfer_footprint(task, task_map, task_scheduled_regions, current_eval_region):
    """
    Computes total cross-region data size (GB) to fetch from parent cloud storage nodes.
    """
    parents = task.get('depends_on', [])
    if not parents:
        return 0

    total_cross_region_gb = 0
    for p_id in parents:
        parent_region = task_scheduled_regions.get(p_id)
        if parent_region and parent_region != current_eval_region:
            parent_task_static = task_map.get(p_id, {})
            total_cross_region_gb += parent_task_static.get('output_size_gb', 0)
                
    return total_cross_region_gb

def _schedule_late_with_parallel_transfers(tasks_metadata, source_region, submission_time,
                                           max_window_hours, base_ci, in_memory_state=None):
    """
    Late scheduling fallback for when the workflow critical path exceeds the deadline.

    Strategy:
      1. Schedule ALL tasks in source_region in topological order, past the
         deadline if necessary (atomic-style baseline, no region hopping yet).
      2. Group tasks whose execution windows genuinely overlap in this baseline
         (parallel siblings — not an ancestor/descendant of one another).
      3. Within each parallel group, a task is only considered for transfer out
         of source_region if:
           (a) source_region is not the greenest available option for it, OR
           (b) source_region cannot hold every task in the group at once
               (combined cores demand of the group exceeds CORES_LIMIT).
         When a group qualifies, the SMALLEST tasks (lowest cores*dur footprint,
         tie-broken by transfer payload size) are offloaded first, since they are
         the cheapest/fastest to move and least likely to be pushed further past
         the deadline by transfer latency.
    """
    DEADLINE_PENALTY_MULTIPLIER = 1.5

    task_map = {t['id']: t for t in tasks_metadata}
    naive_submission = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)

    if in_memory_state is None:
        current_state = load_cluster_state()
    else:
        current_state = copy.deepcopy(in_memory_state)

    for r in REGIONS:
        if r not in current_state:
            current_state[r] = {}

    def topo_sort():
        in_deg = {t['id']: 0 for t in tasks_metadata}
        graph  = {t['id']: [] for t in tasks_metadata}
        for t in tasks_metadata:
            for p in t.get('depends_on', []):
                graph[p].append(t['id'])
                in_deg[t['id']] += 1
        queue = [t['id'] for t in tasks_metadata if in_deg[t['id']] == 0]
        order = []
        while queue:
            tid = queue.pop(0)
            order.append(tid)
            for child in graph[tid]:
                in_deg[child] -= 1
                if in_deg[child] == 0:
                    queue.append(child)
        return [task_map[tid] for tid in order]

    ordered_tasks = topo_sort()

    final_manifest    = {}
    task_end_times     = {}
    task_start_times   = {}
    task_regions        = {}
    task_transfer_carbon = {t['id']: 0.0 for t in ordered_tasks}

    def get_ci(region, hour_idx):
        all_intensities = load_carbon_data(
            region,
            naive_submission,
            naive_submission + dt.timedelta(hours=hour_idx + 48)
        )
        target_dt = naive_submission + dt.timedelta(hours=hour_idx)
        target_dt = target_dt.replace(minute=0, second=0, microsecond=0)
        if target_dt in all_intensities:
            return all_intensities[target_dt]
        if all_intensities:
            closest = min(all_intensities.keys(), key=lambda d: abs((d - target_dt).total_seconds()))
            return all_intensities[closest]
        return 400

    def region_avg_ci(region, start_hour, dur):
        return sum(get_ci(region, start_hour + h) for h in range(dur)) / dur

    def find_slot(region, task, earliest_hour):
        """Find earliest available slot in region from earliest_hour onwards."""
        for h in range(earliest_hour, earliest_hour + 240):
            cap_ok = all(
                current_state[region].get(
                    (naive_submission + dt.timedelta(hours=h + fh)).replace(
                        minute=0, second=0, microsecond=0).isoformat(), 0
                ) + task['cores'] <= CORES_LIMIT
                for fh in range(task['dur'])
            )
            if cap_ok:
                return h
        return earliest_hour

    def book_slot(region, start_hour, task):
        for h in range(task['dur']):
            ts = (naive_submission + dt.timedelta(hours=start_hour + h)).replace(
                minute=0, second=0, microsecond=0).isoformat()
            current_state[region][ts] = current_state[region].get(ts, 0) + task['cores']

    def unbook_slot(region, start_hour, task):
        for h in range(task['dur']):
            ts = (naive_submission + dt.timedelta(hours=start_hour + h)).replace(
                minute=0, second=0, microsecond=0).isoformat()
            current_state[region][ts] = max(0, current_state[region].get(ts, 0) - task['cores'])

    def ancestors_of(task):
        anc = set()
        queue = list(task.get('depends_on', []))
        while queue:
            p = queue.pop()
            anc.add(p)
            queue.extend(task_map.get(p, {}).get('depends_on', []))
        return anc

    def descendants_of(task_id):
        return {t['id'] for t in tasks_metadata if task_id in t.get('depends_on', [])}

    for task in ordered_tasks:
        parents = task.get('depends_on', [])
        safe_start = max((task_end_times.get(p_id, 0) for p_id in parents), default=0)
        slot = find_slot(source_region, task, safe_start)
        book_slot(source_region, slot, task)
        task_start_times[task['id']] = slot
        task_end_times[task['id']]   = slot + task['dur']
        task_regions[task['id']]     = source_region

    task_ids   = [t['id'] for t in ordered_tasks]
    anc_cache  = {t['id']: ancestors_of(t) for t in ordered_tasks}
    desc_cache = {t['id']: descendants_of(t['id']) for t in ordered_tasks}

    def overlaps(a_id, b_id):
        if b_id in anc_cache[a_id] or b_id in desc_cache[a_id]:
            return False
        a_s, a_e = task_start_times[a_id], task_end_times[a_id]
        b_s, b_e = task_start_times[b_id], task_end_times[b_id]
        return a_s < b_e and a_e > b_s

    adjacency = {tid: set() for tid in task_ids}
    for i in range(len(task_ids)):
        for j in range(i + 1, len(task_ids)):
            a_id, b_id = task_ids[i], task_ids[j]
            if overlaps(a_id, b_id):
                adjacency[a_id].add(b_id)
                adjacency[b_id].add(a_id)

    visited, groups = set(), []
    for tid in task_ids:
        if tid in visited or not adjacency[tid]:
            continue
        stack, comp = [tid], set()
        while stack:
            cur = stack.pop()
            if cur in comp:
                continue
            comp.add(cur)
            visited.add(cur)
            stack.extend(adjacency[cur] - comp)
        groups.append(comp)

    for group in groups:
        group_tasks = sorted(
            (task_map[tid] for tid in group),
            key=lambda t: (t['cores'] * t['dur'],
                            task_map[t['id']].get('output_size_gb', DATA_SIZE_GB))
        )
        group_cores_demand = sum(t['cores'] for t in group_tasks)
        capacity_exceeded  = group_cores_demand > CORES_LIMIT

        for idx, task in enumerate(group_tasks):
            tid = task['id']
            if task_regions[tid] != source_region:
                continue

            if idx > 0 and not capacity_exceeded:
                break

            slot = task_start_times[tid]
            is_late = slot >= max_window_hours
            source_ci = region_avg_ci(source_region, slot, task['dur'])
            source_exec_carbon = source_ci * task['cores'] * task['dur'] * 0.2
            source_total = source_exec_carbon * (DEADLINE_PENALTY_MULTIPLIER if is_late else 1.0)

            parents = task.get('depends_on', [])
            safe_start = max((task_end_times.get(p_id, 0) for p_id in parents), default=0)

            best_candidate = None

            for candidate_region in REGIONS:
                if candidate_region == source_region:
                    continue

                data_size = sum(
                    task_map[p_id].get('output_size_gb', DATA_SIZE_GB)
                    for p_id in parents
                    if p_id in task_map
                ) or DATA_SIZE_GB

                t_hours, t_carbon = calculate_transfer_penalty(
                    target_region=candidate_region,
                    source_region=source_region,
                    data_size_gb=data_size
                )

                candidate_earliest = max(safe_start + math.ceil(t_hours), safe_start)
                candidate_slot = find_slot(candidate_region, task, candidate_earliest)

                candidate_ci = region_avg_ci(candidate_region, candidate_slot, task['dur'])
                candidate_exec_carbon = candidate_ci * task['cores'] * task['dur'] * 0.2
                mult = DEADLINE_PENALTY_MULTIPLIER if candidate_slot >= max_window_hours else 1.0
                candidate_total = (candidate_exec_carbon * mult) + t_carbon

                carbon_saved = source_total - candidate_total
                is_greener = carbon_saved > 0

                if not is_greener and not capacity_exceeded:
                    continue

                sort_key = (0 if is_greener else 1, candidate_total)
                if best_candidate is None or sort_key < best_candidate[0]:
                    best_candidate = (sort_key, candidate_region, candidate_slot, t_carbon, carbon_saved)

            if best_candidate is not None:
                _, chosen_region, chosen_slot, transfer_carbon, carbon_saved = best_candidate

                unbook_slot(source_region, slot, task)
                book_slot(chosen_region, chosen_slot, task)

                task_start_times[tid]     = chosen_slot
                task_end_times[tid]       = chosen_slot + task['dur']
                task_regions[tid]         = chosen_region
                task_transfer_carbon[tid] = transfer_carbon

                group_cores_demand -= task['cores']
                capacity_exceeded = group_cores_demand > CORES_LIMIT

    for task in ordered_tasks:
        tid = task['id']
        chosen_region   = task_regions[tid]
        chosen_slot     = task_start_times[tid]
        transfer_carbon = task_transfer_carbon[tid]

        t_lookup = (naive_submission + dt.timedelta(hours=chosen_slot)).replace(
            minute=0, second=0, microsecond=0)

        task_ci     = region_avg_ci(chosen_region, chosen_slot, task['dur'])
        is_late     = chosen_slot >= max_window_hours
        mult        = DEADLINE_PENALTY_MULTIPLIER if is_late else 1.0
        exec_carbon = task_ci * task['cores'] * task['dur'] * 0.2 * mult
        pt_baseline = base_ci * task['dur'] * task['cores']

        final_manifest[tid] = {
            "start":             t_lookup.isoformat(),
            "dur":               task['dur'],
            "region":            chosen_region,
            "cores":             task['cores'],
            "pt_ci":             round(base_ci, 2),
            "target_ci":         round(task_ci, 2),
            "transfer_carbon_g": round(transfer_carbon, 2),
            "scheduled_late":    is_late,
            "saved_g":           round(pt_baseline - exec_carbon - transfer_carbon, 2),
            "pct":               round(((pt_baseline - exec_carbon - transfer_carbon) / pt_baseline * 100), 1)
                                 if pt_baseline > 0 else 0
        }

    save_cluster_state(current_state)
    return final_manifest


def find_greenest_schedule_via_rle(REGIONS, tasks_metadata, all_regions_intensities, rle_index,
                                    submission_time, max_window_hours, base_ci, in_memory_state=None, source_region=None):
    task_map = {t['id']: t for t in tasks_metadata}

    def get_task_priority(task_id, memo=None):
        if memo is None: memo = {}
        if task_id in memo: return memo[task_id]
        t = task_map[task_id]
        dependent_tasks = [other for other in tasks_metadata if task_id in other.get('depends_on', [])]
        if not dependent_tasks:
            memo[task_id] = t['dur']
            return t['dur']
        memo[task_id] = t['dur'] + max(get_task_priority(dep['id'], memo) for dep in dependent_tasks)
        return memo[task_id]

    sorted_tasks = sorted(tasks_metadata, key=lambda x: get_task_priority(x['id']), reverse=True)

    if in_memory_state is not None:
        current_state = copy.deepcopy(in_memory_state)
    else:
        current_state = load_cluster_state()
    for r in REGIONS:
        if r not in current_state: current_state[r] = {}

    final_manifest = {}
    task_end_times = {}
    task_scheduled_regions = {}
    naive_submission = submission_time.replace(tzinfo=None)

    def get_carbon_at_hour(reg, hour_idx):
        intensities = all_regions_intensities.get(reg, {})
        sorted_ts = sorted(intensities.keys())
        if hour_idx < len(sorted_ts):
            return intensities[sorted_ts[hour_idx]]
        return intensities[sorted_ts[-1]] if sorted_ts else 400

    def build_ranked_pool(bucket_types):
        pool = []
        for region in REGIONS:
            for b_type in bucket_types:
                if b_type in rle_index.get(region, {}):
                    for interval in rle_index[region][b_type]:
                        total_ci = sum(get_carbon_at_hour(region, interval['start'] + h) for h in range(interval['length']))
                        avg_ci = total_ci / interval['length'] if interval['length'] > 0 else 400
                        pool.append({
                            "region": region,
                            "start": interval['start'],
                            "length": interval['length'],
                            "avg_ci": avg_ci
                        })
        pool.sort(key=lambda x: x['avg_ci'])
        return pool

    all_bucket_pool = build_ranked_pool(["DARK_GREEN", "LIGHT_GREEN", "YELLOW", "DIRTY_RED"])

    for idx, pt in enumerate(sorted_tasks):
        parents = pt.get('depends_on', [])
        latest_allowed_start = max_window_hours - pt['dur']

        best_slot = None
        best_region = None
        best_transfer_carbon = 0
        best_avg_ci = float('inf')

        def evaluate_pool(bucket_pool):
            nonlocal best_slot, best_region, best_transfer_carbon, best_avg_ci

            for bucket in bucket_pool:
                if bucket['avg_ci'] >= best_avg_ci:
                    break

                region = bucket['region']
                earliest_start = 0
                current_node_transfer_carbon = 0

                if parents:
                    parent_times = []
                    task_data_size = get_incoming_transfer_footprint(pt, task_map, task_scheduled_regions, region)
                    for p_id in parents:
                        p_end_hour = task_end_times.get(p_id, 0)
                        p_region = task_scheduled_regions.get(p_id, region)
                        t_hours, t_carbon = calculate_transfer_penalty(
                            target_region=region,
                            source_region=p_region,
                            data_size_gb=task_data_size
                        )
                        parent_times.append(p_end_hour + math.ceil(t_hours))
                        current_node_transfer_carbon += t_carbon
                    earliest_start = max(parent_times) if parent_times else 0

                search_start = max(bucket['start'], earliest_start)
                bucket_end = bucket['start'] + bucket['length']
                search_end = min(bucket_end, latest_allowed_start)

                if search_start > search_end:
                    continue

                for start_hour in range(int(search_start), int(search_end) + 1):
                    capacity_violated = False
                    for h in range(int(pt['dur'])):
                        t_slot = (naive_submission + dt.timedelta(hours=start_hour + h)).replace(
                            minute=0, second=0, microsecond=0).isoformat()
                        if current_state[region].get(t_slot, 0) + pt['cores'] > CORES_LIMIT:
                            capacity_violated = True
                            break
                    if capacity_violated:
                        continue

                    trial_state = copy.deepcopy(current_state)
                    trial_end_times = copy.deepcopy(task_end_times)
                    trial_regions = copy.deepcopy(task_scheduled_regions)

                    trial_end_times[pt['id']] = start_hour + pt['dur']
                    trial_regions[pt['id']] = region
                    for h in range(int(pt['dur'])):
                        t_slot = (naive_submission + dt.timedelta(hours=start_hour + h)).replace(
                            minute=0, second=0, microsecond=0).isoformat()
                        trial_state[region][t_slot] = trial_state[region].get(t_slot, 0) + pt['cores']

                    lookahead_success = True
                    for future_pt in sorted_tasks[idx + 1:]:
                        f_parents = future_pt.get('depends_on', [])
                        if not f_parents:
                            continue

                        child_fit_found = False
                        for r_check in REGIONS:
                            f_parent_times = []
                            f_data_size = get_incoming_transfer_footprint(
                                future_pt, task_map, trial_regions, r_check)
                            for p_id in f_parents:
                                if p_id in trial_end_times:
                                    p_end = trial_end_times[p_id]
                                    p_reg = trial_regions.get(p_id, r_check)
                                    f_t_hours, _ = calculate_transfer_penalty(
                                        target_region=r_check,
                                        source_region=p_reg,
                                        data_size_gb=f_data_size
                                    )
                                    f_parent_times.append(p_end + math.ceil(f_t_hours))

                            f_earliest = max(f_parent_times) if f_parent_times else 0
                            f_latest = max_window_hours - future_pt['dur']

                            for f_slot in range(int(f_earliest), int(f_latest) + 1):
                                f_cap_violated = False
                                for fh in range(int(future_pt['dur'])):
                                    ft_slot = (naive_submission + dt.timedelta(hours=f_slot + fh)).replace(
                                        minute=0, second=0, microsecond=0).isoformat()
                                    if trial_state[r_check].get(ft_slot, 0) + future_pt['cores'] > CORES_LIMIT:
                                        f_cap_violated = True
                                        break
                                if not f_cap_violated:
                                    child_fit_found = True
                                    trial_end_times[future_pt['id']] = f_slot + future_pt['dur']
                                    trial_regions[future_pt['id']] = r_check
                                    for fh in range(int(future_pt['dur'])):
                                        ft_slot = (naive_submission + dt.timedelta(hours=f_slot + fh)).replace(
                                            minute=0, second=0, microsecond=0).isoformat()
                                        trial_state[r_check][ft_slot] = trial_state[r_check].get(ft_slot, 0) + future_pt['cores']
                                    break
                            if child_fit_found:
                                break

                        if not child_fit_found:
                            lookahead_success = False
                            break

                    if lookahead_success:
                        slot_ci = sum(get_carbon_at_hour(region, start_hour + h) for h in range(int(pt['dur']))) / pt['dur']
                        total_carbon = slot_ci * pt['cores'] * pt['dur'] + current_node_transfer_carbon
                        if total_carbon < best_avg_ci * pt['cores'] * pt['dur']:
                            best_slot = start_hour
                            best_region = region
                            best_transfer_carbon = current_node_transfer_carbon
                            best_avg_ci = slot_ci

        evaluate_pool(all_bucket_pool)

        if best_region is not None and best_slot is not None:
            t_lookup = (naive_submission + dt.timedelta(hours=best_slot)).replace(
                minute=0, second=0, microsecond=0)

            task_ci_sum = sum(get_carbon_at_hour(best_region, best_slot + h) for h in range(int(pt['dur'])))
            pt_ci = task_ci_sum / pt['dur']
            pt_aware_carbon = (pt_ci * (pt['dur'] * pt['cores'])) + best_transfer_carbon
            pt_baseline_carbon = base_ci * (pt['dur'] * pt['cores'])

            final_manifest[pt['id']] = {
                "start": t_lookup.isoformat(),
                "dur": pt['dur'],
                "region": best_region,
                "cores": pt['cores'],
                "pt_ci": round(base_ci, 2),
                "target_ci": round(pt_ci, 2),
                "transfer_carbon_g": round(best_transfer_carbon, 2),
                "saved_g": round(pt_baseline_carbon - pt_aware_carbon, 2),
                "pct": round(((pt_baseline_carbon - pt_aware_carbon) / pt_baseline_carbon * 100), 1) if pt_baseline_carbon > 0 else 0
            }

            for h in range(int(pt['dur'])):
                ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                current_state[best_region][ts] = current_state[best_region].get(ts, 0) + pt['cores']

            task_end_times[pt['id']] = best_slot + pt['dur']
            task_scheduled_regions[pt['id']] = best_region

        else:
            safe_start = max(
                (task_end_times.get(p_id, 0) for p_id in parents), default=0
            )

            source_slot = None
            extended_window = safe_start + 240

            for h in range(int(safe_start), int(extended_window)):
                cap_ok = all(
                    current_state[source_region].get(
                        (naive_submission + dt.timedelta(hours=h + fh)).replace(
                            minute=0, second=0, microsecond=0).isoformat(), 0
                    ) + pt['cores'] <= CORES_LIMIT
                    for fh in range(int(pt['dur']))
                )
                if cap_ok:
                    source_slot = h
                    break

            if source_slot is None:
                source_slot = safe_start

            chosen_slot   = source_slot
            chosen_region = source_region
            transfer_carbon = 0.0

            is_parallel = any(
                task_end_times.get(other_id, 0) > source_slot and
                source_slot + pt['dur'] > (task_end_times.get(other_id, 0) - task_map[other_id]['dur']
                                            if other_id in task_map else 0)
                for other_id in task_scheduled_regions
                if other_id not in parents and pt['id'] not in [
                    t['id'] for t in tasks_metadata
                    if other_id in t.get('depends_on', [])
                ]
            )

            if is_parallel:
                source_ci_at_slot = sum(
                    get_carbon_at_hour(source_region, source_slot + h)
                    for h in range(int(pt['dur']))
                ) / pt['dur']

                source_carbon = source_ci_at_slot * pt['cores'] * pt['dur'] * 0.2

                for candidate_region in REGIONS:
                    if candidate_region == source_region:
                        continue

                    task_data_size = get_incoming_transfer_footprint(
                        pt, task_map, task_scheduled_regions, candidate_region
                    )
                    t_hours, t_carbon = calculate_transfer_penalty(
                        target_region=candidate_region,
                        source_region=source_region,
                        data_size_gb=task_data_size if task_data_size > 0 else DATA_SIZE_GB
                    )

                    candidate_slot = max(source_slot, safe_start + math.ceil(t_hours))

                    candidate_actual_slot = None
                    for h in range(int(candidate_slot), int(extended_window)):
                        cap_ok = all(
                            current_state[candidate_region].get(
                                (naive_submission + dt.timedelta(hours=h + fh)).replace(
                                    minute=0, second=0, microsecond=0).isoformat(), 0
                            ) + pt['cores'] <= CORES_LIMIT
                            for fh in range(int(pt['dur']))
                        )
                        if cap_ok:
                            candidate_actual_slot = h
                            break

                    if candidate_actual_slot is None:
                        continue

                    candidate_ci = sum(
                        get_carbon_at_hour(candidate_region, candidate_actual_slot + h)
                        for h in range(int(pt['dur']))
                    ) / pt['dur']
                    candidate_carbon = candidate_ci * pt['cores'] * pt['dur'] * 0.2

                    carbon_saved = source_carbon - (candidate_carbon + t_carbon)

                    if carbon_saved > 0:
                        chosen_slot   = candidate_actual_slot
                        chosen_region = candidate_region
                        transfer_carbon = t_carbon
                        break

            is_late = chosen_slot >= max_window_hours

            t_lookup = (naive_submission + dt.timedelta(hours=chosen_slot)).replace(
                minute=0, second=0, microsecond=0)

            task_ci_sum = sum(
                get_carbon_at_hour(chosen_region, chosen_slot + h)
                for h in range(int(pt['dur']))
            )
            pt_ci = task_ci_sum / pt['dur']
            pt_aware_carbon = (pt_ci * pt['dur'] * pt['cores'] * 0.2) + transfer_carbon
            pt_baseline_carbon = base_ci * pt['dur'] * pt['cores']

            final_manifest[pt['id']] = {
                "start":             t_lookup.isoformat(),
                "dur":               pt['dur'],
                "region":            chosen_region,
                "cores":             pt['cores'],
                "pt_ci":             round(base_ci, 2),
                "target_ci":         round(pt_ci, 2),
                "transfer_carbon_g": round(transfer_carbon, 2),
                "scheduled_late":    is_late,
                "saved_g":           round(pt_baseline_carbon - pt_aware_carbon, 2),
                "pct":               round(((pt_baseline_carbon - pt_aware_carbon) / pt_baseline_carbon * 100), 1)
                                        if pt_baseline_carbon > 0 else 0
            }

            for h in range(int(pt['dur'])):
                ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                current_state[chosen_region][ts] = current_state[chosen_region].get(ts, 0) + pt['cores']

            task_end_times[pt['id']] = chosen_slot + pt['dur']
            task_scheduled_regions[pt['id']] = chosen_region

    save_cluster_state(current_state)
    return final_manifest




def plan_workflow_alg1(run_id, submission_time, deadline_date, tasks_metadata, source_region=REGIONS[0]):
    """
    IMPLEMENTS ALG 1:
    - Finds the best window for the WHOLE workflow.
    - Locks resources exclusively for this workflow's path.
    """
    plan_path = get_safe_plan_path(run_id)
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    total_duration = sum(t['dur'] for t in tasks_metadata)
    total_cores = max(t['cores'] for t in tasks_metadata)
    
    intensities = load_carbon_data(source_region, submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    
    if sub_naive in intensities:
        base = intensities[sub_naive]
    elif intensities:
        base = intensities[sorted(intensities.keys())[0]]
    else:
        base = 400

    best_avg = base
    best_start = submission_time
    best_reg = source_region

    if submission_time.tzinfo is None:
        submission_time = submission_time.replace(tzinfo=dt.timezone.utc)

    if isinstance(deadline_date, str):
        deadline_date = dt.datetime.fromisoformat(deadline_date)
    if deadline_date.tzinfo is None:
        deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)
    window_hours = int((deadline_date - submission_time).total_seconds() // 3600)

    for region in REGIONS:
        intensities = load_carbon_data(region, submission_time, deadline_date)
        
        for offset in range(max(1, window_hours - total_duration + 1)):
            start_cand = submission_time + dt.timedelta(hours=offset)
            start_cand = start_cand.replace(minute=0, second=0, microsecond=0, tzinfo=None)
            
            window_vals = [intensities.get(start_cand + dt.timedelta(hours=h), 999) for h in range(total_duration)]
            avg_carbon = sum(window_vals) / len(window_vals)

            if has_capacity(region, start_cand, total_duration, total_cores):
                if avg_carbon < best_avg:
                    best_avg = avg_carbon
                    best_start = start_cand
                    best_reg = region

    manifest = {}
    current_task_time = best_start
    for t in tasks_metadata:
        manifest[t['id']] = {
            "start": current_task_time.isoformat(),
            "region": best_reg,
            "cores": t['cores'],
            "dur": t['dur']
        }
        current_task_time += dt.timedelta(hours=t['dur'])

    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    
    pt_intensities = load_carbon_data(source_region, submission_time, deadline_date)
    if sub_naive in pt_intensities:
        pt_start_intensity = pt_intensities[sub_naive]
    elif pt_intensities:
        available_times = sorted(pt_intensities.keys())
        pt_start_intensity = pt_intensities[available_times[0]]
        _log(f"[DEBUG] Exact time {sub_naive} not in {source_region} logs. Using closest: {available_times[0]}")
    else:
        pt_start_intensity = 400
        _log(f"[DEBUG] {source_region} logs empty for this window. Using global fallback.")

    local_execution_carbon = pt_start_intensity * (total_duration * total_cores)
    carbon_saved = local_execution_carbon - best_avg * (total_duration * total_cores)
    _log("\n[ALGORITHM 1 REPORT]")
    _log(f"  - Local {source_region} Start Intensity: {pt_start_intensity} g/kWh")
    _log(f"  - Winner {best_reg} Start Intensity: {best_avg} g/kWh")
    _log(f"  - Net Savings: {round(carbon_saved, 2)} gCO2eq")

    stats = calculate_standardized_stats(manifest, {reg: load_carbon_data(reg, submission_time, deadline_date) for reg in REGIONS}, sub_naive, len(tasks_metadata))

    plot_optimization_window_heatmap(
        manifest,
        {reg: load_carbon_data(reg, submission_time, deadline_date) for reg in REGIONS},
        sub_naive,
        deadline_date,
        plot_name=f"alg1_{run_id}",
        stats=stats
    )

    lock_resources(best_reg, best_start, total_duration, total_cores)
    with open(plan_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    return manifest

def plan_workflow_alg2(run_id, submission_time, deadline_date, tasks_metadata, data_size_gb, in_memory_state=None, source_region=REGIONS[0]):
    """
    IMPLEMENTS ALG 2:
    - Finds the best window for the WHOLE workflow in a single region.
    - Tasks are executed strictly sequentially back-to-back without gaps.
    - Incorporates data transfer time penalties and network carbon taxes.
    - Uses high-fidelity closest-match lookups to prevent carbon calculation failures.
    """
    _log(f"[ALG 2] Planning with high-fidelity tracking for {data_size_gb}GB | Workload: {run_id}")
    plan_path = get_safe_plan_path(run_id)
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    total_duration = sum(t['dur'] for t in tasks_metadata)
   
    if isinstance(submission_time, str):
        submission_time = dt.datetime.fromisoformat(submission_time)
    if submission_time.tzinfo is None:
        submission_time = submission_time.replace(tzinfo=dt.timezone.utc)
        
    if isinstance(deadline_date, str): 
        deadline_date = dt.datetime.fromisoformat(deadline_date)
    deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)

    source_traces = load_carbon_data(source_region, submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    source_base_ci = source_traces.get(sub_naive, source_traces[min(source_traces.keys(), key=lambda d: abs(d-sub_naive))] if source_traces else 450)

    best_total_impact = float('inf')
    best_reg, best_start, best_manifest = None, None, {}

    if in_memory_state is None:
        in_memory_state = load_cluster_state()

    for region in REGIONS:
        time_penalty, carbon_penalty = calculate_transfer_penalty(region, data_size_gb, source_region)
        eff_deadline = deadline_date - dt.timedelta(hours=time_penalty)
        _log(f"\n[DEBUG] Evaluating {region} | Transfer Time Penalty: {round(time_penalty, 2)}h | Carbon Penalty: {round(carbon_penalty, 2)}g | Effective Deadline: {eff_deadline} | Total Workflow Duration: {total_duration}h")
        intensities = load_carbon_data(region, submission_time, eff_deadline)
        if not intensities: 
            continue

        window_hours = int((eff_deadline - submission_time).total_seconds() // 3600)

        _log(f"[DEBUG] Available hours in {region} for this workflow (after transfer penalty): {window_hours}h")
        for offset in range(max(1, int(window_hours - total_duration + 1))):
            start_cand = submission_time + dt.timedelta(hours=offset + time_penalty)
            _log(f"[DEBUG] Testing {region} with candidate start {start_cand} (offset {offset}h + transfer penalty {round(time_penalty, 2)}h)")
            sim_carbon, can_fit, cand_manifest = 0, True, {}
            temp_search_state = copy.deepcopy(in_memory_state)
            
            current_task_time = start_cand

            for t in tasks_metadata:
                t_lookup = current_task_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
                
                can_reserve = True
                for h in range(t['dur']):
                    ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                    usage = temp_search_state.get(region, {}).get(ts, 0)
                    if usage + t['cores'] > CORES_LIMIT:
                        can_reserve = False
                        break
                
                if not can_reserve:
                    can_fit = False
                    break

                t_ci = intensities.get(min(intensities.keys(), key=lambda d: abs(d - t_lookup)), 400)
                t_aware_carbon = t_ci * (t['dur'] * t['cores'])
                sim_carbon += t_aware_carbon
                t_baseline_carbon = source_base_ci * (t['dur'] * t['cores'])

                cand_manifest[t['id']] = {
                    "start": t_lookup.isoformat(), 
                    "dur": t['dur'], 
                    "region": region, 
                    "cores": t['cores'],
                    "pt_ci": round(source_base_ci, 2),
                    "target_ci": round(t_ci, 2),
                    "saved_g": round(t_baseline_carbon - t_aware_carbon, 2), 
                    "pct": round(((t_baseline_carbon - t_aware_carbon) / t_baseline_carbon * 100), 1) if t_baseline_carbon > 0 else 0
                }

                for h in range(t['dur']):
                    ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                    if region not in temp_search_state: 
                        temp_search_state[region] = {}
                    temp_search_state[region][ts] = temp_search_state[region].get(ts, 0) + t['cores']

                current_task_time += dt.timedelta(hours=t['dur'])

            total_impact = sim_carbon + carbon_penalty
            if can_fit and total_impact < best_total_impact:
                best_total_impact = total_impact
                best_start = start_cand
                best_reg = region
                best_manifest = cand_manifest

    if best_reg:
        _log(f"\n[ALGORITHM 2 REPORT] Workflow: {run_id}")
        _log(f"{'Task ID':<15} | {'Region':<8} | {'Saved (g)':<10} | {'Reduction %'}")
        _log("-" * 50)
        
        all_regions_window_data = {reg: load_carbon_data(reg, submission_time, deadline_date) for reg in REGIONS}

        stats = calculate_standardized_stats(best_manifest, all_regions_window_data, sub_naive, len(tasks_metadata))
        
        plot_optimization_window_heatmap(
            best_manifest=best_manifest,
            all_regions_data=all_regions_window_data,
            submission_time=submission_time,
            deadline=deadline_date,
            plot_name=f"alg2_{run_id}",
            stats=stats
        )

        actual_sum_saved_g = 0
        actual_sum_baseline_g = 0
        
        for tid, d in best_manifest.items():
            _log(f"{tid:<15} | {d['region']:<8} | {d['saved_g']:<10} | {d['pct']}%")
            actual_sum_saved_g += d['saved_g']
            actual_sum_baseline_g += (d['saved_g'] / (d['pct']/100)) if d['pct'] > 0 else 0
        
        total_reduction_pct = (actual_sum_saved_g / actual_sum_baseline_g * 100) if actual_sum_baseline_g > 0 else 0
        
        _log(f"--- TOTAL SAVED (Execution): {round(actual_sum_saved_g, 2)}g ({round(total_reduction_pct, 1)}%) ---\n")
        _log(f"--- TOTAL if ran in PT: {round(actual_sum_baseline_g, 2)}g ---\n")
        
        current_cluster_state = load_cluster_state()
        for tid, tdata in best_manifest.items():
            reg = tdata['region']
            task_start = dt.datetime.fromisoformat(tdata['start']).replace(minute=0, second=0, microsecond=0, tzinfo=None)
            
            if reg not in current_cluster_state:
                current_cluster_state[reg] = {}
                
            for h in range(tdata['dur']):
                ts_str = (task_start + dt.timedelta(hours=h)).isoformat()
                current_cluster_state[reg][ts_str] = current_cluster_state[reg].get(ts_str, 0) + tdata['cores']
                
        save_cluster_state(current_cluster_state)
        with open(plan_path, 'w') as f: 
            json.dump(best_manifest, f, indent=4)
            
        return best_manifest

    return {}

def plan_workflow_alg3(run_id, submission_time, deadline_date, tasks_metadata, edges, sla_level, in_memory_state=None, source_region=None):
    """IMPLEMENTS ALG 3:
    - Spatio-Temporal Shifting with Task-Level Granularity.
    - Evaluates each task independently, allowing them to be scheduled in different regions and times within the same workflow.
    - Uses the dynamic critical path to understand the workflow's temporal constraints and prioritize scheduling decisions.
    - Aims to minimize total carbon impact while respecting the workflow's deadline and SLA requirements.
    """

    if source_region is None:
        _log("[ALG 3] No source region provided. Defaulting to REGIONS[0].")
        source_region = REGIONS[0]
    else:
        _log(f"[ALG 3] Planning with source region: {source_region}")

    plan_path = get_safe_plan_path(run_id)
    if in_memory_state is None and os.path.exists(plan_path):
        with open(plan_path, 'r') as f:
            return json.load(f)

    cp_duration_hours = get_dynamic_critical_path(tasks_metadata, edges, sla_level)
    if isinstance(submission_time, str):
        submission_time = dt.datetime.fromisoformat(submission_time)

    if submission_time.tzinfo is None: submission_time = submission_time.replace(tzinfo=dt.timezone.utc)
    if isinstance(deadline_date, str): deadline_date = dt.datetime.fromisoformat(deadline_date)
    deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)
    max_window_hours = int((deadline_date - submission_time).total_seconds() // 3600)

    source_traces = load_carbon_data(source_region, submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    source_base_ci = source_traces.get(sub_naive, source_traces[min(source_traces.keys(), key=lambda d: abs(d-sub_naive))] if source_traces else 450)
    
    best_reg, best_start, best_manifest, best_duration = None, None, {}, int(cp_duration_hours)

    if in_memory_state is None:
        in_memory_state = load_cluster_state()

    all_regions_intensities = {}
    for r in REGIONS:
        time_penalty, _ = calculate_transfer_penalty(r, DATA_SIZE_GB, source_region)
        eff_deadline = deadline_date - dt.timedelta(hours=time_penalty)
        intensities = load_carbon_data(r, submission_time, eff_deadline)
        if intensities:
            all_regions_intensities[r] = intensities

    rle_index = preprocess_carbon_to_rle_buckets(all_regions_intensities, REGIONS, max_window_hours)

    workflow_fits = cp_duration_hours <= max_window_hours
    _log(f"[ALG3] Workflow {run_id} critical path duration: {cp_duration_hours}h | Window: {max_window_hours}h | Fits: {workflow_fits}")
    if not workflow_fits:
        best_manifest = _schedule_late_with_parallel_transfers(
            tasks_metadata=tasks_metadata,
            source_region=source_region,
            submission_time=submission_time,
            max_window_hours=max_window_hours,
            base_ci=source_base_ci,
            in_memory_state=in_memory_state
        )
    else:
        best_manifest = find_greenest_schedule_via_rle(
            REGIONS=REGIONS,
            tasks_metadata=tasks_metadata,
            all_regions_intensities=all_regions_intensities,
            rle_index=rle_index,
            submission_time=submission_time,
            max_window_hours=max_window_hours,
            base_ci=source_base_ci,
            in_memory_state=in_memory_state,
            source_region=source_region
        )

    best_reg = None
    if best_manifest:
        first_task_id = list(best_manifest.keys())[0]
        best_reg = best_manifest[first_task_id]['region']

    if best_manifest:

        all_regions_window_data = {}
        for reg in REGIONS:
            all_regions_window_data[reg] = load_carbon_data(reg, submission_time, deadline_date)

        sub_naive = submission_time.replace(tzinfo=None)

        last_task_end = max(
            dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None) + dt.timedelta(hours=tdata['dur'])
            for tdata in best_manifest.values()
        )
        stats_window_end = max(deadline_date.replace(tzinfo=None), last_task_end)

        # Reload traces covering the extended window if needed
        if last_task_end > deadline_date.replace(tzinfo=None):
            for reg in REGIONS:
                all_regions_window_data[reg] = load_carbon_data(reg, submission_time, last_task_end)

        stats = calculate_standardized_stats(
            best_manifest, all_regions_window_data, sub_naive,
            len(tasks_metadata), deadline=deadline_date
        )
        stats = _apply_late_carbon_penalty(best_manifest, all_regions_window_data, stats, DEADLINE_PENALTY_MULTIPLIER)

        plot_optimization_window_heatmap(
            best_manifest=best_manifest,
            all_regions_data=all_regions_window_data,
            submission_time=submission_time,
            deadline=stats_window_end,
            plot_name=f"alg3_{run_id}",
            stats=stats
        )

        for tid, d in best_manifest.items():
            late_flag = " [LATE]" if d.get("scheduled_late") else ""

        with open(plan_path, 'w') as f:
            json.dump(best_manifest, f, indent=4)

        return best_manifest

    return {}

    

def plan_batch_alg4(dag_runs_metadata, heuristic="EDF", source_region=REGIONS[0], batch_label=None):
    """
    Explores heuristics like 'Longest Job First' (LWF) and 'Earliest Deadline First' (EDF) to reduce fragmentation.
    The 'heuristic' argument can be set to either "EDF" or "LWF" to switch approaches seamlessly.
    """
    global_scratchpad = load_cluster_state()

    if heuristic == "EDF":
        _log("[BATCH PLANNER] Using Earliest Deadline First (EDF) heuristic.")
        queue = sorted(
            dag_runs_metadata,
            key=lambda x: dt.datetime.fromisoformat(x['deadline']) if isinstance(x['deadline'], str) else x['deadline']
        )
    else:
        _log("[BATCH PLANNER] Using Longest Workflow First (LWF) heuristic.")
        queue = sorted(
            dag_runs_metadata,
            key=lambda x: sum(t['dur'] for t in x['tasks']) * max(t['cores'] for t in x['tasks']),
            reverse=True
        )

    all_subs = []
    all_dls = []
    total_requested_tasks = 0
    for wf in dag_runs_metadata:
        s, d = wf['submission'], wf['deadline']
        dl_dt = dt.datetime.fromisoformat(d) if isinstance(d, str) else d
        all_dls.append(dl_dt.replace(tzinfo=None))
        sub_dt = dt.datetime.fromisoformat(s) if isinstance(s, str) else s
        all_subs.append(sub_dt.replace(tzinfo=None))
        total_requested_tasks += len(wf['tasks'])

    batch_start_bound = min(all_subs)
    batch_end_bound = max(all_dls)
    _log(f"[BATCH PLANNER] Batch Window: {batch_start_bound} to {batch_end_bound} | Total Requested Tasks: {total_requested_tasks}")

    batch_results = {}
    total_manifest = {}

    for workflow in queue:
        wf_submission = workflow['submission']
        wf_submission = (dt.datetime.fromisoformat(wf_submission)
                          if isinstance(wf_submission, str) else wf_submission)

        plan = plan_workflow_alg3(
            workflow['run_id'],
            wf_submission, 
            workflow['deadline'],
            workflow['tasks'],
            workflow['edges'],
            workflow['sla'],
            in_memory_state=global_scratchpad,
            source_region=source_region
        )

        batch_results[workflow['run_id']] = plan

        for tid, tdata in plan.items():
            unique_key = f"{workflow['run_id']}_{tid}"
            total_manifest[unique_key] = tdata

            start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
            reg = tdata['region']
            t_lookup = start_dt.replace(minute=0, second=0, microsecond=0)

            for h in range(int(tdata['dur'])):
                slot_time_obj = t_lookup + dt.timedelta(hours=h)
                ts = slot_time_obj.isoformat()

                if reg not in global_scratchpad:
                    global_scratchpad[reg] = {}

                current_usage = global_scratchpad[reg].get(ts, 0)
                global_scratchpad[reg][ts] = current_usage + tdata.get('cores', 2)

    save_cluster_state(global_scratchpad)

    if not total_manifest:
        _log("[BATCH PLANNER] WARNING: No tasks were planned. Skipping heatmap.", logging.WARNING)
        return batch_results

    all_traces = {reg: load_carbon_data(reg, batch_start_bound, batch_end_bound) for reg in REGIONS}
    start_naive = batch_start_bound.replace(tzinfo=None)
    stats_to_display = calculate_standardized_stats(total_manifest, all_traces, start_naive, total_requested_tasks)

    save_chosen_path(
        batch_label=batch_label or f"alg4_{heuristic}_unlabeled",
        algorithm=f"Alg4_Batch_{heuristic}",
        order_run_ids=[wf['run_id'] for wf in queue],
        task_regions={
            f"{run_id}::{tid}": tdata['region']
            for run_id, plan in batch_results.items()
            for tid, tdata in (plan or {}).items()
        },
        exec_carbon_g=stats_to_display.get('total_carbon', 0.0),
        transfer_carbon_g=stats_to_display.get('total_transfer_carbon', 0.0),
        extra={"heuristic": heuristic},
    )

    batch_id = f"batch_{heuristic}_{dt.datetime.now().strftime('%H%M%S')}"
    plot_optimization_window_heatmap(
        total_manifest,
        all_traces,
        batch_start_bound,
        batch_end_bound,
        batch_id,
        stats=stats_to_display
    )

    return batch_results


def plan_batch_baseline(dag_runs_metadata, baseline_type="temporal", heuristic="EDF", source_region=REGIONS[0], batch_label=None):
    """
    Unified BATCH BASELINE Wrapper (Temporal-Only or Spatial-Only)
    - baseline_type: Accepts "temporal" or "spatial" to toggle the underlying baseline algorithm.
    - heuristic: Explores 'Longest Job First' (LWF) or 'Earliest Deadline First' (EDF).
    """
    global_scratchpad = load_cluster_state()
    
    is_temporal = (baseline_type.lower() == "temporal")
    prefix = "[BATCH TEMPORAL]" if is_temporal else "[BATCH SPATIAL]"
    algo_name = f"Base_Temporal_Batch_{heuristic}" if is_temporal else f"Base_Spatial_Batch_{heuristic}"
    default_label = f"base_temporal_{heuristic}_unlabeled" if is_temporal else f"base_spatial_{heuristic}_unlabeled"
    batch_id_prefix = "batch_base_temp" if is_temporal else "batch_base_spat"

    if heuristic == "EDF":
        _log(f"{prefix} Using Earliest Deadline First (EDF) heuristic.")
        queue = sorted(
            dag_runs_metadata,
            key=lambda x: dt.datetime.fromisoformat(x['deadline']) if isinstance(x['deadline'], str) else x['deadline']
        )
    else:
        _log(f"{prefix} Using Longest Workflow First (LWF) heuristic.")
        queue = sorted(
            dag_runs_metadata,
            key=lambda x: sum(t['dur'] for t in x['tasks']) * max(t['cores'] for t in x['tasks']),
            reverse=True
        )

    all_subs = []
    all_dls = []
    total_requested_tasks = 0
    for wf in dag_runs_metadata:
        s, d = wf['submission'], wf['deadline']
        dl_dt = dt.datetime.fromisoformat(d) if isinstance(d, str) else d
        all_dls.append(dl_dt.replace(tzinfo=None))
        sub_dt = dt.datetime.fromisoformat(s) if isinstance(s, str) else s
        all_subs.append(sub_dt.replace(tzinfo=None))
        total_requested_tasks += len(wf['tasks'])

    batch_start_bound = min(all_subs)
    batch_end_bound = max(all_dls)
    _log(f"{prefix} Batch Window: {batch_start_bound} to {batch_end_bound} | Total Requested Tasks: {total_requested_tasks}")

    batch_results = {}
    total_manifest = {}

    for workflow in queue:
        wf_submission = workflow['submission']
        wf_submission = (dt.datetime.fromisoformat(wf_submission)
                          if isinstance(wf_submission, str) else wf_submission)

        if is_temporal:
            plan = plan_baseline_task_temporal_only(
                run_id=workflow['run_id'],
                submission_time=wf_submission, 
                deadline_date=workflow['deadline'],
                tasks_metadata=workflow['tasks'],
                data_size_gb=workflow.get('data_size_gb', DATA_SIZE_GB),
                in_memory_state=global_scratchpad,
                source_region=source_region
            )
        else:
            plan = plan_baseline_task_spatial_only(
                run_id=workflow['run_id'],
                submission_time=wf_submission, 
                deadline_date=workflow['deadline'],
                tasks_metadata=workflow['tasks'],
                data_size_gb=workflow.get('data_size_gb', DATA_SIZE_GB),
                in_memory_state=global_scratchpad,
                source_region=source_region
            )

        batch_results[workflow['run_id']] = plan

        for tid, tdata in (plan or {}).items():
            unique_key = f"{workflow['run_id']}_{tid}"
            total_manifest[unique_key] = tdata

            start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
            reg = tdata['region']
            t_lookup = start_dt.replace(minute=0, second=0, microsecond=0)

            dur_slots = max(1, int(math.ceil(tdata.get('dur', 1.0))))
            for h in range(dur_slots):
                slot_time_obj = t_lookup + dt.timedelta(hours=h)
                ts = slot_time_obj.isoformat()

                if reg not in global_scratchpad:
                    global_scratchpad[reg] = {}

                current_usage = global_scratchpad[reg].get(ts, 0)
                global_scratchpad[reg][ts] = current_usage + tdata.get('cores', 1)

    save_cluster_state(global_scratchpad)

    if not total_manifest:
        _log(f"{prefix} WARNING: No tasks were planned. Skipping heatmap.", logging.WARNING)
        return batch_results

    max_task_end = batch_end_bound
    for tdata in total_manifest.values():
        if 'start' in tdata and 'dur' in tdata:
            t_end = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None) + dt.timedelta(hours=float(tdata['dur']))
            if t_end > max_task_end:
                max_task_end = t_end

    all_traces = {reg: load_carbon_data(reg, batch_start_bound, max_task_end) for reg in REGIONS}
    start_naive = batch_start_bound.replace(tzinfo=None)
    stats_to_display = calculate_standardized_stats(total_manifest, all_traces, start_naive, total_requested_tasks)

    save_chosen_path(
        batch_label=batch_label or default_label,
        algorithm=algo_name,
        order_run_ids=[wf['run_id'] for wf in queue],
        task_regions={
            f"{run_id}::{tid}": tdata['region']
            for run_id, plan in batch_results.items()
            for tid, tdata in (plan or {}).items()
        },
        exec_carbon_g=stats_to_display.get('total_carbon', 0.0),
        transfer_carbon_g=stats_to_display.get('total_transfer_carbon', 0.0),
        extra={"heuristic": heuristic, "baseline": baseline_type},
    )

    batch_id = f"{batch_id_prefix}_{heuristic}_{dt.datetime.now().strftime('%H%M%S')}"
    plot_optimization_window_heatmap(
        total_manifest,
        all_traces,
        batch_start_bound,
        max_task_end,
        batch_id,
        stats=stats_to_display
    )

    return batch_results

def plan_oracle(dag_runs_metadata, source_region=REGIONS[0], batch_label=None):
    """
    ORACLE SCHEDULER (Strict Carbon Minimization)
    """
    initial_cluster_state = load_cluster_state()

    all_subs, all_dls, total_requested_tasks = [], [], 0
    for wf in dag_runs_metadata:
        s, d = wf['submission'], wf['deadline']
        dl_dt = dt.datetime.fromisoformat(d) if isinstance(d, str) else d
        all_dls.append(dl_dt.replace(tzinfo=None))
        sub_dt = dt.datetime.fromisoformat(s) if isinstance(s, str) else s
        all_subs.append(sub_dt.replace(tzinfo=None))
        total_requested_tasks += len(wf['tasks'])

    batch_start_bound = min(all_subs)
    batch_end_bound = max(all_dls)

    total_perms = math.factorial(len(dag_runs_metadata))
    _log(f"\n[ORACLE] Batch Window: {batch_start_bound} to {batch_end_bound} | "
          f"Total Requested Tasks: {total_requested_tasks}")
    _log(f"[ORACLE] Testing all {total_perms} possible orderings for lowest carbon...")

    best_carbon      = float('inf')
    best_results     = {}
    best_manifest    = {}
    best_order       = []
    best_stats       = {}
    best_scratchpad  = None

    for queue in itertools.permutations(dag_runs_metadata):

        scratchpad     = copy.deepcopy(initial_cluster_state)
        batch_results  = {}
        total_manifest = {}

        for workflow in queue:
            wf_submission = workflow['submission']
            wf_submission = (dt.datetime.fromisoformat(wf_submission)
                              if isinstance(wf_submission, str) else wf_submission)

            plan = plan_workflow_alg3(
                workflow['run_id'],
                wf_submission,
                workflow['deadline'],
                workflow['tasks'],
                workflow['edges'],
                workflow['sla'],
                in_memory_state=scratchpad,
                source_region=source_region
            )

            if not plan:
                batch_results[workflow['run_id']] = {}
                continue

            batch_results[workflow['run_id']] = plan

            for tid, tdata in plan.items():
                unique_key = f"{workflow['run_id']}_{tid}"
                total_manifest[unique_key] = tdata

                start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
                reg = tdata['region']
                scratchpad = update_scratchpad(scratchpad, reg, start_dt, tdata['dur'], tdata['cores'])

        if not total_manifest:
            continue

        all_traces = {reg: load_carbon_data(reg, batch_start_bound, batch_end_bound) for reg in REGIONS}
        start_naive = batch_start_bound.replace(tzinfo=None)
        stats = calculate_standardized_stats(total_manifest, all_traces, start_naive, total_requested_tasks)

        ordering_carbon = stats.get(
            'total_carbon_with_transfer',
            stats.get('total_carbon', float('inf')),
        )
        if ordering_carbon < best_carbon:
            best_carbon     = ordering_carbon
            best_results    = batch_results
            best_manifest   = total_manifest
            best_order      = [wf['run_id'] for wf in queue]
            best_stats      = stats
            best_scratchpad = scratchpad

    if not best_results:
        _log("[ORACLE] WARNING: No valid ordering found across all permutations.", logging.WARNING)
        return {}

    _log(f"[ORACLE] Evaluation complete. Winner ({total_perms} orderings tested): "
          f"{round(best_carbon, 2)}gCO2eq (execution + transfer)")

    save_cluster_state(best_scratchpad)

    batch_id = f"oracle_{dt.datetime.now().strftime('%H%M%S')}"
    plot_optimization_window_heatmap(
        best_manifest,
        {reg: load_carbon_data(reg, batch_start_bound, batch_end_bound) for reg in REGIONS},
        batch_start_bound,
        batch_end_bound,
        batch_id,
        stats=best_stats
    )

    save_chosen_path(
        batch_label=batch_label or "oracle_unlabeled",
        algorithm="Oracle",
        order_run_ids=best_order,
        task_regions={
            f"{run_id}::{tid}": tdata['region']
            for run_id, plan in best_results.items()
            for tid, tdata in (plan or {}).items()
        },
        exec_carbon_g=best_stats.get('total_carbon', 0.0),
        transfer_carbon_g=best_stats.get('total_transfer_carbon', 0.0),
        extra={"permutations_tested": total_perms},
    )

    return best_results


def get_safe_plan_path(run_id):
    """Build a stable filesystem path for a workflow plan JSON."""
    safe_id = run_id.replace(":", "_").replace("+", "_")
    return os.path.join(PLAN_DIR, f"plan_{safe_id}.json")

def has_capacity(region, start_time, duration, cores, file_path=CLUSTER_FILE):
    """Check whether a region has enough free hourly capacity for a task."""
    if not os.path.exists(file_path): return True
    
    with open(file_path, 'r') as f:
        cluster = json.load(f)
    
    base_time = start_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    for h in range(int(duration)):
        t_str = (base_time + dt.timedelta(hours=h)).isoformat()
        
        usage = cluster.get(region, {}).get(t_str, 0)
        
        _log(f"[CAPACITY CHECK] {region} at {t_str} | Current: {usage} | Request: {cores} | Limit: {CORES_LIMIT}")
        
        if usage + cores > CORES_LIMIT:
            return False
    return True



def lock_resources(region, start_time, duration, cores, file_path=CLUSTER_FILE):
    """Reserve capacity in the cluster-state file for a scheduled task."""
    data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
    if region not in data:
        data[region] = {}
    _log(f"[RESOURCE LOCK] Reserving {cores} cores in {region} from {start_time} for {duration} hours.")
    base_time = start_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        
    for h in range(int(duration)):
        t_str = (base_time + dt.timedelta(hours=h)).isoformat()
        
        current = data[region].get(t_str, 0)
        if isinstance(current, dict): current = 0
        
        data[region][t_str] = current + cores
        
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def task_policy(task):
    """Airflow hook that schedules each task according to the saved carbon-aware plan."""

    def carbon_resource_manager(context):
        ti = context['ti']
        dr = context['dag_run']
        
        plan_path = get_safe_plan_path(dr.run_id)

        if not os.path.exists(plan_path):
            
            if ACTIVE_ALGORITHM in [1, 2, 3, 5, 6, 7]:
                sub_time = dr.start_date if dr.start_date else dt.datetime.now(dt.timezone.utc)
                deadline_str = dr.conf.get('deadline_iso', (dt.datetime.now() + dt.timedelta(hours=24)).isoformat())
                deadline = dt.datetime.fromisoformat(deadline_str)
                tasks = dr.conf.get('workflow_structure', [])
                edges = []
                if hasattr(dr, 'dag') and dr.dag:
                    for task in dr.dag.tasks:
                        for downstream_id in task.downstream_task_ids:
                            edges.append((task.task_id, downstream_id))
                _log(f"[METADATA] edges: {edges}")
                sla = dr.conf.get('sla_level', 95)
                _log(f"{context}")
                if ACTIVE_ALGORITHM == 1:
                    plan_workflow_alg1(dr.run_id, sub_time, deadline, tasks)
                elif ACTIVE_ALGORITHM == 2:
                    plan_workflow_alg2(dr.run_id, sub_time, deadline, tasks, DATA_SIZE_GB)
                elif ACTIVE_ALGORITHM == 3:
                    plan_workflow_alg3(dr.run_id, sub_time, deadline, tasks, edges, sla)
                elif ACTIVE_ALGORITHM == 5:
                    plan_baseline_A(dr.run_id, sub_time, deadline, tasks, edges, load_cluster_state())
                elif ACTIVE_ALGORITHM == 6:
                    plan_baseline_B(dr.run_id, sub_time, deadline, tasks, edges, load_cluster_state())
                elif ACTIVE_ALGORITHM == 7:
                    plan_atomic(dr.run_id, sub_time, deadline, tasks, edges, load_cluster_state())
               
            elif ACTIVE_ALGORITHM in [4, 8, 9, 13, 15]:
                reg_file = os.path.join(WAITING_ROOM_DIR, f"{dr.run_id}.json")
                
                if not os.path.exists(reg_file):
                    metadata = {
                        "run_id": dr.run_id,
                        "submission": dr.start_date.isoformat() if dr.start_date else dt.datetime.now(dt.timezone.utc).isoformat(),
                        "deadline": dr.conf.get('deadline_iso', (dt.datetime.now() + dt.timedelta(hours=24)).isoformat()),
                        "tasks": dr.conf.get('workflow_structure', []),
                        "edges": dr.conf.get('edges', []),
                        "sla": dr.conf.get('sla_level', 95)
                    }
                    with open(reg_file, 'w') as f:
                        json.dump(metadata, f)
                    _log(f"[WAITING ROOM] {dr.run_id} entered the queue.")

                waiting_files = [f for f in os.listdir(WAITING_ROOM_DIR) if f.endswith('.json')]
                
                if len(waiting_files) >= BATCH_SIZE:
                    lock_file = os.path.join(WAITING_ROOM_DIR, "leader.lock")
                    try:
                        with open(lock_file, "w") as lf:
                            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            
                            run_batch_coordinator(waiting_files)
                            
                            fcntl.flock(lf, fcntl.LOCK_UN)
                            os.remove(lock_file)
                    except (BlockingIOError, IOError):
                        pass
                
                if not os.path.exists(plan_path):
                    raise AirflowRescheduleException(
                        reschedule_date=dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60)
                    )
        _log(f"[{ti.task_id}] Plan found at {plan_path}. Preparing to execute with carbon-aware scheduling.")
        with open(plan_path, 'r') as f:
            manifest = json.load(f)
            
        my_plan = manifest.get(ti.task_id)
        if not my_plan: return
            
        target_time = dt.datetime.fromisoformat(my_plan['start']).replace(tzinfo=dt.timezone.utc)
        _log(f"[{ti.task_id}] Scheduled to run at {target_time} in region {my_plan['region']}")
        if dt.datetime.now(dt.timezone.utc) < target_time - dt.timedelta(minutes=2):
            raise AirflowRescheduleException(reschedule_date=target_time)

        ti.task.executor_config = {"pod_override": {"spec": {"nodeSelector": {"custom_region": my_plan['region']}}}}

    task.pre_execute = carbon_resource_manager

def run_batch_coordinator(waiting_files):
    """Dispatch the queued workflows to the selected batch scheduling algorithm."""
    _log(f"\n[COORDINATOR] Queue threshold reached! Running Algorithm 4 for {len(waiting_files)} workflows.")
    
    batch_data = []
    for f_name in waiting_files:
        try:
            with open(os.path.join(WAITING_ROOM_DIR, f_name), 'r') as f:
                batch_data.append(json.load(f))
        except Exception:
            continue
            
    if ACTIVE_ALGORITHM == 8:
        plan_oracle(batch_data)
    elif ACTIVE_ALGORITHM == 9:
        plan_atomic_oracle(batch_data)
    elif ACTIVE_ALGORITHM == 13:
        plan_batch_baseline(batch_data, baseline_type="temporal", heuristic="LWF")
    elif ACTIVE_ALGORITHM == 15:
        plan_batch_baseline(batch_data, baseline_type="spatial", heuristic="LWF")
    else:
        plan_batch_alg4(batch_data)

    for f_name in waiting_files:
        try:
            os.remove(os.path.join(WAITING_ROOM_DIR, f_name))
        except OSError:
            pass
    _log("[COORDINATOR] Plans distributed. Waiting room cleared.\n")
