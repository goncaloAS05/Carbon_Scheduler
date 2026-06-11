import os
import json
import fcntl  # NOTE: also imported again at line ~2014 (duplicate removed below)
import copy
import datetime as dt
import math
# REMOVED: 'from datetime import timedelta, datetime' — redundant since all usages
# already use the dt.timedelta / dt.datetime namespace consistently throughout the file.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from airflow.exceptions import AirflowRescheduleException

# --- System Paths & Storage Configurations ---
AIRFLOW_BASE_DIR = os.path.expanduser("~/Carbon_Scheduler/airflow")
CLUSTER_FILE     = os.path.join(AIRFLOW_BASE_DIR, "cluster_state.json")
DATA_DIR         = os.path.join(AIRFLOW_BASE_DIR, "plugins", "data", "")
PLAN_DIR         = os.path.join(AIRFLOW_BASE_DIR, "plans", "")
HISTORY_FILE     = os.path.join(AIRFLOW_BASE_DIR, "task_history.json")
WAITING_ROOM_DIR = os.path.join(AIRFLOW_BASE_DIR, "waiting_room", "")

# --- Simulation Engine Parameters ---
ACTIVE_ALGORITHM = 4
BATCH_SIZE = 3
TOTAL_CORES_PER_REGION = 4
REGIONS = ["DE", "PL"]  

# --- Network Modeling Coefficients ---
DATA_SIZE_GB = 1
TRANSFER_SPEED_GBPS = 400 / 8  # Converted to MB/s equivalent bounds
KWH_PER_GB = 0.001875

# Initialize required execution directories
os.makedirs(PLAN_DIR, exist_ok=True)
os.makedirs(WAITING_ROOM_DIR, exist_ok=True)

def plot_optimization_window_heatmap(best_manifest, all_regions_data, submission_time, deadline, plot_name="default", stats=None):
    """
    Plots the carbon intensity heatmap with blue task boxes.
    Includes professor-requested multi-task workflow performance metrics in the footer.
    """
    # 1. timezones are stripped
    sub_naive = submission_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    dl_naive = deadline.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    # 2. total hours
    total_window_hours = int((dl_naive - sub_naive).total_seconds() // 3600) + 1
    
    # 3. time slots
    time_slots = [(sub_naive + dt.timedelta(hours=i)) for i in range(total_window_hours)]
    time_labels = [t.strftime("%H:00\n%d/%m") for t in time_slots]
    
    regions = list(all_regions_data.keys())
    heatmap_matrix = np.zeros((len(regions), len(time_slots)))
    
    # 4. fill heatmap matrix
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

    # 5. Plotting
    plt.figure(figsize=(max(14, total_window_hours * 0.6), 8.5)) # Slightly increased height for larger textbox
    df = pd.DataFrame(heatmap_matrix, index=regions, columns=time_labels)
    
    ax = sns.heatmap(df, cmap="RdYlGn_r", annot=True, fmt=".0f", annot_kws={"size": 7},
                     cbar_kws={'label': 'gCO2eq/kWh'})

    # 6. Draw Task Boxes with Label Offsetting
    label_collision_counter = {}

    for task_id, data in best_manifest.items():
        if data['region'] in regions:
            reg_row = regions.index(data['region'])
            t_start = dt.datetime.fromisoformat(data['start']).replace(tzinfo=None)
            duration = data.get('dur', 1) 

            hour_idx = int((t_start - sub_naive).total_seconds() // 3600)
            
            if 0 <= hour_idx < len(time_slots):
                # Draw the box
                ax.add_patch(plt.Rectangle((hour_idx, reg_row), duration, 1, 
                                           fill=False, edgecolor='blue', lw=3, zorder=10, alpha=0.6))
                
                # COLLISION MANAGEMENT:
                cell_key = (hour_idx, reg_row)
                offset_idx = label_collision_counter.get(cell_key, 0)
                label_collision_counter[cell_key] = offset_idx + 1
                
                y_pos = reg_row + 0.2 + (offset_idx * 0.15)
                display_id = str(task_id).split('__')[-1]
                
                plt.text(hour_idx + 0.1, y_pos, display_id, 
                         color='blue', weight='bold', ha='left', va='center', 
                         fontsize=8, zorder=11, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0))

    # -------------------------------------------------------------------------
    # 7. UPDATED: Add Professor's Performance Metrics Textbox
    # -------------------------------------------------------------------------
    if stats:
        stats_str = (
            f"Success Rate: {stats['success_rate']}%   |   "
            f"Execution Time: {stats.get('total_execution_time_hours', 0)}h   |   "
            f"Makespan: {stats.get('makespan_hours', 0)}h   |   "
            f"Waiting Time: {stats.get('waiting_time_hours', 0)}h\n"
            f"Avg Task Dur: {stats.get('avg_task_execution_time_hours', 0)}h   |   "
            f"Deadline Fulfilled: {stats.get('deadline_fulfillment_pct', 100.0)}%   |   "
            f"Energy: {stats['total_energy']} kWh   |   "
            f"Total Carbon: {stats['total_carbon']} gCO2"
        )
        
        # Placed at the very bottom center using figure coordinates
        plt.gcf().text(0.5, 0.03, stats_str, fontsize=10.5,
                       ha='center', va='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.6', facecolor='#f9f9f9', alpha=1.0, edgecolor='gray'))

    plt.title(f"Carbon Opportunity Window: {total_window_hours} Hours Total", fontsize=14, pad=25)
    plt.xticks(rotation=0)
    
    # Adjusted rect bottom bounds to 0.12 so the two-line text box has plenty of room
    plt.tight_layout(rect=[0, 0.12, 1, 0.95]) 
    
    # Save
    save_path = os.path.join(PLAN_DIR, f"heatmap_{plot_name}.png")
    plt.savefig(save_path) 
    plt.close()

    
def calculate_standardized_stats(plan, all_traces, window_start, total_requested, deadline=None):
    total_energy = 0.0
    total_carbon = 0.0
    planned_starts = []
    planned_ends = []
    task_durations = []
    active_hours = set()
    POWER_FACTOR = 0.2  # 0.2 kW per core

    for tid, tdata in plan.items():
        # 1. Energy Calculation
        task_energy = tdata['cores'] * tdata['dur'] * POWER_FACTOR
        total_energy += task_energy
        
        # 2. Per-Hour Carbon Calculation
        start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
        planned_starts.append(start_dt)
        task_durations.append(tdata['dur'])
        reg = tdata['region']
        
        for h in range(tdata['dur']):
            hour_obj = start_dt + dt.timedelta(hours=h)
            intensity = all_traces.get(reg, {}).get(hour_obj, 400) 
            total_carbon += (tdata['cores'] * POWER_FACTOR) * intensity
            active_hours.add(hour_obj.isoformat())
            
        planned_ends.append(start_dt + dt.timedelta(hours=tdata['dur']))

    success_rate = round((len(plan) / total_requested) * 100, 1) if total_requested > 0 else 0
    
    if plan and planned_starts and planned_ends:
        first_start = min(planned_starts)
        last_end = max(planned_ends)
        
        total_execution_time = len(active_hours)
        makespan = round((last_end - first_start).total_seconds() / 3600, 2)
        waiting_time = round((first_start - window_start).total_seconds() / 3600, 2)
        avg_task_execution = round(sum(task_durations) / len(task_durations), 2)
        
        if deadline:
            naive_deadline = deadline.replace(tzinfo=None)
            deadline_fulfilled_pct = 100.0 if last_end <= naive_deadline else 0.0
        else:
            deadline_fulfilled_pct = 100.0
            
        # Reverting this key back to 'total_hours' to fix the script crash
        total_hours = round((last_end - window_start).total_seconds() / 3600, 2)
    else:
        total_execution_time = 0
        makespan = 0
        waiting_time = 0
        avg_task_execution = 0
        deadline_fulfilled_pct = 0.0
        total_hours = 0

    return {
        "success_rate": success_rate,
        "total_hours": total_hours,  # Changed back from 'legacy_total_hours'
        "total_energy": round(total_energy, 2),
        "total_carbon": round(total_carbon, 2),
        
        # Professor's metrics remain completely intact
        "total_execution_time_hours": total_execution_time,
        "makespan_hours": makespan,
        "waiting_time_hours": waiting_time,
        "avg_task_execution_time_hours": avg_task_execution,
        "deadline_fulfillment_pct": deadline_fulfilled_pct
    }

def find_earliest_slot(region, requested_cores, state):
    """Finds the first hour (starting from now) where requested_cores are available."""
    now = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    for h in range(240): # Look ahead up to 10 days
        slot = (now + dt.timedelta(hours=h)).isoformat()
        used = state.get(region, {}).get(slot, 0)
        if (used + requested_cores) <= TOTAL_CORES_PER_REGION: 
            return now + dt.timedelta(hours=h)
    return now


# def check_conservative_resource_limits(current_state, region, start_hour, path, TOTAL_CORES_PER_REGION):
#     """
#     Verifica se a alocação deste caminho respeita uma abordagem conservadora,
#     não esgotando os recursos que outros caminhos independentes possam vir a precisar.
#     """
#     # Definimos um teto máximo conservador (ex: usar no máximo 75% da capacidade total da região)
#     CONSERVATIVE_CAPFACTOR = 0.75
#     safe_core_limit = TOTAL_CORES_PER_REGION * CONSERVATIVE_CAPFACTOR
    
#     # Calcular o offset de tempo sequencial de cada tarefa dentro deste caminho específico
#     current_offset = start_hour
#     for pt in path['tasks']:
#         for h in range(pt['dur']):
#             # Simular a verificação hora a hora na linha temporal em formato ISO
#             # Se o estado atual não tiver o registo, o uso é zero
#             import datetime as dt
#             ts = (dt.datetime.min + dt.timedelta(hours=current_offset + h)).time().isoformat() # Representação genérica de slot
            
#             # Se usar uma string timestamp baseada na submissão original:
#             # ts = (naive_submission + dt.timedelta(hours=current_offset + h)).isoformat()
            
#             current_usage = current_state.get(region, {}).get(ts, 0)
            
#             # Se o nosso pedido somado ao uso atual violar o limite seguro/conservador, rejeitamos o slot
#             if current_usage + pt['cores'] > safe_core_limit:
#                 return False
                
#         current_offset += pt['dur']
        
#     return True

def find_globally_greenest_schedule(REGIONS, tasks_metadata, all_regions_intensities, in_memory_state, 
                                    submission_time, deadline_date, DATA_SIZE_GB, base_ci, source_region=REGIONS[0]):
    """
    Finds the absolute greenest schedule using Pre-flight macro filters, Greedy Core Packing,
    and Look-Ahead Carbon Trend Pruning to eliminate checking suboptimal subsequent slots.

    NOTE: This function is currently NOT called anywhere in the active algorithm paths.
    Kept here for reference / future use.
    """
    global_best_impact = float('inf')
    global_best_manifest = {}

    naive_deadline = deadline_date.replace(tzinfo=None)
    naive_submission = submission_time.replace(tzinfo=None)
    max_window_hours = int((naive_deadline - naive_submission).total_seconds() // 3600)

    # 1. Pre-flight region filtering to keep branching small
    filtered_regions = []
    home_intensities = all_regions_intensities.get(source_region, {})
    home_avg_ci = sum(home_intensities.values()) / len(home_intensities) if home_intensities else base_ci
    total_core_hours = sum(t['dur'] * t['cores'] for t in tasks_metadata)

    for r in REGIONS:
        if r == source_region:
            filtered_regions.append(r)
            continue
        intensities = all_regions_intensities.get(r, {})
        if not intensities:
            continue
        remote_avg_ci = sum(intensities.values()) / len(intensities)
        _, transfer_carbon_penalty = calculate_transfer_penalty(r, DATA_SIZE_GB)
        estimated_operational_savings = (home_avg_ci - remote_avg_ci) * total_core_hours
        
        if estimated_operational_savings > transfer_carbon_penalty:
            filtered_regions.append(r)

    # Pre-calculate suffix core-hours for remaining tasks to make look-ahead fast
    # suffix_remaining_core_hours[i] tells us exactly how many core-hours are left from task i onwards
    task_core_hours = [t['dur'] * t['cores'] for t in tasks_metadata]
    suffix_remaining_core_hours = [sum(task_core_hours[i:]) for i in range(len(task_core_hours))] + [0]

    def dfs(task_idx, current_hour_offset, current_region, accumulated_carbon, current_manifest, current_state):
        nonlocal global_best_impact, global_best_manifest
        
        # All tasks successfully mapped
        if task_idx >= len(tasks_metadata):
            if accumulated_carbon < global_best_impact:
                global_best_impact = accumulated_carbon
                global_best_manifest = copy.deepcopy(current_manifest)
            return

        # ---------------------------------------------------------------------
        # LOOK-AHEAD CARBON TREND PRUNING
        # ---------------------------------------------------------------------
        if current_region:
            intensities = all_regions_intensities.get(current_region, {})
            # Find the absolute best (lowest) carbon intensity remaining in the timeline
            # from our current hour offset up to the very end of the deadline window
            remaining_intensities = [
                v for k, v in intensities.items() 
                if int((k.replace(tzinfo=None) - naive_submission).total_seconds() // 3600) >= current_hour_offset
            ]
            
            if remaining_intensities:
                absolute_best_remaining_ci = min(remaining_intensities)
                remaining_workload = suffix_remaining_core_hours[task_idx]
                
                # Theoretical absolute lowest carbon this branch could possibly achieve
                best_possible_future_impact = accumulated_carbon + (absolute_best_remaining_ci * remaining_workload)
                
                # If even this impossible best-case scenario can't beat our champion, abort!
                if best_possible_future_impact >= global_best_impact:
                    return
        elif accumulated_carbon >= global_best_impact:
            return

        t = tasks_metadata[task_idx]

        # ---------------------------------------------------------------------
        # DECISION 1: Try to schedule and greedily pack tasks at current offset
        # ---------------------------------------------------------------------
        for target_region in filtered_regions:
            intensities = all_regions_intensities.get(target_region, {})

            if current_region and target_region != current_region:
                time_penalty, carbon_penalty = calculate_transfer_penalty(target_region, DATA_SIZE_GB)
            else:
                time_penalty, carbon_penalty = 0, 0

            actual_running_offset = current_hour_offset + time_penalty
            base_t = tasks_metadata[task_idx]

            if actual_running_offset + base_t['dur'] > max_window_hours:
                continue

            t_start = naive_submission + dt.timedelta(hours=actual_running_offset)
            t_lookup = t_start.replace(minute=0, second=0, microsecond=0)

            # Greedy Multi-Task Packing
            packed_tasks = []
            accumulated_cores_for_block = 0
            
            for lookahead_idx in range(task_idx, len(tasks_metadata)):
                cand_t = tasks_metadata[lookahead_idx]
                can_fit_cores = True
                
                for h in range(max(base_t['dur'], cand_t['dur'])):
                    ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                    current_usage = current_state.get(target_region, {}).get(ts, 0)
                    if current_usage + accumulated_cores_for_block + cand_t['cores'] > TOTAL_CORES_PER_REGION:
                        can_fit_cores = False
                        break
                
                if actual_running_offset + max(base_t['dur'], cand_t['dur']) > max_window_hours:
                    can_fit_cores = False

                if can_fit_cores:
                    packed_tasks.append(cand_t)
                    accumulated_cores_for_block += cand_t['cores']
                else:
                    break

            if not packed_tasks:
                continue

            # Commit Packed Batch
            total_batch_carbon = 0
            for pt in packed_tasks:
                pt_ci = intensities.get(min(intensities.keys(), key=lambda d: abs(d.replace(tzinfo=None) - t_lookup)), 400)
                pt_aware_carbon = pt_ci * (pt['dur'] * pt['cores'])
                total_batch_carbon += pt_aware_carbon
                pt_baseline_carbon = base_ci * (pt['dur'] * pt['cores'])

                current_manifest[pt['id']] = {
                    "start": t_lookup.isoformat(), "dur": pt['dur'], "region": target_region, "cores": pt['cores'],
                    "pt_ci": round(base_ci, 2), "target_ci": round(pt_ci, 2),
                    "saved_g": round(pt_baseline_carbon - pt_aware_carbon, 2),
                    "pct": round(((pt_baseline_carbon - pt_aware_carbon) / pt_baseline_carbon * 100), 1) if pt_baseline_carbon > 0 else 0
                }

                for h in range(pt['dur']):
                    ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                    if target_region not in current_state: 
                        current_state[target_region] = {}
                    current_state[target_region][ts] = current_state[target_region].get(ts, 0) + pt['cores']

            # Recurse forward
            dfs(
                task_idx = task_idx + len(packed_tasks), 
                current_hour_offset = actual_running_offset,
                current_region = target_region,
                accumulated_carbon = accumulated_carbon + total_batch_carbon + carbon_penalty, 
                current_manifest = current_manifest, 
                current_state = current_state
            )

            # Backtrack
            for pt in packed_tasks:
                for h in range(pt['dur']):
                    ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                    current_state[target_region][ts] -= pt['cores']
                if pt['id'] in current_manifest:
                    del current_manifest[pt['id']]

        # ---------------------------------------------------------------------
        # DECISION 2: Shift execution timeline baseline forward 1 hour
        # ---------------------------------------------------------------------
        if current_hour_offset + 1 < max_window_hours:
            dfs(task_idx, current_hour_offset + 1, current_region, accumulated_carbon, current_manifest, current_state)

    initial_state = copy.deepcopy(in_memory_state)
    dfs(task_idx=0, current_hour_offset=0, current_region=None, accumulated_carbon=0.0, current_manifest={}, current_state=initial_state)

    winning_region = global_best_manifest.get("T1", {}).get("region", source_region) if global_best_manifest else source_region
    return global_best_impact, global_best_manifest, winning_region

def update_scratchpad(state, region, start_dt, duration, cores):
    """
    Updates the in-memory cluster state to reserve cores for a planned task.
    """
    # Ensure the region exists in the state
    if region not in state:
        state[region] = {}
    
    # Standardize the start time to the top of the hour
    t_lookup = start_dt.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    for h in range(duration):
        # Calculate the timestamp for each hour the task will run
        ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
        
        # Add the cores to whatever is already booked for that hour
        current_usage = state[region].get(ts, 0)
        state[region][ts] = current_usage + cores
        
    return state

def get_estimated_duration(task_id, sla_level):
    if not os.path.exists(HISTORY_FILE):
        return 1.0  # default: 1 hour
    
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    # Get history for task or use DEFAULT key
    data = history.get(task_id, history.get("DEFAULT", [60]))
    
    # Ensure we don't return 0
    val = np.percentile(data, sla_level)
    return max(val, 1) / 60

# def calculate_workflow_savings(plan_file, traces):
#     with open(plan_file, 'r') as f:
#         plan = json.load(f)
    
#     total_baseline_carbon = 0
#     total_aware_carbon = 0
    
#     for task in plan['tasks']:
#         # 1. Baseline: Running in PT at the moment of submission
#         pt_intensity = traces['PT'][plan['submission_time']] 
#         baseline = task['energy_usage'] * pt_intensity
        
#         # 2. Aware: Running in the chosen Region at the chosen Start Time
#         target_region = task['selected_region']
#         target_time = task['selected_start_time']
#         aware_intensity = traces[target_region][target_time]
#         aware = task['energy_usage'] * aware_intensity
        
#         total_baseline_carbon += baseline
#         total_aware_carbon += aware

#     saved = total_baseline_carbon - total_aware_carbon
#     reduction_pct = (saved / total_baseline_carbon) * 100
    
#     return saved, reduction_pct

def calculate_transfer_penalty(target_region, data_size_gb, source_region=REGIONS[0]):
    """
    Calculates penalties based on 400Gbps technology.
    """
    if target_region == source_region:
        return 0, 0 

    # 400 Gbps = 50 GB/s. 
    transfer_speed_gbps = 400 / 8
    transfer_seconds = data_size_gb / transfer_speed_gbps
    transfer_time_h = transfer_seconds / 3600
    
    # Carbon Penalty: Size * Energy * Average Network Intensity
    transfer_carbon_cost = data_size_gb * KWH_PER_GB * 250 
    
    return transfer_time_h, transfer_carbon_cost

def load_carbon_data(region, start_limit, end_limit):
    df_path = os.path.join(DATA_DIR, f"log_{region}.csv")
    if not os.path.exists(df_path): return {}
    
    df = pd.read_csv(df_path)
    df['Datetime (UTC)'] = pd.to_datetime(df['Datetime (UTC)']).dt.tz_localize(None)
    
    # Logic: Align simulation (2026) with historical data (2025)
    target_start = start_limit.replace(year=2025, tzinfo=None)
    target_end = end_limit.replace(year=2025, tzinfo=None)
    
    # Use the full window requested by the planner
    search_start = target_start
    search_end = target_end

    mask = (df['Datetime (UTC)'] >= search_start) & (df['Datetime (UTC)'] <= search_end)
    mask_results = df[mask]
    
    print(f"[DEBUG] {region} Data: Found {len(mask_results)} hours between {search_start} and {search_end} (Aligned to 2025) | Original Request: {start_limit} to {end_limit} (2026)")
    
    # Return mapping back to 2026 so the scheduler keys match
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
        print(f"Error saving cluster state: {e}")

def get_dynamic_critical_path(tasks_metadata, edges, sla_level):
    """
    Calculates the longest path through the workflow using Kahn's algorithm,
    adjusting task durations based on historical data and the requested SLA.
    """
    # 1. Get SLA-adjusted durations for all tasks
    durations = {t['id']: get_estimated_duration(t['id'], sla_level) for t in tasks_metadata}
    
    # If there are no edges, the workflow is just parallel tasks. The critical path is the longest single task.
    if not edges:
        return max(durations.values()) if durations else 1.0

    # 2. Build the graph mapping
    graph = {t['id']: [] for t in tasks_metadata}
    in_degree = {t['id']: 0 for t in tasks_metadata}
    
    for edge in edges:
        # Check if edge is (source, target) or (type, source, target)
        if len(edge) == 2:
            u, v = edge
        elif len(edge) == 3:
            _, u, v = edge # Ignore the first item (the label)
        else:
            print(f"[WARNING] Skipping malformed edge: {edge}")
            continue

        if u in graph and v in in_degree:
            graph[u].append(v)
            in_degree[v] += 1
            
    # 3. Find the critical path
    earliest_start = {t['id']: 0 for t in tasks_metadata}
    queue = [t['id'] for t in tasks_metadata if in_degree[t['id']] == 0]
    
    while queue:
        u = queue.pop(0)
        for v in graph[u]:
            earliest_start[v] = max(earliest_start[v], earliest_start[u] + durations[u])
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                
    # The critical path length is the maximum of (start_time + duration) across all nodes
    cp_length = max(earliest_start[n] + durations[n] for n in earliest_start)
    return cp_length

def plan_baseline_A(run_id, submission, deadline, tasks, edges, in_memory_state):
    """Baseline A: Immediate Schedule (Earliest Start across all regions)"""
    print(f"[BASELINE A] Planning {run_id} for immediate execution...")
    plan = {}
    
    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(submission)
    dl_dt = deadline if isinstance(deadline, dt.datetime) else dt.datetime.fromisoformat(deadline)
    
    # Ensure they are timezone-naive for comparison with our scratchpad
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt = dl_dt.replace(tzinfo=None)

    best_region = None

    for task in tasks:
        best_start = None
        best_region = REGIONS[0]
        
        for reg in REGIONS:
            earliest_for_reg = find_earliest_slot(reg, task['cores'], in_memory_state)
            if best_start is None or earliest_for_reg < best_start:
                best_start = earliest_for_reg
                best_region = reg
        
        plan[task['id']] = {
            "region": best_region, "start": best_start.isoformat(),
            "dur": task['dur'], "cores": task['cores']
        }
        in_memory_state = update_scratchpad(in_memory_state, best_region, best_start, task['dur'], task['cores'])

    save_cluster_state(in_memory_state)

    # Visualization
    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in REGIONS}
    stats = calculate_standardized_stats(plan, all_traces, sub_dt, len(tasks))
    plot_optimization_window_heatmap(
        plan, 
        all_traces, 
        sub_dt, 
        dl_dt, 
        plot_name=f"baseline_A_{run_id}",
        stats=stats
    )
    plan_path = get_safe_plan_path(run_id)
    with open(plan_path, 'w') as f:
        json.dump(plan, f)
    
    print(f"[BASELINE A] Plan saved to {plan_path}")
    return plan

def plan_baseline_B(run_id, submission, deadline, tasks, edges, in_memory_state):
    """
    Baseline B: Local-First (Portugal). 
    Schedule in Portugal if deadline allows; otherwise, migrate to the 
    region with the earliest possible start time.
    """
    print(f"[BASELINE B] Planning {run_id}...")
    plan = {}
    local_reg = REGIONS[0]  
    
    # 1. Defensive Date Parsing
    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(str(submission))
    dl_dt = deadline if isinstance(deadline, dt.datetime) else dt.datetime.fromisoformat(str(deadline))
    
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt = dl_dt.replace(tzinfo=None)

    # 2. Process each task individually
    for task in tasks:
        # Check when this specific task can start in Portugal
        start_portugal = find_earliest_slot(local_reg, task['cores'], in_memory_state)
        finish_portugal = start_portugal + dt.timedelta(hours=task['dur'])

        # If it can finish in Portugal before the deadline, stay there.
        if finish_portugal <= dl_dt:
            chosen_reg = local_reg
            chosen_start = start_portugal
        else:
            # Portugal is too busy! Finding the region that can start it the SOONEST.
            print(f"[BASELINE B] Task {task['id']} missed Portugal deadline. Migrating...")
            
            best_start = start_portugal
            chosen_reg = local_reg
            
            for reg in REGIONS:
                earliest_elsewhere = find_earliest_slot(reg, task['cores'], in_memory_state)
                if earliest_elsewhere < best_start:
                    best_start = earliest_elsewhere
                    chosen_reg = reg
            
            chosen_start = best_start

        # 3. Save to plan
        plan[task['id']] = {
            "region": chosen_reg,
            "start": chosen_start.isoformat(),
            "dur": task['dur'],
            "cores": task['cores']
        }

        in_memory_state = update_scratchpad(in_memory_state, chosen_reg, chosen_start, task['dur'], task['cores'])

    # 5. Commit decisions to disk and visualize
    save_cluster_state(in_memory_state)
    
    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in REGIONS}
    stats = calculate_standardized_stats(plan, all_traces, sub_dt, len(tasks))
    
    plot_optimization_window_heatmap(plan, all_traces, sub_dt, dl_dt, plot_name=f"baseline_B_{run_id}", stats=stats)

    plan_path = get_safe_plan_path(run_id)
    with open(plan_path, 'w') as f:
        json.dump(plan, f)

    return plan



def plan_baseline_atomic(run_id, submission, deadline, tasks, edges, in_memory_state):
    """
    Baseline C: Black-Box Atomic Scheduler (Thread-Safe and Cluster-Capacity Aware).
    Treats the workflow as a single un-splittable block. Verifies that cluster 
    resources are continuously available for the full duration before booking.
    """
    print(f"[BASELINE ATOMIC] Planning {run_id} with full capacity verification...")
    plan = {}
    
    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(str(submission))
    dl_dt = deadline if isinstance(deadline, dt.datetime) else dt.datetime.fromisoformat(str(deadline))
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt = dl_dt.replace(tzinfo=None)

    total_duration = sum(t['dur'] for t in tasks)
    max_cores_needed = max(t['cores'] for t in tasks)
    
    best_start = None
    best_region = None
    lowest_average_carbon = float('inf')

    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in REGIONS}

    for reg in REGIONS:
        # Find the absolute earliest the cluster can accept the FIRST task
        earliest_possible_start = find_earliest_slot(reg, max_cores_needed, in_memory_state)
        available_hours = int((dl_dt - earliest_possible_start).total_seconds() / 3600)
        
        for offset in range(available_hours - total_duration + 1):
            candidate_start = earliest_possible_start + dt.timedelta(hours=offset)
            
            block_is_valid = True
            current_block_carbon = 0.0
            
            for h in range(total_duration):
                hour_obj = candidate_start + dt.timedelta(hours=h)
                hour_str = hour_obj.isoformat()
                
                # Check how many cores are already used in this region at this hour
                cores_used_already = in_memory_state.get(reg, {}).get(hour_str, 0)
                
                # If adding our needed cores exceeds cluster capacity, this entire block is invalid
                if cores_used_already + max_cores_needed > TOTAL_CORES_PER_REGION:
                    block_is_valid = False
                    break # Stop checking this candidate slot
                
                # If space is valid, accumulate the carbon metric for this hour
                intensity = all_traces.get(reg, {}).get(hour_obj, 400) 
                current_block_carbon += (max_cores_needed * 0.2) * intensity
            
            # If the full block fits and saves more carbon than our best find, keep it
            if block_is_valid and current_block_carbon < lowest_average_carbon:
                lowest_average_carbon = current_block_carbon
                best_start = candidate_start
                best_region = reg

    # 4. Fallback: If no continuous slot fits safely, force immediate execution at earliest slot
    if best_start is None:
        print("[WARNING] No valid continuous block fits capacity limits. Forcing earliest execution.")
        best_region = REGIONS[0]
        best_start = find_earliest_slot(best_region, max_cores_needed, in_memory_state)

    # 5. Build the final plan mapping ALL tasks back-to-back
    current_task_start = best_start
    for task in tasks:
        plan[task['id']] = {
            "region": best_region,
            "start": current_task_start.isoformat(),
            "dur": task['dur'],
            "cores": task['cores']
        }
        
        # Sequentially book the state for each task hour
        in_memory_state = update_scratchpad(
            in_memory_state, 
            best_region, 
            current_task_start, 
            task['dur'], 
            task['cores']
        )
        current_task_start += dt.timedelta(hours=task['dur'])

    # 6. Post-processing, stats, saving, and plotting
    save_cluster_state(in_memory_state)
    stats = calculate_standardized_stats(plan, all_traces, sub_dt, len(tasks))
    
    plot_optimization_window_heatmap(
        plan, all_traces, sub_dt, dl_dt, 
        plot_name=f"baseline_atomic_{run_id}", stats=stats
    )

    plan_path = get_safe_plan_path(run_id)
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=4)
        
    return plan

# def preprocess_carbon_to_rle_buckets(all_regions_intensities, REGIONS, max_window_hours):
#     """
#     Dynamically normalizes carbon thresholds for each region based on local min/max 
#     intensity bounds, then applies Run-Length Encoding (RLE) to capture clean intervals.
#     """
#     rle_index = {r: {"DARK_GREEN": [], "LIGHT_GREEN": [], "YELLOW": [], "DIRTY_RED": []} for r in REGIONS}

#     for r in REGIONS:
#         intensities = all_regions_intensities.get(r, {})
#         if not intensities:
#             continue

#         sorted_ts = sorted(intensities.keys())
#         timeline = [intensities[ts] for ts in sorted_ts]

#         # Extract active values inside our planning horizon to establish the normalization baseline
#         active_values = timeline[:max_window_hours] if len(timeline) >= max_window_hours else timeline
#         if not active_values:
#             continue

#         current_bucket = None
#         current_start = 0
#         current_length = 0

#         for hour_offset in range(max_window_hours):
#             if hour_offset < len(timeline):
#                 ci_val = timeline[hour_offset]
#             else:
#                 ci_val = timeline[-1] if timeline else 400

#             if ci_val <= 150:
#                 bucket_type = "DARK_GREEN"
#             elif ci_val <= 300:
#                 bucket_type = "LIGHT_GREEN"
#             elif ci_val <= 400:
#                 bucket_type = "YELLOW"
#             else:
#                 bucket_type = "DIRTY_RED"

#             if current_bucket is None:
#                 current_bucket = bucket_type
#                 current_start = hour_offset
#                 current_length = 1
#             elif bucket_type == current_bucket:
#                 current_length += 1
#             else:
#                 rle_index[r][current_bucket].append({
#                     "start": current_start,
#                     "length": current_length
#                 })
#                 current_bucket = bucket_type
#                 current_start = hour_offset
#                 current_length = 1

#         if current_bucket is not None:
#             rle_index[r][current_bucket].append({
#                 "start": current_start,
#                 "length": current_length
#             })

#     return rle_index

def preprocess_carbon_to_rle_buckets(all_regions_intensities, REGIONS, max_window_hours):
    # --- GLOBAL PERCENTILE THRESHOLDS ---
    # Collect all active intensity values across every region into one pool,
    # then derive thresholds from that combined distribution. This ensures
    # bucket labels are absolute (a PL hour only gets DARK_GREEN if it's
    # genuinely clean, not just clean relative to PL's own dirty grid).
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

    print(f"[RLE] Global thresholds ({len(all_active_values)} hours across {len(REGIONS)} regions) | "
          f"DARK_GREEN ≤{p25:.1f} | LIGHT_GREEN ≤{p50:.1f} | YELLOW ≤{p75:.1f} | DIRTY_RED >{p75:.1f}")

    def classify(ci_val):
        if ci_val <= p25:
            return "DARK_GREEN"
        elif ci_val <= p50:
            return "LIGHT_GREEN"
        elif ci_val <= p75:
            return "YELLOW"
        else:
            return "DIRTY_RED"

    # --- PER-REGION RLE ENCODING (unchanged) ---
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
            print(f"[RLE]   {r} {bucket:<12} → {len(intervals):>2} interval(s), {total_hours:>3}h total")

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
        # If the parent was processed in a different region, track its data payload size
        if parent_region and parent_region != current_eval_region:
            parent_task_static = task_map.get(p_id, {})
            total_cross_region_gb += parent_task_static.get('output_size_gb', 0)
                
    return total_cross_region_gb

# def commit_path_to_schedule_via_rle_intervals(final_manifest, current_state, path, region, start_hour, naive_submission, base_ci, all_regions_intensities):
#     """
#     Lightweight commit tracker that calculates carbon metrics on-the-fly 
#     only for the final assigned execution window.
#     """
#     current_offset = start_hour
#     region_intensities = all_regions_intensities.get(region, {})
#     sorted_ts = sorted(region_intensities.keys())
#     timeline = [region_intensities[ts] for ts in sorted_ts]


#     for pt in path['tasks']:
#         # Duplication shield to protect shared execution nodes
#         if pt['id'] in final_manifest:
#             # If already scheduled by a previous overlapping path, inherit its end offset
#             p_start_dt = dt.datetime.fromisoformat(final_manifest[pt['id']]['start'])
#             current_offset = int((p_start_dt - naive_submission).total_seconds() // 3600) + final_manifest[pt['id']]['dur']
#             continue
            
#         # --- FIXED: INFRASTRUCTURE CAPACITY AND GAP GUARD LOOP ---
#         while True:
#             t_start = naive_submission + dt.timedelta(hours=current_offset)
#             t_lookup = t_start.replace(minute=0, second=0, microsecond=0)
            
#             # Guard 1: Verify we aren't spilling past our planning deadline
#             if current_offset + pt['dur'] > max_window_hours:
#                 print(f"[CRITICAL WARNING] Task {pt['id']} pushed out of deadline bounds due to capacity saturation!")
#                 break

#             # Guard 2: Scan across the task's execution hours to check for a core overflow
#             capacity_overflow = False
#             for h in range(pt['dur']):
#                 ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
#                 cores_already_used = current_state.get(region, {}).get(ts, 0)
                
#                 # If adding this task's cores breaks the cluster limit, we have a collision
#                 if cores_already_used + pt['cores'] > TOTAL_CORES_PER_REGION:
#                     capacity_overflow = True
#                     break
            
#             # If the cluster has enough physical room, break out of the while loop and schedule it!
#             if not capacity_overflow:
#                 break
                
#             # If the slot is congested, delay this task by 1 hour and test again
#             current_offset += 1

#         # Re-calculate correct dates based on our final un-congested offset
#         t_start = naive_submission + dt.timedelta(hours=current_offset)
#         t_lookup = t_start.replace(minute=0, second=0, microsecond=0)

#         # Calculate task carbon intensity dynamically for its running duration
#         task_ci_sum = 0
#         for h in range(pt['dur']):
#             h_idx = current_offset + h
#             task_ci_sum += timeline[h_idx] if h_idx < len(timeline) else (timeline[-1] if timeline else 400)
            
#         pt_ci = task_ci_sum / pt['dur'] if pt['dur'] > 0 else 400
        
#         pt_aware_carbon = pt_ci * (pt['dur'] * pt['cores'])
#         pt_baseline_carbon = base_ci * (pt['dur'] * pt['cores'])
        
#         # Commit cleanly to our workflow manifest blueprint
#         final_manifest[pt['id']] = {
#             "start": t_lookup.isoformat(), 
#             "dur": pt['dur'], 
#             "region": region, 
#             "cores": pt['cores'],
#             "pt_ci": round(base_ci, 2), 
#             "target_ci": round(pt_ci, 2),
#             "saved_g": round(pt_baseline_carbon - pt_aware_carbon, 2),
#             "pct": round(((pt_baseline_carbon - pt_aware_carbon) / pt_baseline_carbon * 100), 1) if pt_baseline_carbon > 0 else 0
#         }
        
#         # Deduct available resources from cluster ledger (Locking the keys)
#         if region not in current_state:
#             current_state[region] = {}
#         for h in range(pt['dur']):
#             ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
#             current_state[region][ts] = current_state[region].get(ts, 0) + pt['cores']
            
#         # Advance the offset past this task's block so the next sibling task respects the dependency gap
#         current_offset += pt['dur']

def find_greenest_schedule_via_rle(REGIONS, tasks_metadata, all_regions_intensities, rle_index,
                                    submission_time, max_window_hours, base_ci):
    """
    FIXED: Resilient Trial-Validated Bucket Scheduler with Network Transfer Penalties.
    Simulates remaining workflow paths while evaluating storage cross-region network overhead.
    """
    # 1. Build an index map of tasks for parent/dependency lookups
    task_map = {t['id']: t for t in tasks_metadata}
    
    # Calculate downstream critical path depth for each task to prioritize important nodes
    def get_task_priority(task_id, memo=None):
        if memo is None: memo = {}
        if task_id in memo:
            return memo[task_id]
        t = task_map[task_id]
        dependent_tasks = [other for other in tasks_metadata if task_id in other.get('depends_on', [])]
        if not dependent_tasks:
            memo[task_id] = t['dur']
            return t['dur']
        memo[task_id] = t['dur'] + max(get_task_priority(dep['id'], memo) for dep in dependent_tasks)
        return memo[task_id]

    # Order tasks globally by execution priority (Critical path sequencing)
    sorted_tasks = sorted(tasks_metadata, key=lambda x: get_task_priority(x['id']), reverse=True)

    # 2. Synchronize multi-run cluster state
    current_state = load_cluster_state()
    for r in REGIONS:
        if r not in current_state: current_state[r] = {}

    final_manifest = {}
    task_end_times = {}           # Tracks raw integer hour index when each task finishes
    task_scheduled_regions = {}   # Tracks region identity where each task was finalized
    naive_submission = submission_time.replace(tzinfo=None)

    # Helper to pull regional carbon intensity safely
    def get_carbon_at_hour(reg, hour_idx):
        intensities = all_regions_intensities.get(reg, {})
        sorted_ts = sorted(intensities.keys())
        if hour_idx < len(sorted_ts):
            return intensities[sorted_ts[hour_idx]]
        return intensities[sorted_ts[-1]] if sorted_ts else 400

    # Build target search pools sorted by absolute carbon footprint
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

    clean_bucket_pool = build_ranked_pool(["DARK_GREEN"])
    print(clean_bucket_pool)
    # 3. Schedule Task-by-Task
    for idx, pt in enumerate(sorted_tasks):
        parents = pt.get('depends_on', [])
        latest_allowed_start = max_window_hours - pt['dur']
        
        best_slot = None
        best_region = None
        best_transfer_carbon = 0

        # Internal helper to sweep a specific bucket pool with look-ahead trial validation
        def evaluate_pool(bucket_pool):
            nonlocal best_slot, best_region, best_transfer_carbon
            for bucket in bucket_pool:
                region = bucket['region']
                
                # --- CALCULATE STORAGE TRANSFER PENALTY OVERHEAD ---
                earliest_start = 0
                current_node_transfer_carbon = 0
                
                if parents:
                    parent_times = []
                    # Get network transit size requirements from parent metrics
                    task_data_size = get_incoming_transfer_footprint(pt, task_map, task_scheduled_regions, region)
                    print(f"[DEBUG] task_size: {task_data_size} GB for task {pt['id']} in region {region}")
                    for p_id in parents:
                        p_end_hour = task_end_times.get(p_id, 0)
                        p_region = task_scheduled_regions.get(p_id, region)
                        
                        # Process cross-regional penalties via external framework utility
                        t_hours, t_carbon = calculate_transfer_penalty(
                            target_region=region,
                            source_region=p_region,
                            data_size_gb=task_data_size
                        )
                        print(f"[DEBUG] transfer_carbon: {t_carbon} for task {pt['id']} in region {region}")
                        
                        # Pad availability with raw integer time delay caps
                        parent_times.append(p_end_hour + math.ceil(t_hours))
                        current_node_transfer_carbon += t_carbon
                    
                    earliest_start = max(parent_times) if parent_times else 0

                workflow_remaining_dur = get_task_priority(pt['id']) 
                search_start = max(bucket['start'], earliest_start)
                bucket_end = bucket['start'] + bucket['length']
                search_end = min(bucket_end + workflow_remaining_dur, latest_allowed_start)
                
                if search_start > search_end:
                    continue

                for start_hour in range(search_start, search_end + 1):
                    # --- SIMULATION TRIAL CAPACITY CHECK ---
                    capacity_violated = False
                    for h in range(pt['dur']):
                        t_slot = (naive_submission + dt.timedelta(hours=start_hour + h)).replace(minute=0, second=0, microsecond=0).isoformat()
                        cores_used = current_state[region].get(t_slot, 0)
                        if cores_used + pt['cores'] > TOTAL_CORES_PER_REGION:
                            capacity_violated = True
                            break

                    if capacity_violated:
                        continue

                    # --- DOWNSTREAM VALIDATION LOOK-AHEAD SIMULATION ---
                    trial_state = copy.deepcopy(current_state)
                    trial_end_times = copy.deepcopy(task_end_times)
                    trial_regions = copy.deepcopy(task_scheduled_regions)
                    
                    # Project current node placement cleanly into simulation sandbox
                    trial_end_times[pt['id']] = start_hour + pt['dur']
                    trial_regions[pt['id']] = region
                    
                    for h in range(pt['dur']):
                        t_slot = (naive_submission + dt.timedelta(hours=start_hour + h)).replace(minute=0, second=0, microsecond=0).isoformat()
                        trial_state[region][t_slot] = trial_state[region].get(t_slot, 0) + pt['cores']

                    lookahead_success = True
                    
                    for future_pt in sorted_tasks[idx + 1:]:
                        f_parents = future_pt.get('depends_on', [])
                        if not f_parents:
                            continue
                        
                        child_fit_found = False
                        for r_check in REGIONS:
                            f_parent_times = []
                            f_data_size = get_incoming_transfer_footprint(future_pt, task_map, trial_regions, r_check)
                            
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
                            
                            for f_slot in range(f_earliest, f_latest + 1):
                                f_cap_violated = False
                                for fh in range(future_pt['dur']):
                                    ft_slot = (naive_submission + dt.timedelta(hours=f_slot + fh)).replace(minute=0, second=0, microsecond=0).isoformat()
                                    if trial_state[r_check].get(ft_slot, 0) + future_pt['cores'] > TOTAL_CORES_PER_REGION:
                                        f_cap_violated = True
                                        break
                                if not f_cap_violated:
                                    child_fit_found = True
                                    trial_end_times[future_pt['id']] = f_slot + future_pt['dur']
                                    trial_regions[future_pt['id']] = r_check
                                    for fh in range(future_pt['dur']):
                                        ft_slot = (naive_submission + dt.timedelta(hours=f_slot + fh)).replace(minute=0, second=0, microsecond=0).isoformat()
                                        trial_state[r_check][ft_slot] = trial_state[r_check].get(ft_slot, 0) + future_pt['cores']
                                    break
                            if child_fit_found:
                                break
                        
                        if not child_fit_found:
                            lookahead_success = False
                            break

                    if lookahead_success:
                        best_slot = start_hour
                        best_region = region
                        best_transfer_carbon = current_node_transfer_carbon
                        return 

        # Process search sequences
        evaluate_pool(clean_bucket_pool)
        if best_slot is None:
            print(f"[FALLBACK] Task {pt['id']} triggered a look-ahead bottleneck. Expanding search boundaries...")
            fallback_bucket_pool = build_ranked_pool(["LIGHT_GREEN", "YELLOW", "DIRTY_RED"])
            evaluate_pool(fallback_bucket_pool)

        # 4. Final Commit and Ledger Updates
        if best_region is not None and best_slot is not None:
            t_lookup = (naive_submission + dt.timedelta(hours=best_slot)).replace(minute=0, second=0, microsecond=0)
            
            task_ci_sum = sum(get_carbon_at_hour(best_region, best_slot + h) for h in range(pt['dur']))
            pt_ci = task_ci_sum / pt['dur']
            
            # Incorporate transit carbon cost into total manifest accountability
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
            
            for h in range(pt['dur']):
                ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                current_state[best_region][ts] = current_state[best_region].get(ts, 0) + pt['cores']
            
            task_end_times[pt['id']] = best_slot + pt['dur']
            task_scheduled_regions[pt['id']] = best_region
        else:
            print(f"[FATAL FAILURE] Task {pt['id']} cannot fit within cluster limits.")
            safe_fallback_start = max((task_end_times.get(p_id, 0) for p_id in parents), default=0)
            task_end_times[pt['id']] = safe_fallback_start + pt['dur']
            task_scheduled_regions[pt['id']] = REGIONS[0]

    save_cluster_state(current_state)
    return final_manifest


def plan_workflow_alg1(run_id, submission_time, deadline_date, tasks_metadata):
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
    
    intensities = load_carbon_data(REGIONS[0], submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    
    # Get baseline intensity (current moment)
    if sub_naive in intensities:
        base = intensities[sub_naive]
    elif intensities:
        base = intensities[sorted(intensities.keys())[0]]
    else:
        base = 400

    # Set the baseline as the current "best"
    best_avg = base
    best_start = submission_time
    best_reg = REGIONS[0]

    # Calculate search window in hours
    if submission_time.tzinfo is None:
        submission_time = submission_time.replace(tzinfo=dt.timezone.utc)

    # Ensure deadline_date is aware
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
            
            # Calculate average carbon for the workflow duration
            window_vals = [intensities.get(start_cand + dt.timedelta(hours=h), 999) for h in range(total_duration)]
            avg_carbon = sum(window_vals) / len(window_vals)

            # Capacity Check
            if has_capacity(region, start_cand, total_duration, total_cores):
                if avg_carbon < best_avg:
                    best_avg = avg_carbon
                    best_start = start_cand
                    best_reg = region

    # Create the sequence plan
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

    # 2. Floor the submission time to the hour
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    
    pt_intensities = load_carbon_data("PT", submission_time, deadline_date)
    if sub_naive in pt_intensities:
        pt_start_intensity = pt_intensities[sub_naive]
    elif pt_intensities:
        available_times = sorted(pt_intensities.keys())
        pt_start_intensity = pt_intensities[available_times[0]]
        print(f"[DEBUG] Exact time {sub_naive} not in PT logs. Using closest: {available_times[0]}")
    else:
        pt_start_intensity = 400 # Realistic fallback for PT if logs are empty
        print("[DEBUG] PT logs empty for this window. Using global fallback.")

    local_execution_carbon = pt_start_intensity * (total_duration * total_cores)
    carbon_saved = local_execution_carbon - best_avg * (total_duration * total_cores)
    # Save and lock resources immediately (Exclusivity)
    print(f"\n[ALGORITHM 1 REPORT]")
    print(f"  - Local PT Start Intensity: {pt_start_intensity} g/kWh")
    print(f"  - Winner {best_reg} Start Intensity: {best_avg} g/kWh")
    print(f"  - Net Savings: {round(carbon_saved, 2)} gCO2eq")

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

def plan_workflow_alg2(run_id, submission_time, deadline_date, tasks_metadata, data_size_gb, in_memory_state=None):
    """
    IMPLEMENTS ALG 2:
    - Finds the best window for the WHOLE workflow in a single region.
    - Tasks are executed strictly sequentially back-to-back without gaps.
    - Incorporates data transfer time penalties and network carbon taxes.
    - Uses high-fidelity closest-match lookups to prevent carbon calculation failures.
    """
    print(f"[ALG 2] Planning with high-fidelity tracking for {data_size_gb}GB | Workload: {run_id}")
    plan_path = get_safe_plan_path(run_id)
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    # 1. Setup and Timezone Normalization
    total_duration = sum(t['dur'] for t in tasks_metadata)
   
    if isinstance(submission_time, str):
        submission_time = dt.datetime.fromisoformat(submission_time)
    if submission_time.tzinfo is None:
        submission_time = submission_time.replace(tzinfo=dt.timezone.utc)
        
    if isinstance(deadline_date, str): 
        deadline_date = dt.datetime.fromisoformat(deadline_date)
    deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)

    # 2. Get baseline carbon details from Portugal (PT)
    pt_traces = load_carbon_data("PT", submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    pt_base_ci = pt_traces.get(sub_naive, pt_traces[min(pt_traces.keys(), key=lambda d: abs(d-sub_naive))] if pt_traces else 450)

    best_total_impact = float('inf')
    best_reg, best_start, best_manifest = None, None, {}

    if in_memory_state is None:
        in_memory_state = load_cluster_state()

    # 3. Optimization Loop (Whole-Workflow Placement)
    for region in REGIONS:
        # Calculate data transfer overheads for this candidate region
        time_penalty, carbon_penalty = calculate_transfer_penalty(region, data_size_gb, REGIONS[0])
        eff_deadline = deadline_date - dt.timedelta(hours=time_penalty)
        print(f"\n[DEBUG] Evaluating {region} | Transfer Time Penalty: {round(time_penalty, 2)}h | Carbon Penalty: {round(carbon_penalty, 2)}g | Effective Deadline: {eff_deadline} | Total Workflow Duration: {total_duration}h")
        intensities = load_carbon_data(region, submission_time, eff_deadline)
        if not intensities: 
            continue

        window_hours = int((eff_deadline - submission_time).total_seconds() // 3600)

        print(f"[DEBUG] Available hours in {region} for this workflow (after transfer penalty): {window_hours}h")
        # Scan available start windows for the entire block
        for offset in range(max(1, int(window_hours - total_duration + 1))):
            start_cand = submission_time + dt.timedelta(hours=offset + time_penalty)
            print(f"[DEBUG] Testing {region} with candidate start {start_cand} (offset {offset}h + transfer penalty {round(time_penalty, 2)}h)")
            sim_carbon, can_fit, cand_manifest = 0, True, {}
            temp_search_state = copy.deepcopy(in_memory_state)
            
            # Tasks run strictly back-to-back in a sequential chain
            current_task_time = start_cand

            for t in tasks_metadata:
                t_lookup = current_task_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
                
                # Check real capacity availability across the task's duration
                can_reserve = True
                for h in range(t['dur']):
                    ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                    usage = temp_search_state.get(region, {}).get(ts, 0)
                    if usage + t['cores'] > TOTAL_CORES_PER_REGION:
                        can_reserve = False
                        break
                
                if not can_reserve:
                    can_fit = False
                    break

                # High-fidelity closest key lookup matching Alg 3's logic
                t_ci = intensities.get(min(intensities.keys(), key=lambda d: abs(d - t_lookup)), 400)
                t_aware_carbon = t_ci * (t['dur'] * t['cores'])
                sim_carbon += t_aware_carbon
                t_baseline_carbon = pt_base_ci * (t['dur'] * t['cores'])

                # Track precise structured parameters for metrics/visualizations
                cand_manifest[t['id']] = {
                    "start": t_lookup.isoformat(), 
                    "dur": t['dur'], 
                    "region": region, 
                    "cores": t['cores'],
                    "pt_ci": round(pt_base_ci, 2),
                    "target_ci": round(t_ci, 2),
                    "saved_g": round(t_baseline_carbon - t_aware_carbon, 2), 
                    "pct": round(((t_baseline_carbon - t_aware_carbon) / t_baseline_carbon * 100), 1) if t_baseline_carbon > 0 else 0
                }

                # Reserve resources inside the simulation timeline block
                for h in range(t['dur']):
                    ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                    if region not in temp_search_state: 
                        temp_search_state[region] = {}
                    temp_search_state[region][ts] = temp_search_state[region].get(ts, 0) + t['cores']

                # Advance clock precisely by task duration to remain back-to-back sequential
                current_task_time += dt.timedelta(hours=t['dur'])

            # Total regional impact includes network data tax
            total_impact = sim_carbon + carbon_penalty
            if can_fit and total_impact < best_total_impact:
                best_total_impact = total_impact
                best_start = start_cand
                best_reg = region
                best_manifest = cand_manifest

    # 4. Final Output, Graphics, and State Saving Pipeline
    if best_reg:
        print(f"\n[ALGORITHM 2 REPORT] Workflow: {run_id}")
        print(f"{'Task ID':<15} | {'Region':<8} | {'Saved (g)':<10} | {'Reduction %'}")
        print("-" * 50)
        
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
            print(f"{tid:<15} | {d['region']:<8} | {d['saved_g']:<10} | {d['pct']}%")
            actual_sum_saved_g += d['saved_g']
            actual_sum_baseline_g += (d['saved_g'] / (d['pct']/100)) if d['pct'] > 0 else 0
        
        total_reduction_pct = (actual_sum_saved_g / actual_sum_baseline_g * 100) if actual_sum_baseline_g > 0 else 0
        
        print(f"--- TOTAL SAVED (Execution): {round(actual_sum_saved_g, 2)}g ({round(total_reduction_pct, 1)}%) ---\n")
        print(f"--- TOTAL if ran in PT: {round(actual_sum_baseline_g, 2)}g ---\n")
        
        # Atomically secure physical resource tracks on the cluster state file
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

def plan_workflow_alg3(run_id, submission_time, deadline_date, tasks_metadata, edges, sla_level, in_memory_state=None):
    """IMPLEMENTS ALG 3:
    - Spatio-Temporal Shifting with Task-Level Granularity.
    - Evaluates each task independently, allowing them to be scheduled in different regions and times within the same workflow.
    - Uses the dynamic critical path to understand the workflow's temporal constraints and prioritize scheduling decisions.
    - Aims to minimize total carbon impact while respecting the workflow's deadline and SLA requirements.
    """
    plan_path = get_safe_plan_path(run_id)
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    # 1. Setup inicial e Normalização
    cp_duration_hours = get_dynamic_critical_path(tasks_metadata, edges, sla_level)
    if isinstance(submission_time, str):
        submission_time = dt.datetime.fromisoformat(submission_time)

    if submission_time.tzinfo is None: submission_time = submission_time.replace(tzinfo=dt.timezone.utc)
    if isinstance(deadline_date, str): deadline_date = dt.datetime.fromisoformat(deadline_date)
    deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)
    max_window_hours = int((deadline_date - submission_time).total_seconds() // 3600)

    # 2. Baseline de Portugal
    pt_traces = load_carbon_data("PT", submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    pt_base_ci = pt_traces.get(sub_naive, pt_traces[min(pt_traces.keys(), key=lambda d: abs(d-sub_naive))] if pt_traces else 450)
    
    best_reg, best_start, best_manifest, best_duration = None, None, {}, int(cp_duration_hours)

    if in_memory_state is None:
        in_memory_state = load_cluster_state()

    # 3. Loop de Otimização (Spatio-Temporal Shifting Iterativo)
    all_regions_intensities = {}
    for r in REGIONS:
        time_penalty, _ = calculate_transfer_penalty(r, DATA_SIZE_GB)
        eff_deadline = deadline_date - dt.timedelta(hours=time_penalty)
        intensities = load_carbon_data(r, submission_time, eff_deadline)
        if intensities:
            all_regions_intensities[r] = intensities

    # Chamada única à nova estratégia heurística e polinomial
    rle_index = preprocess_carbon_to_rle_buckets(all_regions_intensities, REGIONS, max_window_hours)

    best_manifest = find_greenest_schedule_via_rle(
        REGIONS=REGIONS,
        tasks_metadata=tasks_metadata,
        all_regions_intensities=all_regions_intensities,
        rle_index=rle_index,
        submission_time=submission_time,
        max_window_hours=max_window_hours,
        base_ci=pt_base_ci
    )

    # Extrair dinamicamente a região vencedora dominante para manter a compatibilidade com os teus prints
    best_reg = None
    if best_manifest:
        # Pega na região definida para a primeira tarefa ativa como a principal
        first_task_id = list(best_manifest.keys())[0]
        best_reg = best_manifest[first_task_id]['region']

    # 4. Output, Heatmap e Report Histórico
    if best_reg:
        print(f"\n[ALGORITHM 3 REPORT] Workflow: {run_id}")
        print(f"{'Task ID':<15} | {'Region':<8} | {'Saved (g)':<10} | {'Reduction %'}")
        print("-" * 50)
        
        all_regions_window_data = {}
        for reg in REGIONS:
            all_regions_window_data[reg] = load_carbon_data(reg, submission_time, deadline_date)

        sub_naive = submission_time.replace(tzinfo=None)

        stats = calculate_standardized_stats(best_manifest, all_regions_window_data, sub_naive, len(tasks_metadata))
        
        plot_optimization_window_heatmap(
            best_manifest=best_manifest,
            all_regions_data=all_regions_window_data,
            submission_time=submission_time,
            deadline=deadline_date,
            plot_name=f"alg3_{run_id}",
            stats=stats
        )

        actual_sum_saved_g = 0
        actual_sum_baseline_g = 0
        
        for tid, d in best_manifest.items():
            print(f"{tid:<15} | {d['region']:<8} | {d['saved_g']:<10} | {d['pct']}%")
            actual_sum_saved_g += d['saved_g']
            actual_sum_baseline_g += (d['saved_g'] / (d['pct']/100)) if d['pct'] > 0 else 0
        
        total_reduction_pct = (actual_sum_saved_g / actual_sum_baseline_g * 100) if actual_sum_baseline_g > 0 else 0
        print(f"--- TOTAL SAVED: {round(actual_sum_saved_g, 2)}g ({round(total_reduction_pct, 1)}%) ---\n")
        
        # Gravar apenas o manifesto de planeamento final do Airflow (o ficheiro JSON do plano)
        with open(plan_path, 'w') as f: 
            json.dump(best_manifest, f, indent=4)
            
        return best_manifest

    return {}
    

def plan_batch_alg4(dag_runs_metadata):
    """
    Explores heuristics like 'Longest Job First' to reduce fragmentation.
    Now includes statistics calculation for Energy, Carbon, and Success Rate.
    """
    HEURISTIC = "other" 
    global_scratchpad = load_cluster_state()
    
    # 1. Sort the queue based on Heuristic
    if HEURISTIC == "EDF":
        print("[BATCH PLANNER] Using Earliest Deadline First (EDF) heuristic.")
        queue = sorted(
            dag_runs_metadata, 
            key=lambda x: dt.datetime.fromisoformat(x['deadline']) if isinstance(x['deadline'], str) else x['deadline']
        )
    else:
        print("[BATCH PLANNER] Using Longest Workflow First (LWF) heuristic.")
        queue = sorted(
            dag_runs_metadata, 
            key=lambda x: sum(t['dur'] for t in x['tasks']) * max(t['cores'] for t in x['tasks']),
            reverse=True
        )

    # 2. Determine batch boundaries
    all_subs = []
    all_dls = []
    now = dt.datetime.now().replace(minute=0, second=0, microsecond=0, tzinfo=None)
    total_requested_tasks = 0
    for wf in dag_runs_metadata:
        s, d = wf['submission'], wf['deadline']
        all_dls.append(dt.datetime.fromisoformat(d) if isinstance(d, str) else d)
        sub_dt = dt.datetime.fromisoformat(s) if isinstance(s, str) else s
        all_subs.append(sub_dt.replace(tzinfo=None))
        total_requested_tasks += len(wf['tasks'])
    
    batch_start_bound = max(min(all_subs), now)
    batch_end_bound = max(all_dls)
    print(f"[BATCH PLANNER] Batch Window: {batch_start_bound} to {batch_end_bound} | Total Requested Tasks: {total_requested_tasks}")
    
    batch_results = {}
    total_manifest = {}

    # 3. Plan each workflow
    for workflow in queue:
        plan = plan_workflow_alg3(
            workflow['run_id'], 
            batch_start_bound, 
            workflow['deadline'], 
            workflow['tasks'],
            workflow['edges'],
            workflow['sla'],
            in_memory_state=global_scratchpad 
        )
        
        batch_results[workflow['run_id']] = plan
        
        # 4. Update Manifest and Scratchpad
        for tid, tdata in plan.items():
            unique_key = f"{workflow['run_id']}_{tid}"
            total_manifest[unique_key] = tdata

            start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
            reg = tdata['region']
            t_lookup = start_dt.replace(minute=0, second=0, microsecond=0)
            
            for h in range(tdata['dur']):
                slot_time_obj = t_lookup + dt.timedelta(hours=h)
                ts = slot_time_obj.isoformat() 
                
                if reg not in global_scratchpad: 
                    global_scratchpad[reg] = {}
                    
                current_usage = global_scratchpad[reg].get(ts, 0)
                global_scratchpad[reg][ts] = current_usage + tdata.get('cores', 2)

    save_cluster_state(global_scratchpad)
    
    # 6. --- STATISTICS ---
    all_traces = {reg: load_carbon_data(reg, batch_start_bound, batch_end_bound) for reg in REGIONS}
    start_naive = batch_start_bound.replace(tzinfo=None)
    stats_to_display = calculate_standardized_stats(total_manifest, all_traces, start_naive, total_requested_tasks)

    # 7. VISUALIZE
    batch_id = f"batch_{HEURISTIC}_{dt.datetime.now().strftime('%H%M%S')}"
    
    plot_optimization_window_heatmap(
        total_manifest, 
        all_traces, 
        batch_start_bound, 
        batch_end_bound, 
        batch_id, 
        stats=stats_to_display
    )

    return batch_results

def get_safe_plan_path(run_id):
    safe_id = run_id.replace(":", "_").replace("+", "_")
    return os.path.join(PLAN_DIR, f"plan_{safe_id}.json")

def has_capacity(region, start_time, duration, cores, file_path=CLUSTER_FILE):
    if not os.path.exists(file_path): return True
    
    with open(file_path, 'r') as f:
        cluster = json.load(f)
    
    base_time = start_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    for h in range(int(duration)):
        t_str = (base_time + dt.timedelta(hours=h)).isoformat()
        
        usage = cluster.get(region, {}).get(t_str, 0)
        
        print(f"[CAPACITY CHECK] {region} at {t_str} | Current: {usage} | Request: {cores} | Limit: {TOTAL_CORES_PER_REGION}")
        
        if usage + cores > TOTAL_CORES_PER_REGION:
            return False
    return True

def check_physical_hardware_limits(current_state, region, start_hour, path, TOTAL_CORES_PER_REGION, naive_submission):
    """
    Validates if a path physically fits into the remaining cluster cores
    without exceeding 100% hardware utilization.

    NOTE: This function is functionally equivalent to calling:
        has_capacity_for_path(..., cap_factor=1.0)
    which already exists below. Kept here in case it is referenced externally,
    but consider consolidating callers to use has_capacity_for_path directly.
    """
    current_offset = start_hour
    
    for pt in path['tasks']:
        for h in range(pt['dur']):
            slot_time = naive_submission + dt.timedelta(hours=current_offset + h)
            ts_key = slot_time.replace(minute=0, second=0, microsecond=0).isoformat()
            
            current_usage = current_state.get(region, {}).get(ts_key, 0)
            
            # ABSOLUTE HARDWARE CEILING
            if current_usage + pt['cores'] > TOTAL_CORES_PER_REGION:
                return False
                
        current_offset += pt['dur']
        
    return True

def lock_resources(region, start_time, duration, cores, file_path=CLUSTER_FILE):
    data = {}
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            
    if region not in data:
        data[region] = {}
    print(f"[RESOURCE LOCK] Reserving {cores} cores in {region} from {start_time} for {duration} hours.")
    base_time = start_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
        
    for h in range(int(duration)):
        t_str = (base_time + dt.timedelta(hours=h)).isoformat()
        
        # Increment as a simple integer
        current = data[region].get(t_str, 0)
        if isinstance(current, dict): current = 0 # Wipe out old format
        
        data[region][t_str] = current + cores
        
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def task_policy(task):
    # The "optimizer" check is gone!

    def carbon_resource_manager(context):
        ti = context['ti']
        dr = context['dag_run']
        
        plan_path = get_safe_plan_path(dr.run_id)

        if not os.path.exists(plan_path):
            
            # --- PATH A: INSTANT PLANNING (Alg 1, 2, 3, 5, 6, 7) ---
            if ACTIVE_ALGORITHM in [1, 2, 3, 5, 6, 7]:
                # Extract metadata
                sub_time = dr.start_date if dr.start_date else dt.datetime.now(dt.timezone.utc)
                deadline_str = dr.conf.get('deadline_iso', (dt.datetime.now() + dt.timedelta(hours=24)).isoformat())
                deadline = dt.datetime.fromisoformat(deadline_str)
                tasks = dr.conf.get('workflow_structure', [])
                edges = []
                if hasattr(dr, 'dag') and dr.dag:
                    for task in dr.dag.tasks:
                        for downstream_id in task.downstream_task_ids:
                            edges.append((task.task_id, downstream_id))
                print(f"[METADATA] edges: {edges}")
                sla = dr.conf.get('sla_level', 95)
                print(f"{context}")
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
                    plan_baseline_atomic(dr.run_id, sub_time, deadline, tasks, edges, load_cluster_state())

            # --- PATH B: BATCH PLANNING (Alg 4) ---
            elif ACTIVE_ALGORITHM == 4:
                reg_file = os.path.join(WAITING_ROOM_DIR, f"{dr.run_id}.json")
                
                # 1. Register
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
                    print(f"[WAITING ROOM] {dr.run_id} entered the queue.")

                # 2. Check Queue
                waiting_files = [f for f in os.listdir(WAITING_ROOM_DIR) if f.endswith('.json')]
                
                if len(waiting_files) >= BATCH_SIZE:
                    # 3. Leader Election
                    lock_file = os.path.join(WAITING_ROOM_DIR, "leader.lock")
                    try:
                        with open(lock_file, "w") as lf:
                            fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                            
                            # We are the leader, run Alg 4 for everyone
                            run_batch_coordinator(waiting_files)
                            
                            fcntl.flock(lf, fcntl.LOCK_UN)
                            os.remove(lock_file)
                    except (BlockingIOError, IOError):
                        pass # Someone else is coordinating
                
                # 4. If we didn't plan it yet (either waiting for peers, or waiting for leader to finish), reschedule
                if not os.path.exists(plan_path):
                    raise AirflowRescheduleException(
                        reschedule_date=dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=60)
                    )

        # PHASE 2: EXECUTION (Plan exists now)

        print(f"[{ti.task_id}] Plan found at {plan_path}. Preparing to execute with carbon-aware scheduling.")
        with open(plan_path, 'r') as f:
            manifest = json.load(f)
            
        my_plan = manifest.get(ti.task_id)
        if not my_plan: return # Failsafe
            
        target_time = dt.datetime.fromisoformat(my_plan['start']).replace(tzinfo=dt.timezone.utc)
        print(f"[{ti.task_id}] Scheduled to run at {target_time} in region {my_plan['region']}")
        # Hold execution until the scheduled start time
        if dt.datetime.now(dt.timezone.utc) < target_time - dt.timedelta(minutes=2):
            raise AirflowRescheduleException(reschedule_date=target_time)

        ti.task.executor_config = {"pod_override": {"spec": {"nodeSelector": {"custom_region": my_plan['region']}}}}

    task.pre_execute = carbon_resource_manager

def run_batch_coordinator(waiting_files):
    """
    Executes Algorithm 4 for the queue.
    """
    print(f"\n[COORDINATOR] Queue threshold reached! Running Algorithm 4 for {len(waiting_files)} workflows.")
    
    batch_data = []
    for f_name in waiting_files:
        try:
            with open(os.path.join(WAITING_ROOM_DIR, f_name), 'r') as f:
                batch_data.append(json.load(f))
        except Exception:
            continue
            
    # Run Algorithm 4
    plan_batch_alg4(batch_data)

    # Clear the waiting room
    for f_name in waiting_files:
        try:
            os.remove(os.path.join(WAITING_ROOM_DIR, f_name))
        except OSError:
            pass
    print("[COORDINATOR] Plans distributed. Waiting room cleared.\n")