import json
import os
import pandas as pd
import datetime as dt
import numpy as np
import copy
from datetime import timedelta
from airflow.exceptions import AirflowRescheduleException


# CONFIGURATION
CLUSTER_FILE = "/home/gonca/airflow/cluster_state.json"
DATA_DIR = "/home/gonca/airflow/plugins/data/" 
PLAN_DIR = "/home/gonca/airflow/plans/"
HISTORY_FILE = "/home/gonca/airflow/task_history.json"
REGIONS = ["DE", "PL"] 
TOTAL_CORES_PER_REGION = 7
WAITING_ROOM_DIR = "/home/gonca/airflow/waiting_room/"
os.makedirs(WAITING_ROOM_DIR, exist_ok=True)

# THE MASTER SWITCH: Choose which algorithm to test (1, 2, 3, o4 4 and 5 and 6 for the default baselines)
ACTIVE_ALGORITHM = 4
BATCH_SIZE = 3

# Ensure plan directory exists
os.makedirs(PLAN_DIR, exist_ok=True)

# --- ALG 2 CONSTANTS ---
DATA_SIZE_GB = 1  # Example: The workflow needs 1GB of input data
TRANSFER_SPEED_GBPS = 400/8 # Average inter-datacenter speed
KWH_PER_GB = 0.001875 # Energy cost of moving 1GB across the network

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def plot_optimization_window_heatmap(best_manifest, all_regions_data, submission_time, deadline, plot_name="default", stats=None):
    """
    Plots the carbon intensity heatmap with blue task boxes.
    Includes performance metrics and handles multiple tasks in the same cell.
    """
    # 1. Ensure timezones are stripped
    sub_naive = submission_time.replace(tzinfo=None)
    dl_naive = deadline.replace(tzinfo=None)
    
    # 2. Calculate total hours
    total_window_hours = int((dl_naive - sub_naive).total_seconds() // 3600) + 1
    
    # 3. Generate time slots
    time_slots = [(sub_naive + dt.timedelta(hours=i)) for i in range(total_window_hours)]
    time_labels = [t.strftime("%H:00\n%d/%m") for t in time_slots]
    
    regions = list(all_regions_data.keys())
    heatmap_matrix = np.zeros((len(regions), len(time_slots)))
    
    # 4. Fill heatmap matrix
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
    plt.figure(figsize=(max(14, total_window_hours * 0.6), 8))
    df = pd.DataFrame(heatmap_matrix, index=regions, columns=time_labels)
    
    ax = sns.heatmap(df, cmap="RdYlGn_r", annot=True, fmt=".0f", annot_kws={"size": 7},
                     cbar_kws={'label': 'gCO2eq/kWh'})

    # 6. Draw Task Boxes with Label Offsetting
    # Use a dictionary to keep track of how many labels land in the same cell
    label_collision_counter = {}

    for task_id, data in best_manifest.items():
        if data['region'] in regions:
            reg_row = regions.index(data['region'])
            t_start = dt.datetime.fromisoformat(data['start']).replace(tzinfo=None)
            duration = data.get('dur', 1) 

            hour_idx = int((t_start - sub_naive).total_seconds() // 3600)
            
            if 0 <= hour_idx < len(time_slots):
                # Draw the box (all tasks in same slot share the same box)
                ax.add_patch(plt.Rectangle((hour_idx, reg_row), duration, 1, 
                                           fill=False, edgecolor='blue', lw=3, zorder=10, alpha=0.6))
                
                # COLLISION MANAGEMENT:
                # If 10 tasks start at the same time/region, stack their names vertically
                cell_key = (hour_idx, reg_row)
                offset_idx = label_collision_counter.get(cell_key, 0)
                label_collision_counter[cell_key] = offset_idx + 1
                
                # Calculate vertical offset (starts at center, moves down)
                y_pos = reg_row + 0.2 + (offset_idx * 0.15)
                
                # Display shortened ID if it's too long
                display_id = str(task_id).split('__')[-1] # Grabs 'T1' from 'runid__T1'
                
                plt.text(hour_idx + 0.1, y_pos, display_id, 
                         color='blue', weight='bold', ha='left', va='center', 
                         fontsize=8, zorder=11, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=0))

    # 7. Add Stats Textbox
    if stats:
        stats_str = (
            f"✅ Success Rate: {stats['success_rate']}%  |  "
            f"⏱ Makespan: {stats['total_hours']}h  |  "
            f"⚡ Energy: {stats['total_energy']} kWh  |  "
            f"🌱 Carbon: {stats['total_carbon']} gCO2"
        )
        
        # We use figure coordinates (0.5 is center) 
        # y=0.02 puts it at the very bottom
        plt.gcf().text(0.5, 0.02, stats_str, fontsize=11,
                       ha='center', weight='bold',
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='#f9f9f9', alpha=1.0, edgecolor='gray'))

    # Final visual tweaks
    plt.title(f"Carbon Opportunity Window: {total_window_hours} Hours Total", fontsize=14, pad=25)
    plt.xticks(rotation=0)
    
    # Use bottom padding to ensure the stats box isn't cut off
    plt.tight_layout(rect=[0, 0.08, 1, 0.95]) 
    
    # Save
    save_path = os.path.join(PLAN_DIR, f"heatmap_{plot_name}.png")
    plt.savefig(save_path) 
    plt.close()

def calculate_standardized_stats(plan, all_traces, window_start, total_requested):
    total_energy = 0.0
    total_carbon = 0.0
    planned_ends = []
    POWER_FACTOR = 0.2  # 0.2 kW per core

    for tid, tdata in plan.items():
        # 1. Energy Calculation
        task_energy = tdata['cores'] * tdata['dur'] * POWER_FACTOR
        total_energy += task_energy
        
        # 2. Per-Hour Carbon Calculation
        start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
        reg = tdata['region']
        
        for h in range(tdata['dur']):
            hour_obj = start_dt + dt.timedelta(hours=h)
            # Match Alg 4's fallback logic
            intensity = all_traces.get(reg, {}).get(hour_obj, 400) 
            total_carbon += (tdata['cores'] * POWER_FACTOR) * intensity
            
        planned_ends.append(start_dt + dt.timedelta(hours=tdata['dur']))

    # 3. Makespan and Success Rate
    success_rate = round((len(plan) / total_requested) * 100, 1) if total_requested > 0 else 0
    
    if planned_ends:
        makespan = round((max(planned_ends) - window_start).total_seconds() / 3600, 2)
    else:
        makespan = 0

    return {
        "success_rate": success_rate,
        "total_hours": makespan,
        "total_energy": round(total_energy, 2),
        "total_carbon": round(total_carbon, 2)
    }

# def plot_optimization_window_heatmap(best_manifest, all_regions_data, submission_time, deadline, plot_name="default"):
#     # 1. Ensure timezones are stripped for comparison
#     sub_naive = submission_time.replace(tzinfo=None)
#     dl_naive = deadline.replace(tzinfo=None)
    
#     # 2. Calculate total hours - add 1 to include the deadline hour itself
#     total_window_hours = int((dl_naive - sub_naive).total_seconds() // 3600) + 1
    
#     # 3. Generate ALL time slots
#     time_slots = [(sub_naive + dt.timedelta(hours=i)) for i in range(total_window_hours)]
#     time_labels = [t.strftime("%H:00\n%d/%m") for t in time_slots]
    
#     regions = list(all_regions_data.keys())
#     heatmap_matrix = np.zeros((len(regions), len(time_slots)))
    
#     # 4. Fill the matrix (ensure we don't go out of bounds)
#     for r_idx, reg in enumerate(regions):
#         reg_intensities = all_regions_data[reg]
#         for t_idx, t_slot in enumerate(time_slots):
#             t_lookup = t_slot.replace(minute=0, second=0, microsecond=0)
#             # Find the closest timestamp in the data if exact hour is missing
#             if t_lookup in reg_intensities:
#                 heatmap_matrix[r_idx, t_idx] = reg_intensities[t_lookup]
#             elif reg_intensities:
#                 closest_t = min(reg_intensities.keys(), key=lambda d: abs(d - t_lookup))
#                 heatmap_matrix[r_idx, t_idx] = reg_intensities[closest_t]
#             else:
#                 heatmap_matrix[r_idx, t_idx] = 450 # Fallback

#     # 5. Plotting with dynamic width
#     plt.figure(figsize=(max(14, total_window_hours * 0.5), 7))
#     df = pd.DataFrame(heatmap_matrix, index=regions, columns=time_labels)
    
#     ax = sns.heatmap(df, cmap="RdYlGn_r", annot=True, fmt=".0f", annot_kws={"size": 7},
#                      cbar_kws={'label': 'gCO2eq/kWh'})

#     # 6. Draw Task Boxes (The "Why")
#     for task_id, data in best_manifest.items():
#         if data['region'] in regions:
#             reg_row = regions.index(data['region'])
#             t_start = dt.datetime.fromisoformat(data['start']).replace(tzinfo=None)
#             duration = data.get('dur', 1) 

#             # Calculate index relative to the start of our window
#             hour_idx = int((t_start - sub_naive).total_seconds() // 3600)
            
#             if 0 <= hour_idx < len(time_slots):
#                 # We draw a rectangle that covers the full duration of the task
#                 ax.add_patch(plt.Rectangle((hour_idx, reg_row), duration, 1, 
#                                             fill=False, edgecolor='blue', lw=4, zorder=10))
#                 plt.text(hour_idx + (duration/2), reg_row + 0.5, task_id, 
#                          color='blue', weight='bold', ha='center', va='center', fontsize=9)

#     plt.title(f"Carbon Opportunity Window: {total_window_hours} Hours Total")
#     plt.xticks(rotation=0)
#     plt.tight_layout()
#     plt.savefig(os.path.join(PLAN_DIR, f"heatmap_{plot_name}.png"))

def find_earliest_slot(region, requested_cores, state):
    """Finds the first hour (starting from now) where requested_cores are available."""
    now = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    for h in range(240): # Look ahead up to 10 days
        slot = (now + dt.timedelta(hours=h)).isoformat()
        used = state.get(region, {}).get(slot, 0)
        if (used + requested_cores) <= TOTAL_CORES_PER_REGION: 
            return now + dt.timedelta(hours=h)
    return now

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
        return 1.0  # Safe default: 1 hour
    
    with open(HISTORY_FILE, 'r') as f:
        history = json.load(f)
    
    # Get history for task or use DEFAULT key
    data = history.get(task_id, history.get("DEFAULT", [60]))
    
    # Ensure we don't return 0
    val = np.percentile(data, sla_level)
    return max(val, 1) / 60

def calculate_workflow_savings(plan_file, traces):
    with open(plan_file, 'r') as f:
        plan = json.load(f)
    
    total_baseline_carbon = 0
    total_aware_carbon = 0
    
    for task in plan['tasks']:
        # 1. Baseline: Running in PT (Portugal) at the moment of submission
        pt_intensity = traces['PT'][plan['submission_time']] 
        baseline = task['energy_usage'] * pt_intensity
        
        # 2. Aware: Running in the chosen Region at the chosen Start Time
        target_region = task['selected_region']
        target_time = task['selected_start_time']
        aware_intensity = traces[target_region][target_time]
        aware = task['energy_usage'] * aware_intensity
        
        total_baseline_carbon += baseline
        total_aware_carbon += aware

    saved = total_baseline_carbon - total_aware_carbon
    reduction_pct = (saved / total_baseline_carbon) * 100
    
    return saved, reduction_pct

def calculate_transfer_penalty(target_region, data_size_gb, source_region="PT"):
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

    # Log debug info to your terminal to verify the window size
    mask = (df['Datetime (UTC)'] >= search_start) & (df['Datetime (UTC)'] <= search_end)
    mask_results = df[mask]
    
    print(f"[DEBUG] {region} Data: Found {len(mask_results)} hours between {search_start} and {search_end} (Aligned to 2025) | Original Request: {start_limit} to {end_limit} (2026)")
    
    # Return mapping back to 2026 so the scheduler keys match
    return {row['Datetime (UTC)'].replace(year=2026): row['Carbon intensity gCO₂eq/kWh (Life cycle)'] 
            for _, row in mask_results.iterrows()}

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
    
    # Track bounds for plotting
    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(submission)
    dl_dt = deadline if isinstance(deadline, dt.datetime) else dt.datetime.fromisoformat(deadline)
    
    # Ensure they are timezone-naive for comparison with our scratchpad
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt = dl_dt.replace(tzinfo=None)

    best_region = None
    total_duration = sum(t['dur'] for t in tasks)
    total_cores = max(t['cores'] for t in tasks)

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

    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in REGIONS}
    stats = calculate_standardized_stats(plan, all_traces, sub_dt, len(tasks))

    # Visualization
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
    local_reg = "PT" # Assuming your 'Portugal' ID is this
    
    # 1. Defensive Date Parsing (Ensuring we have strings/datetimes handled)
    sub_dt = submission if isinstance(submission, dt.datetime) else dt.datetime.fromisoformat(str(submission))
    dl_dt = deadline if isinstance(deadline, dt.datetime) else dt.datetime.fromisoformat(str(deadline))
    
    sub_dt = sub_dt.replace(tzinfo=None)
    dl_dt = dl_dt.replace(tzinfo=None)

    # 2. Process each task individually (The "Immediate" way)
    for task in tasks:
        # Check when this specific task can start in Portugal
        start_portugal = find_earliest_slot(local_reg, task['cores'], in_memory_state)
        finish_portugal = start_portugal + dt.timedelta(hours=task['dur'])

        # Logic: If it can finish in Portugal before the deadline, stay there.
        if finish_portugal <= dl_dt:
            chosen_reg = local_reg
            chosen_start = start_portugal
        else:
            # Portugal is too busy! Find the region that can start it the SOONEST.
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

        # 4. UPDATE STATE IMMEDIATELY
        # This is vital! It tells the next 'chunk' that Portugal is now slightly more full.
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

    all_traces = {reg: load_carbon_data(reg, sub_dt, dl_dt) for reg in ["DE", "PL"]}

    # Total max cluster capacity (Change this to match your cluster configuration setting, e.g., 4 or 8 cores)

    for reg in REGIONS:
        # Find the absolute earliest the cluster can accept the FIRST task
        earliest_possible_start = find_earliest_slot(reg, max_cores_needed, in_memory_state)
        available_hours = int((dl_dt - earliest_possible_start).total_seconds() / 3600)
        
        for offset in range(available_hours - total_duration + 1):
            candidate_start = earliest_possible_start + dt.timedelta(hours=offset)
            
            block_is_valid = True
            current_block_carbon = 0.0
            
            # --- CRITICAL FIX: Scan every single hour of the block for capacity ---
            for h in range(total_duration):
                hour_obj = candidate_start + dt.timedelta(hours=h)
                hour_str = hour_obj.isoformat()
                
                # Check how many cores are already used in this region at this hour
                cores_used_already = in_memory_state.get(reg, {}).get(hour_str, 0)
                
                # If adding our workflow exceeds cluster capacity, this entire block is invalid
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
    
    pt_intensities = load_carbon_data("PT", submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    
    # Get PT baseline intensity (current moment)
    if sub_naive in pt_intensities:
        pt_base = pt_intensities[sub_naive]
    elif pt_intensities:
        pt_base = pt_intensities[sorted(pt_intensities.keys())[0]]
    else:
        pt_base = 400

    # Set the baseline as the current "best"
    best_avg = pt_base
    best_start = submission_time
    best_reg = "PT"

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
        
        for offset in range(max(1, window_hours - total_duration)):
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
    
    # 3. SMART GET: If exact hour is missing, find the first available value
    pt_intensities = load_carbon_data("PT", submission_time, deadline_date)
    if sub_naive in pt_intensities:
        pt_start_intensity = pt_intensities[sub_naive]
    elif pt_intensities:
        # Get the first value in the sorted dictionary as a fallback
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

    plot_optimization_window_heatmap(
        manifest,
        {reg: load_carbon_data(reg, submission_time, deadline_date) for reg in REGIONS},
        sub_naive,
        deadline_date,
        plot_name=f"alg1_{run_id}"
    )

    lock_resources(best_reg, best_start, total_duration, total_cores)
    with open(plan_path, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    return manifest

def plan_workflow_alg2(run_id, submission_time, deadline_date, tasks_metadata, data_size_gb):
    print(f"[ALG 2] Planning for {data_size_gb}GB")
    plan_path = get_safe_plan_path(run_id)
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    total_duration = sum(t['dur'] for t in tasks_metadata)
    total_cores = max(t['cores'] for t in tasks_metadata)
    
    # Initialize competitive variables
    best_total_impact = float('inf')
    best_intensity = 0
    best_start = submission_time
    best_reg = REGIONS[0]

    # Timezone normalization
    if submission_time.tzinfo is None: submission_time = submission_time.replace(tzinfo=dt.timezone.utc)
    if isinstance(deadline_date, str): deadline_date = dt.datetime.fromisoformat(deadline_date)
    if deadline_date.tzinfo is None: deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)

    for region in REGIONS:
        # 1. Calculate penalty BEFORE searching slots in this region
        time_penalty, network_tax = calculate_transfer_penalty(region, data_size_gb)
        
        # Adjust search window based on transfer time
        effective_deadline = deadline_date - dt.timedelta(hours=time_penalty)
        window_hours = int((effective_deadline - submission_time).total_seconds() // 3600)
        
        intensities = load_carbon_data(region, submission_time, effective_deadline)

        for offset in range(max(1, window_hours - total_duration)):
            # Start time must account for the time it takes to move the data
            start_cand = submission_time + dt.timedelta(hours=offset + time_penalty)
            lookup_time = start_cand.replace(tzinfo=None, minute=0, second=0, microsecond=0)
            
            # 2. Get the specific intensity at the chosen start moment
            current_intensity = intensities.get(lookup_time, 999)
            
            # 3. Calculate Total Impact for THIS specific slot
            execution_carbon = current_intensity * (total_duration * total_cores)
            total_impact = execution_carbon + network_tax

            if has_capacity(region, start_cand, total_duration, total_cores):
                if total_impact < best_total_impact:
                    best_total_impact = total_impact
                    best_intensity = current_intensity
                    best_start = start_cand
                    best_reg = region

    # --- POST-DECISION REPORTING ---
    # Baseline: Portugal at the moment of submission
    pt_intensities = load_carbon_data("PT", submission_time, deadline_date)
    
    # 2. Floor the submission time to the hour
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    
    # 3. SMART GET: If exact hour is missing, find the first available value
    if sub_naive in pt_intensities:
        pt_start_intensity = pt_intensities[sub_naive]
    elif pt_intensities:
        # Get the first value in the sorted dictionary as a fallback
        available_times = sorted(pt_intensities.keys())
        pt_start_intensity = pt_intensities[available_times[0]]
        print(f"[DEBUG] Exact time {sub_naive} not in PT logs. Using closest: {available_times[0]}")
    else:
        pt_start_intensity = 400 # Realistic fallback for PT if logs are empty
        print("[DEBUG] PT logs empty for this window. Using global fallback.")

    local_execution_carbon = pt_start_intensity * (total_duration * total_cores)
    
    # Final Winner Stats
    remote_execution_carbon = best_intensity * (total_duration * total_cores)
    _, winner_tax = calculate_transfer_penalty(best_reg, data_size_gb)
    
    carbon_saved = local_execution_carbon - best_total_impact
    
    print(f"\n[ALGORITHM 2 REPORT]")
    print(f"  - Local PT Start Intensity: {pt_start_intensity} g/kWh")
    print(f"  - Winner {best_reg} Start Intensity: {best_intensity} g/kWh")
    print(f"  - Transfer Carbon Cost: {round(winner_tax, 2)} gCO2eq")
    print(f"  - Net Savings: {round(carbon_saved, 2)} gCO2eq")

    # Build manifest
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

    plot_optimization_window_heatmap(
        manifest, 
        {reg: load_carbon_data(reg, submission_time, deadline_date) for reg in REGIONS}, 
        sub_naive, 
        deadline_date, 
        plot_name=f"alg2_{run_id}"
    )

    lock_resources(best_reg, best_start, total_duration, total_cores)
    with open(plan_path, 'w') as f: json.dump(manifest, f, indent=4)
    return manifest

# ... (mantenha todo o seu código inicial de importação e constantes igual)

def plan_workflow_alg3(run_id, submission_time, deadline_date, tasks_metadata, edges, sla_level, in_memory_state=None):
    plan_path = get_safe_plan_path(run_id)
    if os.path.exists(plan_path):
        with open(plan_path, 'r') as f: return json.load(f)

    # 1. Setup inicial e Normalização
    cp_duration_hours = get_dynamic_critical_path(tasks_metadata, edges, sla_level)
    total_cores = max(t['cores'] for t in tasks_metadata)
    if isinstance(submission_time, str):
        submission_time = dt.datetime.fromisoformat(submission_time)

    if submission_time.tzinfo is None: submission_time = submission_time.replace(tzinfo=dt.timezone.utc)
    if isinstance(deadline_date, str): deadline_date = dt.datetime.fromisoformat(deadline_date)
    deadline_date = deadline_date.replace(tzinfo=dt.timezone.utc)

    # 2. Baseline de Portugal (PT) - Metodologia EuroSys '24
    pt_traces = load_carbon_data("PT", submission_time, deadline_date)
    sub_naive = submission_time.replace(tzinfo=None, minute=0, second=0, microsecond=0)
    pt_base_ci = pt_traces.get(sub_naive, pt_traces[min(pt_traces.keys(), key=lambda d: abs(d-sub_naive))] if pt_traces else 450)
    
    best_total_impact = float('inf')
    best_reg, best_start, best_manifest, best_duration = None, None, {}, int(cp_duration_hours)

    if in_memory_state is None:
        in_memory_state = load_cluster_state()

    # 3. Loop de Otimização (Spatio-Temporal Shifting)
    for region in REGIONS:
        # ... (Penalties and intensity loading)
        time_penalty, carbon_penalty = calculate_transfer_penalty(region, DATA_SIZE_GB)
        eff_deadline = deadline_date - dt.timedelta(hours=time_penalty)
        intensities = load_carbon_data(region, submission_time, eff_deadline)
        if not intensities: continue

        window_hours = int((eff_deadline - submission_time).total_seconds() // 3600)

        for offset in range(max(1, int(window_hours - cp_duration_hours))):
            start_cand = submission_time + dt.timedelta(hours=offset + time_penalty)
            sim_carbon, max_off, can_fit, cand_manifest = 0, 0, True, {}
            
            # We use a temporary copy for this specific "search" iteration
            temp_search_state = copy.deepcopy(in_memory_state)

            for t in tasks_metadata:
                task_planned = False
                remaining_window = int((eff_deadline - start_cand).total_seconds() // 3600)
                for h_off in range(max(1, remaining_window)):               
                    t_start = start_cand + dt.timedelta(hours=h_off)
                    t_lookup = t_start.replace(tzinfo=None, minute=0, second=0, microsecond=0)
                    
                    # --- THE CORE FIX: Check against the dynamic state ---
                    can_reserve = True
                    for h in range(t['dur']):
                        ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                        usage = temp_search_state.get(region, {}).get(ts, 0)
                        if usage + t['cores'] > TOTAL_CORES_PER_REGION:
                            can_reserve = False
                            break
                    
                    if can_reserve:
                        # Plan it!
                        t_ci = intensities.get(min(intensities.keys(), key=lambda d: abs(d - t_lookup)), 400)
                        t_aware_carbon = t_ci * (t['dur'] * t['cores'])
                        sim_carbon += t_aware_carbon
                        t_baseline_carbon = pt_base_ci * (t['dur'] * t['cores'])
                        
                        # 2. SAVE EVERYTHING into the dictionary
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

                        max_off = max(max_off, h_off + t['dur'])
                        # Update the state IMMEDIATELY so T2 sees T1's cores
                        for h in range(t['dur']):
                            ts = (t_lookup + dt.timedelta(hours=h)).isoformat()
                            if region not in temp_search_state: temp_search_state[region] = {}
                            temp_search_state[region][ts] = temp_search_state[region].get(ts, 0) + t['cores']
                        
                        task_planned = True
                        break
                
                if not task_planned: (can_fit := False); break
            
            if can_fit and (sim_carbon + carbon_penalty) < best_total_impact:
                best_total_impact = sim_carbon + carbon_penalty
                best_start, best_reg, best_manifest, best_duration = start_cand, region, cand_manifest, max_off

    # 4. Relatório Final (Terminal Output)
    if best_reg:
        print(f"\n[ALGORITHM 3 REPORT] Workflow: {run_id}")
        print(f"{'Task ID':<15} | {'Region':<8} | {'Saved (g)':<10} | {'Reduction %'}")
        print("-" * 50)
        
        all_regions_window_data = {}
        for reg in REGIONS:
            # Load the data for the full window (Submission -> Deadline)
            all_regions_window_data[reg] = load_carbon_data(reg, submission_time, deadline_date)

        # --- CALLING THE HEATMAP ---
        # Use the finalized best_manifest and the full window data
        plot_optimization_window_heatmap(
            best_manifest=best_manifest,
            all_regions_data=all_regions_window_data,
            submission_time=submission_time,
            deadline=deadline_date,
            plot_name=f"alg3_{run_id}"
        )

        actual_sum_saved_g = 0
        actual_sum_baseline_g = 0
        
        for tid, d in best_manifest.items():
            print(f"{tid:<15} | {d['region']:<8} | {d['saved_g']:<10} | {d['pct']}%")
            actual_sum_saved_g += d['saved_g']
            # Reconstruct baseline to get accurate total %
            actual_sum_baseline_g += (d['saved_g'] / (d['pct']/100)) if d['pct'] > 0 else 0
        
        total_reduction_pct = (actual_sum_saved_g / actual_sum_baseline_g * 100) if actual_sum_baseline_g > 0 else 0
        
        print(f"--- TOTAL SAVED: {round(actual_sum_saved_g, 2)}g ({round(total_reduction_pct, 1)}%) ---\n")
        print(f"--- TOTAL if ran in PT: {round(actual_sum_baseline_g, 2)}g ---\n")
        lock_resources(best_reg, best_start, best_duration, total_cores)
        with open(plan_path, 'w') as f: json.dump(best_manifest, f, indent=4)
        return best_manifest

    return {}
    
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

def plan_batch_alg4(dag_runs_metadata):
    """
    Explores heuristics like 'Longest Job First' to reduce fragmentation.
    Now includes statistics calculation for Energy, Carbon, and Success Rate.
    """
    HEURISTIC = "EDF" 
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

    # 5. ATOMIC COMMIT
    save_cluster_state(global_scratchpad)
    
    # 6. --- CALCULATE STATISTICS ---
    all_traces = {reg: load_carbon_data(reg, batch_start_bound, batch_end_bound) for reg in REGIONS}
    
    total_energy = 0.0
    total_carbon = 0.0
    planned_ends = []

    for tid, tdata in total_manifest.items():
        # Energy = Cores * Hours * Power_Factor (e.g., 0.2 kW per core)
        task_energy = tdata['cores'] * tdata['dur'] * 0.2
        total_energy += task_energy
        
        # Carbon = Energy per hour * Intensity for that hour/region
        start_dt = dt.datetime.fromisoformat(tdata['start']).replace(tzinfo=None)
        reg = tdata['region']
        for h in range(tdata['dur']):
            ts_obj = start_dt + dt.timedelta(hours=h)
            # Use the intensity from our loaded traces
            intensity = all_traces.get(reg, {}).get(ts_obj, 400) # Fallback to 400 if missing
            total_carbon += (tdata['cores'] * 0.2) * intensity
            
        planned_ends.append(start_dt + dt.timedelta(hours=tdata['dur']))

    # Success Rate and Makespan
    success_rate = round((len(total_manifest) / total_requested_tasks) * 100, 1) if total_requested_tasks > 0 else 0

    if planned_ends:
        # Ensure batch_start_bound is naive before subtraction
        start_naive = batch_start_bound.replace(tzinfo=None)
        max_end_naive = max(planned_ends).replace(tzinfo=None)
        
        makespan = round((max_end_naive - start_naive).total_seconds() / 3600, 2)
    else:
        makespan = 0

    stats_to_display = {
        "success_rate": success_rate,
        "total_hours": makespan,
        "total_energy": round(total_energy, 2),
        "total_carbon": round(total_carbon, 2)
    }

    # 7. VISUALIZE
    batch_id = f"batch_{HEURISTIC}_{dt.datetime.now().strftime('%H%M%S')}"
    print(f"[DEBUG] Total tasks: {len(total_manifest)} | Carbon: {total_carbon} gCO2eq")
    
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
    # Characters like ':' and '+' break filenames in some OS environments
    safe_id = run_id.replace(":", "_").replace("+", "_")
    return os.path.join(PLAN_DIR, f"plan_{safe_id}.json")

def has_capacity(region, start_time, duration, cores, file_path=CLUSTER_FILE):
    if not os.path.exists(file_path): return True
    
    with open(file_path, 'r') as f:
        cluster = json.load(f)
    
    # FIX 1: Strip minutes/seconds so we match the "Hour" slots in the JSON
    base_time = start_time.replace(minute=0, second=0, microsecond=0, tzinfo=None)
    
    # FIX 2: Ensure duration is an integer for range()
    for h in range(int(duration)):
        t_str = (base_time + dt.timedelta(hours=h)).isoformat()
        
        usage = cluster.get(region, {}).get(t_str, 0)
        
        # DEBUG PRINT: This will tell you exactly what's happening in the logs
        print(f"[CAPACITY CHECK] {region} at {t_str} | Current: {usage} | Request: {cores} | Limit: {TOTAL_CORES_PER_REGION}")
        
        if usage + cores > TOTAL_CORES_PER_REGION:
            return False
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

import fcntl


def task_policy(task):
    # The "optimizer" check is gone!

    def carbon_resource_manager(context):
        ti = context['ti']
        dr = context['dag_run']
        
        plan_path = get_safe_plan_path(dr.run_id)
        
        # ========================================================
        # PHASE 1: GENERATE THE PLAN
        # ========================================================
        if not os.path.exists(plan_path):
            
            # --- PATH A: INSTANT PLANNING (Alg 1, 2, 3) ---
            if ACTIVE_ALGORITHM in [1, 2, 3, 5, 6, 7]:
                # Extract metadata once
                sub_time = dr.start_date if dr.start_date else dt.datetime.now(dt.timezone.utc)
                deadline_str = dr.conf.get('deadline_iso', (dt.datetime.now() + dt.timedelta(hours=24)).isoformat())
                deadline = dt.datetime.fromisoformat(deadline_str)
                tasks = dr.conf.get('workflow_structure', [])
                edges = dr.conf.get('edges', [])
                sla = dr.conf.get('sla_level', 95)

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

            # --- PATH B: BATCH PLANNING (Alg 4 only) ---
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

        # ========================================================
        # PHASE 2: EXECUTION (Plan definitely exists now)
        # ========================================================
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

        # Apply region constraint
        task.executor_config = {"pod_override": {"spec": {"nodeSelector": {"custom_region": my_plan['region']}}}}

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