import seaborn as sns
import matplotlib.pyplot as plt
import os
import requests
from datetime import datetime, timedelta
from airflow.listeners import hookimpl
from airflow.utils.state import State
from airflow import settings
import pandas as pd
import glob



# If you are port-forwarding to your local machine for testing:
# PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

KEPLER_URL = "http://127.0.0.1:9102/metrics"
AUDIT_LOG = os.path.expanduser("~/airflow/listener_audit.log")
MATRIX_LOG = os.path.expanduser("~/airflow/tradeoff_matrix.csv")
TEMP_DIR = os.path.expanduser("~/airflow/airflow_energy_metadata")
RE_STORE = os.path.expanduser("~/airflow/recommendation_store.json")

def get_carbon_intensity(zone):
    api_key = "jdQCGj8jgrv8dswKbgUj"
    url = f"https://api.electricitymaps.com/v3/carbon-intensity/latest?zone={zone}"
    
    headers = {"auth-token": api_key}
    
    try:
        response = requests.get(url, headers=headers)
        data = response.json()
        return data['carbonIntensity'] # Returns value in gCO2eq/kWh
    except Exception as e:
        print(f"API Error: {e}. Falling back to random simulation.")
        return 0

def get_task_energy(pod_name):
    """Fetches Joules consumed by a Pod ONLY during the task duration."""
    # We ask Prometheus for the increase in Joules over the last X seconds
    try:
        response = requests.get(KEPLER_URL, timeout=2)
        response.raise_for_status()
        
        # We look for 'core_joules' which we know exists from your terminal
        target_metric = "kepler_container_core_joules_total"
        with open(AUDIT_LOG, "a") as f:
                    f.write(f"DEBUG: Successfully pulled {target_metric} Wh directly from Kepler for {pod_name}\n")
        for line in response.text.splitlines():
            # Check if this line is our metric AND belongs to our pod
            if target_metric in line and f'pod_name="{pod_name}"' in line:
                # Line format: metric_name{labels} value
                # We split by space and take the last element (the number)
                joules = float(line.split()[-1])
                wh = joules / 3600
                
                with open(AUDIT_LOG, "a") as f:
                    f.write(f"DEBUG: Successfully pulled {wh:.6f} Wh directly from Kepler for {pod_name}\n")
                return wh
                
    except Exception as e:
        with open(AUDIT_LOG, "a") as f:
            f.write(f"DEBUG: Direct Kepler pull failed: {str(e)}\n")
    return 0.05  # Fallback

def generate_tradeoff_matrix(wh_consumed, start_time):
    """Simulates the CO2 cost across 5 regions for the next 7 days."""
    data = []
    # Note: Use 'SE' for Sweden if that's what is in your Zone id/filename
    regions = ["PT", "DE", "PL", "ES", "SW"] 
    
    # Standard column names from your CSV format
    time_col = 'Datetime (UTC)'
    intensity_col = 'Carbon intensity gCO₂eq/kWh (Life cycle)' 
    sim_start_naive = start_time.replace(tzinfo=None)
    for region in regions:
        # 1. Load the specific CSV for this region
        # Assumes filenames are like: logPT.csv, logDE.csv, etc.
        try:
            df_carbon = pd.read_csv(f"~/airflow/plugins/data/log_{region}.csv")
            
            # 2. Convert time column to datetime and set as index for fast lookup
            df_carbon[time_col] = pd.to_datetime(df_carbon[time_col]).dt.tz_localize(None)
            df_carbon.set_index(time_col, inplace=True)
            
            # 3. Simulate starting the task at different hours (0 to 168h)
            for delay in range(169):
                sim_time = sim_start_naive + timedelta(hours=delay)
                
                # Round sim_time to the top of the hour to match CSV format
                lookup_time = sim_time.replace(minute=0, second=0, microsecond=0)

                # 4. Get intensity from the index
                if lookup_time in df_carbon.index:
                    intensity = df_carbon.loc[lookup_time, intensity_col]
                    
                    # Carbon calculation: (Wh / 1000 to get kWh) * gCO2/kWh
                    co2_score = (wh_consumed / 1000) * intensity
                    
                    data.append({
                        "Region": region,
                        "Delay_Hours": delay,
                        "CO2_grams": co2_score,
                        "Wait_Time_Penalty": delay * 0.1  # Your thesis penalty metric
                    })
                else:
                    # If data is missing for a specific hour in the CSV
                    with open(AUDIT_LOG, "a") as f:
                        f.write(f"DEBUG: Could not find {lookup_time} in {region} index. First index: {df_carbon.index[0]}\n")
                    continue
                    
        except FileNotFoundError:
            print(f"Warning: Could not find file log_{region}.csv")
            continue

    return pd.DataFrame(data)


def save_carbon_heatmap(matrix_df, dag_id):
    """Generates a heatmap showing CO2 grams by Region and Delay."""
    # Values will be the CO2 grams
    pivot_df = matrix_df.pivot(index="Region", columns="Delay_Hours", values="CO2_grams")
    import matplotlib
    matplotlib.use('Agg')

    # 2. Create the plot
    plt.figure(figsize=(15, 8))
    sns.heatmap(
        pivot_df, 
        cmap="RdYlGn_r",  # Red (High CO2) to Green (Low CO2)
        annot=False,      # Set to True if you want numbers in the boxes
        cbar_kws={'label': 'gCO2eq'}
    )

    plt.title(f"Carbon Intensity Tradeoff for DAG: {dag_id}")
    plt.xlabel("Delay (Hours)")
    plt.ylabel("Region")
    
    # 3. Save it for your thesis document
    plot_path = os.path.expanduser(f"~/airflow/heatmap_{dag_id}.png")
    plt.savefig(plot_path)
    plt.close()
    return plot_path

@hookimpl
def on_task_instance_running(previous_state, task_instance):
    print(f"[Task] {task_instance.task_id} is running")

@hookimpl
def on_task_instance_success(previous_state, task_instance):
    # Obter o tempo de execução (em segundos)

    # Obter a intensidade do Carbono 
    pod_name = task_instance.xcom_pull(task_ids=task_instance.task_id, key='pod_name')
    
    # 1. Get real energy from Prometheus/Kepler
    real_wh = get_task_energy(pod_name)
    
    # 2. If Kepler fails (returns 0 or fallback), use your power formula
    # if real_wh <= 0.05: # if it looks like the fallback
    duration_h = (task_instance.end_date - task_instance.start_date).total_seconds() / 3600
    #     # Your previous logic: (mem_gb * 0.003 + 0.5 * 0.015) * duration_h * 1000
    #     real_wh = (1.0 * 0.003 + 0.5 * 0.015) * duration_h * 1000 

    # 3. Store this in XCom so the DAG-level hook can find it
    task_instance.xcom_push(key='kepler_energy_wh', value=real_wh)

    ti_task_id = task_instance.task_id
    ti_run_id = task_instance.run_id
    file_path = os.path.join(TEMP_DIR, f"{ti_run_id}_{ti_task_id}.wh")
    with open(file_path, "a") as f:
        f.write(str(real_wh))
    # Log dos Eventos 
    print(f"[Task] {task_instance.task_id} succeeded with estimated energy: {real_wh:.2f} Wh")
    with open(AUDIT_LOG, "a") as f:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{ts}] [METRIC_CO2] Tarefa: {task_instance.task_id} | {pod_name}"
                f" | Tempo: {duration_h}s | "
                f"energy: {real_wh}Wh \n")

@hookimpl
def on_dag_run_success(dag_run, msg):
    import time
    from airflow.utils.session import create_session
    from airflow.models import TaskInstance, XCom
    from airflow.utils.state import State
    
    # 1. Capture plain strings immediately. DO NOT touch dag_run after this.
    dr_dag_id = str(dag_run.dag_id)
    dr_run_id = str(dag_run.run_id)
    dr_start = dag_run.start_date
    dr_logical_date = dag_run.logical_date
    time.sleep(2)
    total_workflow_wh = 0.0
    with open(AUDIT_LOG, "a") as f:
                f.write(f"\n--- DEBUG algo estranhoS:  ---\n")
    # 2. Open a completely independent session
    # with create_session() as session:
    #     try:
    #         # Query TaskInstances manually so we don't trigger lazy-loading on dag_run
    #         tis = session.query(TaskInstance).filter(
    #             TaskInstance.dag_id == dr_dag_id,
    #             TaskInstance.run_id == dr_run_id
    #         ).all()

    #         with open(AUDIT_LOG, "a") as f:
    #             f.write(f"\n--- DAG SUMMARY: {dr_dag_id} ---\n")
                
    #             for ti in tis:
    #                 # 2. Use session.query(XCom) instead of ti.xcom_pull to avoid the 'COMMIT' error
    #                 # We look for the 'kepler_energy_wh' key you pushed in the task hook
    #                 wh_value = XCom.get_value(
    #                     ti=ti,
    #                     key='kepler_energy_wh',
    #                     session=session
    #                 )

    #                 # 3. Pull the value safely
    #                 # Airflow stores XComs as 'value'. We use .value to get the deserialized data.

    #                 # 3. Sum it up
    #                 wh = float(energy_val) if energy_val is not None else 0.05
    #                 total_workflow_wh += wh
    #                 f.write(f"Task: {ti.task_id} | Energy: {wh:.4f} Wh\n")

    #             f.write(f"TOTAL WORKFLOW ENERGY: {total_workflow_wh:.4f} Wh\n")

    #     except Exception as e:
    #         with open(AUDIT_LOG, "a") as f:
    #             f.write(f"ERROR in DAG Listener: {str(e)}\n")
    #         return
    search_pattern = os.path.join(TEMP_DIR, f"{dr_run_id}_*.wh")
    energy_files = glob.glob(search_pattern)
    
    for f_path in energy_files:
        try:
            with open(f_path, "r") as f:
                total_workflow_wh += float(f.read())
            os.remove(f_path) # Clean up
        except Exception:
            pass
    with open(AUDIT_LOG, "a") as f:
        f.write(f"TOTAL WORKFLOW ENERGY from temp files: {total_workflow_wh:.4f} Wh\n")
    # 3. Proceed with Simulation using the local total_workflow_wh
    if total_workflow_wh > 0:
        # Final simulation logic
        sim_start = dr_start.replace(year=2025).replace(tzinfo=None)
        matrix_df = generate_tradeoff_matrix(total_workflow_wh, sim_start)
        
        if not matrix_df.empty:
            best = matrix_df.loc[matrix_df['CO2_grams'].idxmin()]
            plot_file = save_carbon_heatmap(matrix_df, dag_run.dag_id)
            with open(MATRIX_LOG, "w") as f:
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] Final Result: {best['Region']} is optimal.\n")
                f.write(matrix_df.to_string(index=False))


def get_historical_avg(task_id):
    """Simple historical model: in a real thesis, this would query a DB."""
    if "heavy" in task_id.lower(): return 1.5  # Wh
    return 0.1  # Wh default

@hookimpl
def on_task_instance_queued(task_instance):
    """
    SCOUT: Runs while the task is in the queue.
    Finds the absolute best Region AND Time to save the most CO2.
    """
    ti_task_id = task_instance.task_id
    ti_run_id = task_instance.run_id

    print("QUEUD\n")
    
    # 1. Predict energy (Historical/Model)
    predicted_wh = get_historical_avg(ti_task_id)
    
    # 2. Run simulation for the next 48 hours across all regions
    sim_start = datetime.now().replace(year=2025)
    matrix_df = generate_tradeoff_matrix(predicted_wh, sim_start)
    print(f"Matrix DataFrame:\n{matrix_df}")
    if not matrix_df.empty:
        # Find the row with the absolute minimum CO2_grams
        best_row = matrix_df.loc[matrix_df['CO2_grams'].idxmin()]
        
        best_region = best_row['Region']
        delay_hours = int(best_row['Delay_Hours'])
        
        # Calculate the actual future timestamp for rescheduling
        reschedule_time = (datetime.now() + timedelta(hours=delay_hours)).isoformat()
        
        # 3. Save the Decision (Region + Time)
        import json
        recommendations = {}
        if os.path.exists(RE_STORE):
            try:
                with open(RE_STORE, "r") as f: recommendations = json.load(f)
            except: recommendations = {}
            
        recommendations[f"{ti_run_id}_{ti_task_id}"] = {
            "region": best_region,
            "reschedule_until": reschedule_time,
            "predicted_co2": float(best_row['CO2_grams'])
        }
        
        with open(RE_STORE, "w") as f:
            json.dump(recommendations, f)

    with open(AUDIT_LOG, "a") as f:
        f.write(f"[{datetime.now()}] SCOUT: {ti_task_id} | Plan: {best_region} in {delay_hours}h\n")