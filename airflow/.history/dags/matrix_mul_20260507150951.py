import os
import numpy as np
import datetime as dt
from airflow import DAG
from airflow.decorators import task
from airflow.operators import get_current_context

# --- CONFIGURATION ---
DATA_DIR = "/home/gonca/airflow/data/matrix_test/"
os.makedirs(DATA_DIR, exist_ok=True)

# 2000x2000 Matrix
ROWS = 2000
COLS = 2000

# To get 32 tasks:
# Matrix A: 8 chunks (2000 / 250)
# Matrix B: 4 chunks (2000 / 500)
CHUNK_A_SIZE = 250
CHUNK_B_SIZE = 500

# Metadata for your Algorithm 4 Planner
# Every mult task takes ~2 cores and lasts ~1 hour (estimated for simulation)
WORKFLOW_METADATA = []
for i in range(8):
    for j in range(4):
        WORKFLOW_METADATA.append({"id": f"mult_{i}_{j}", "dur": 1, "cores": 2})
WORKFLOW_METADATA.append({"id": "aggregate", "dur": 1, "cores": 4})

EDGES = []
for i in range(8):
    for j in range(4):
        EDGES.append(("mult", f"mult_{i}_{j}", "aggregate"))

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2026, 5, 5),
}

with DAG(
    dag_id="matrix_multiplication_carbon",
    default_args=default_args,
    schedule=None,
    catchup=False,
    params={
        "deadline_iso": (dt.datetime.now() + dt.timedelta(days=3)).isoformat(),
        "workflow_structure": WORKFLOW_METADATA,
        "edges": EDGES,
        "sla_level": 95
    }
) as dag:

    @task
    def setup_matrices():
        """Generates matrices and saves chunks to disk"""
        run_id = get_current_context()['dr'].run_id
        path = os.path.join(DATA_DIR, run_id)
        os.makedirs(path, exist_ok=True)

        matrix_a = np.random.randint(1, 10, (ROWS, COLS))
        matrix_b = np.random.randint(1, 10, (COLS, ROWS))

        a_files = []
        for i in range(0, ROWS, CHUNK_A_SIZE):
            chunk = matrix_a[i:i+CHUNK_A_SIZE, :]
            f_path = f"{path}/a_chunk_{i}.npy"
            np.save(f_path, chunk)
            a_files.append((i, f_path))

        b_files = []
        for j in range(0, ROWS, CHUNK_B_SIZE):
            chunk = matrix_b[:, j:j+CHUNK_B_SIZE]
            f_path = f"{path}/b_chunk_{j}.npy"
            np.save(f_path, chunk)
            b_files.append((j, f_path))

        return {"a": a_files, "b": b_files, "path": path}

    @task(task_id="multiply_chunk")
    def multiply_task(a_info, b_info, base_path):
        """Performs the actual multiplication of two chunks"""
        i_pos, a_file = a_info
        j_pos, b_file = b_info
        
        a_chunk = np.load(a_file)
        b_chunk = np.load(b_file)
        
        product = np.matmul(a_chunk, b_chunk)
        out_path = f"{base_path}/prod_{i_pos}_{j_pos}.npy"
        np.save(out_path, product)
        
        return (i_pos, j_pos, out_path)

    @task
    def aggregate(partial_results, base_path):
        """Stitches the results back together"""
        final_matrix = np.zeros((ROWS, ROWS))
        
        for i, j, f_path in partial_results:
            chunk = np.load(f_path)
            r, c = chunk.shape
            final_matrix[i:i+r, j:j+c] = chunk
            
        final_out = f"{base_path}/final_result.npy"
        np.save(final_out, final_matrix)
        return final_out

    # --- THE WORKFLOW ---
    setup = setup_matrices()
    
    # Create the 32 parallel tasks
    results = []
    for a_chunk in setup['a']:
        for b_chunk in setup['b']:
            # Dynamically creating tasks that Alg 4 will intercept
            res = multiply_task.override(task_id=f"mult_{a_chunk[0]}_{b_chunk[0]}")(
                a_chunk, b_chunk, setup['path']
            )
            results.append(res)
            
    final_result = aggregate(results, setup['path'])