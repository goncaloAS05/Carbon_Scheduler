import os
import numpy as np
import datetime as dt
from airflow import DAG
from airflow.decorators import task

# --- CONFIGURATION ---
DATA_DIR = "/home/gonca/airflow/data/tree_reduction_test/"
os.makedirs(DATA_DIR, exist_ok=True)

ARRAY_SIZE = 5000

# Metadata for Planner matching your structure pattern
WORKFLOW_METADATA = [
    {"id": "generate_leaf__0", "dur": 2, "cores": 2},
    {"id": "generate_leaf__1", "dur": 2, "cores": 2},
    {"id": "generate_leaf__2", "dur": 1, "cores": 1},
    {"id": "generate_leaf__3", "dur": 1, "cores": 1},
    {"id": "reduce_branch__0", "dur": 2, "cores": 2},
    {"id": "reduce_branch__1", "dur": 1, "cores": 2},
    {"id": "final_reduction", "dur": 2, "cores": 1}
]

EDGES = [
    ("generate_leaf__0", "reduce_branch__0"),
    ("generate_leaf__1", "reduce_branch__0"),
    ("generate_leaf__2", "reduce_branch__1"),
    ("generate_leaf__3", "reduce_branch__1"),
    ("reduce_branch__0", "final_reduction"),
    ("reduce_branch__1", "final_reduction")
]

default_args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2026, 5, 5),
}

with DAG(
    dag_id="workflow_tree_reduction_carbon",
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
    def generate_leaf(leaf_id: int, seed_val: float, dag_run=None):
        run_id = dag_run.run_id
        path = os.path.join(DATA_DIR, run_id)
        os.makedirs(path, exist_ok=True)
        
        arr = np.linspace(seed_val, seed_val + 10.0, ARRAY_SIZE, dtype=np.float64)
        f_path = os.path.join(path, f"leaf_{leaf_id}.npy")
        np.save(f_path, arr)
        return f_path

    @task
    def reduce_branch(branch_id: int, leaf_a_path: str, leaf_b_path: str, dag_run=None):
        run_id = dag_run.run_id
        path = os.path.join(DATA_DIR, run_id)
        
        matrix_a = np.load(leaf_a_path)
        matrix_b = np.load(leaf_b_path)
        
        combined = np.sqrt((matrix_a ** 2) + (matrix_b ** 2))
        f_path = os.path.join(path, f"branch_{branch_id}.npy")
        np.save(f_path, combined)
        return f_path

    @task(task_id="final_reduction")
    def final_reduction(branch_a_path: str, branch_b_path: str, dag_run=None):
        run_id = dag_run.run_id
        path = os.path.join(DATA_DIR, run_id)
        
        branch_a = np.load(branch_a_path)
        branch_b = np.load(branch_b_path)
        
        final_vector = branch_a - branch_b
        norm_val = float(np.linalg.norm(final_vector))
        
        final_out = os.path.join(path, "final_tree_result.npy")
        np.save(final_out, final_vector)
        return {"path": final_out, "norm": norm_val}

    # --- EXECUTION FLOW ---
    l0 = generate_leaf(0, 1.5)
    l1 = generate_leaf(1, 2.5)
    l2 = generate_leaf(2, 3.5)
    l3 = generate_leaf(3, 4.5)

    b0 = reduce_branch(0, l0, l1)
    b1 = reduce_branch(1, l2, l3)

    final_result = final_reduction(b0, b1)