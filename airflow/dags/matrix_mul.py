import os
import numpy as np
import datetime as dt
from airflow import DAG
from airflow.sdk import task
from airflow.sdk import get_current_context

# --- CONFIGURATION & METADATA FOR PLANNER ---
DATA_DIR = "/home/gonca/Carbon_Scheduler/airflow/data/matrix_test/"
os.makedirs(DATA_DIR, exist_ok=True)

ROWS = 2000
COLS = 2000
CHUNK_A_SIZE = 250
CHUNK_B_SIZE = 500

# 1. Declarar TODAS as tarefas explicitamente para o Planeador
WORKFLOW_METADATA = [
    {"id": "setup_matrices", "dur": 1, "cores": 2, "depends_on": []},
    {"id": "prepare_pairs", "dur": 1, "cores": 1, "depends_on": ["setup_matrices"]}
]

# Adicionar as 32 tarefas mapeadas dinamicamente
for i in range(32):
    WORKFLOW_METADATA.append({
        "id": f"multiply_chunk__{i}", 
        "dur": 1, 
        "cores": 2, 
        "depends_on": ["prepare_pairs"] # <- Agora o planeador sabe que elas esperam pelo prepare_pairs!
    })

# Adicionar a tarefa de agregação final
WORKFLOW_METADATA.append({
    "id": "aggregate", 
    "dur": 1, 
    "cores": 4, 
    "depends_on": [f"multiply_chunk__{i}" for i in range(32)] # <- Depende de todas as multiplicações
})

# 2. DEFINIR OS EDGES COMPLETOS (Para manter compatibilidade se o Alg 3/4 pedir a lista de arestas)
EDGES = [
    ("setup_matrices", "prepare_pairs")
]
for i in range(32):
    EDGES.append(("prepare_pairs", f"multiply_chunk__{i}"))
    EDGES.append((f"multiply_chunk__{i}", "aggregate"))

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
        "deadline_iso": "2026-06-03T00:00:00" if os.getenv("AIRFLOW_IS_PARSING") else (dt.datetime.now() + dt.timedelta(days=3)).isoformat(),
        "workflow_structure": WORKFLOW_METADATA,
        "edges": EDGES,
        "sla_level": 95
    }
) as dag:

    @task(multiple_outputs=True)
    def setup_matrices(dag_run=None):
        """Generates matrices and saves chunks to disk"""
        run_id = dag_run.run_id
        path = os.path.join(DATA_DIR, run_id)
        os.makedirs(path, exist_ok=True)

        matrix_a = np.random.randint(1, 10, (ROWS, COLS))
        matrix_b = np.random.randint(1, 10, (COLS, ROWS))

        a_files = []
        for i in range(0, ROWS, CHUNK_A_SIZE):
            chunk = matrix_a[i:i+CHUNK_A_SIZE, :]
            f_path = os.path.join(path, f"a_chunk_{i}.npy")
            np.save(f_path, chunk)
            a_files.append((i, f_path))

        b_files = []
        for j in range(0, ROWS, CHUNK_B_SIZE):
            chunk = matrix_b[:, j:j+CHUNK_B_SIZE]
            f_path = os.path.join(path, f"b_chunk_{j}.npy")
            np.save(f_path, chunk)
            b_files.append((j, f_path))

        return {"a": a_files, "b": b_files, "path": path}

    @task
    def prepare_pairs(setup_output):
        """Cross-joins A and B chunks for dynamic mapping"""
        pairs = []
        for a in setup_output['a']:
            for b in setup_output['b']:
                pairs.append({
                    "a_info": a, 
                    "b_info": b, 
                    "base_path": setup_output['path']
                })
        return pairs

    @task(task_id="multiply_chunk")
    def multiply_task(a_info, b_info, base_path):
        """MAPPED TASK: Multiplies two chunks"""
        i_pos, a_file = a_info
        j_pos, b_file = b_info
        
        a_chunk = np.load(a_file)
        b_chunk = np.load(b_file)
        
        product = np.matmul(a_chunk, b_chunk)
        out_path = os.path.join(base_path, f"prod_{i_pos}_{j_pos}.npy")
        np.save(out_path, product)
        
        return (i_pos, j_pos, out_path)

    @task(task_id="aggregate")
    def aggregate_task(partial_results, base_path):
        """Aggregates all 32 results into one matrix"""
        final_matrix = np.zeros((ROWS, ROWS))
        for i, j, f_path in partial_results:
            chunk = np.load(f_path)
            r, c = chunk.shape
            final_matrix[i:i+r, j:j+c] = chunk
            
        final_out = os.path.join(base_path, "final_result.npy")
        np.save(final_out, final_matrix)
        return final_out

    # --- EXECUTION FLOW ---
    setup_data = setup_matrices()
    pairs = prepare_pairs(setup_data)
    
    # This creates the 32 parallel instances
    mapped_results = multiply_task.expand_kwargs(pairs)
    
    # Final aggregation
    final_result = aggregate_task(mapped_results, setup_data['path'])