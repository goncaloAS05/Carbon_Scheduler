from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
}

with DAG(
    'benchmark_tree_reduction',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:

    def execute_node(node_name):
        print(f"Processing tree node: {node_name}")

    # LEVEL 1: 4 Leaf Tasks (High Parallelism)
    leaves = [
        PythonOperator(task_id=f'leaf_{i}', python_callable=execute_node, op_args=[f'leaf_{i}'])
        for i in range(1, 5)
    ]

    # LEVEL 2: 2 Intermediate Tasks (Reduced Parallelism)
    branch_1 = PythonOperator(task_id='branch_1', python_callable=execute_node, op_args=['branch_1'])
    branch_2 = PythonOperator(task_id='branch_2', python_callable=execute_node, op_args=['branch_2'])

    # LEVEL 3: 1 Root Task (Sequential)
    root = PythonOperator(task_id='root_result', python_callable=execute_node, op_args=['root_result'])

    # --- SET DEPENDENCIES (The Tree Structure) ---
    # First half of leaves go to branch 1, second half to branch 2
    leaves[0] >> branch_1
    leaves[1] >> branch_1
    
    leaves[2] >> branch_2
    leaves[3] >> branch_2

    # Both branches merge into the root
    [branch_1, branch_2] >> root