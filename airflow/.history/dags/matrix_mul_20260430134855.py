from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2026, 1, 1),
}

with DAG(
    'benchmark_matrix_multiplication',
    default_args=default_args,
    schedule=None,
    catchup=False
) as dag:

    def execute_task(task_name):
        print(f"Executing {task_name}")
        # Here is where your Algorithm 3 logic would verify the start time

    # 1. Define the Merge Task (The Sink)
    merge_task = PythonOperator(
        task_id='merge_results',
        python_callable=execute_task,
        op_args=['merge_results']
    )

    # 2. Define 8 Parallel Worker Tasks (The Sources)
    for i in range(1, 9):
        worker = PythonOperator(
            task_id=f'matrix_worker_{i}',
            python_callable=execute_task,
            op_args=[f'matrix_worker_{i}']
        )
        
        # 3. Set Dependency: Worker must finish before Merge
        worker >> merge_task