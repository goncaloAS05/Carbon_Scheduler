from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Default parameters for all tasks
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
}

with DAG(
    dag_id="carbon_resource_test_dag",
    start_date=datetime(2026, 1, 1),
    schedule=None,
    catchup=False,
    default_args=default_args,
) as dag:

    # Example: T1 requires 2 cores, lasts 1h (simulated), must start within 12h
    t1 = BashOperator(
        task_id="T1",
        bash_command="sleep 10",
        params={
            "cores": 2, 
            "duration_h": 1, 
            "dag_deadline_h": 48, # The whole DAG must finish in 48h
            "following_tasks": [1, 1, 1, 1] # Estimate of how many hours of work follow this task (sum of T2-T9)
        }
    )

    # T2 is "heavy" and requires more resources
    t2 = BashOperator(
        task_id="T2",
        bash_command="sleep 30",
        params={
            "cores": 3, 
            "duration_h": 1, 
            "dag_deadline_h": 48, # The whole DAG must finish in 48h
            "following_tasks": [1, 1, 1]
        }
    )

    t3 = BashOperator(
        task_id="T3",
        bash_command="sleep 10",
        params={
            "cores": 2, 
            "duration_h": 1, 
            "dag_deadline_h": 48, # The whole DAG must finish in 48h
            "following_tasks": [1, 1, 1]
        }
    )

    t4 = BashOperator(
        task_id="T4",
        bash_command="sleep 10",
        params={
            "cores": 2, 
            "duration_h": 1, 
            "dag_deadline_h": 48, # The whole DAG must finish in 48h
            "following_tasks": [1, 1, 1]
        }
    )

    t5 = BashOperator(
        task_id="T5",
        bash_command="sleep 10",
        params={
            "cores": 2, 
            "duration_h": 1, 
            "dag_deadline_h": 48, # The whole DAG must finish in 48h
            "following_tasks": [1, 1]
        }
    )

    # ... Add similar params to T4 through T9 ...
    # For T4-T9, I'll use a loop-style or default to keep it clean:
    tasks = []
    for i in range(6, 10):
        tasks.append(BashOperator(
            task_id=f"T{i}",
            bash_command="sleep 10",
            params={
                "cores": 2, 
                "duration_h": 1, 
                "dag_deadline_h": 48, # The whole DAG must finish in 48h
                "following_tasks": [1] # Estimate of how many hours of work follow this task
            }
        ))

    # Reconstructing your dependencies
    t6, t7, t8, t9 = tasks[0], tasks[1], tasks[2], tasks[3]

    t1 >> [t2, t3, t4] >> t5 >> [t6, t7, t8] >> t9
    t1 >> t9
    t5 >> t9