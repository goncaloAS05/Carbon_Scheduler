from airflow import DAG
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
from datetime import datetime, timedelta
import datetime as dt
import os

# dur = hours (standardized execution time)
WORKFLOW_METADATA = [
    {'id': 'T1', 'dur': 1, 'cores': 2, 'depends_on': [], "output_size_gb": 15.5},
    {'id': 'T2', 'dur': 1, 'cores': 1, 'depends_on': ['T1'], "output_size_gb": 5.0},
    {'id': 'T3', 'dur': 1, 'cores': 1, 'depends_on': ['T1'], "output_size_gb": 5.0},
    {'id': 'T4', 'dur': 1, 'cores': 1, 'depends_on': ['T1'], "output_size_gb": 5.0},
    {'id': 'T5', 'dur': 1, 'cores': 2, 'depends_on': ['T2', 'T3', 'T4'], "output_size_gb": 15.0}, # Nó de junção (Merge)
    {'id': 'T6', 'dur': 1, 'cores': 1, 'depends_on': ['T5'], "output_size_gb": 5.0},
    {'id': 'T7', 'dur': 1, 'cores': 1, 'depends_on': ['T5'], "output_size_gb": 5.0},
    {'id': 'T8', 'dur': 1, 'cores': 1, 'depends_on': ['T5'], "output_size_gb": 5.0},
    {'id': 'T9', 'dur': 1, 'cores': 2, 'depends_on': ['T6', 'T7', 'T8'], "output_size_gb": 15.0}, # Nó de junção final
]

with DAG(
    dag_id="carbon_scheduler_k8s",
    start_date=datetime(2026, 4, 22), 
    schedule=None, 
    catchup=False,
    params={
        'deadline_iso': "2026-06-03T00:00:00" if os.getenv("AIRFLOW_IS_PARSING") else (dt.datetime.now() + dt.timedelta(days=1)).isoformat(), 
        'workflow_structure': WORKFLOW_METADATA, 
        'input_data_gb': 1, 
        'sla_confidence': 90
    }
    ) as dag:

    # Helper to avoid repetitive code
    def create_pod_task(tid, name, sleep_time):
        return KubernetesPodOperator(
            namespace="default",
            image="python:3.11",
            cmds=["bash", "-cx"],
            arguments=[f"sleep {sleep_time}"],
            labels={"app": "carbon-scheduler"},
            name=name,
            task_id=tid,
            get_logs=True,
        )

    t1 = create_pod_task("T1", "collect-data", 10)
    t2 = create_pod_task("T2", "process-a", 5)
    t3 = create_pod_task("T3", "process-b", 5)
    t4 = create_pod_task("T4", "process-c", 10)
    t5 = create_pod_task("T5", "merge-data", 5)
    t6 = create_pod_task("T6", "report-a", 10)
    t7 = create_pod_task("T7", "report-b", 10)
    t8 = create_pod_task("T8", "report-c", 10)
    t9 = create_pod_task("T9", "finalize", 10)

    # Dependencies
    t1 >> [t2, t3, t4] >> t5 >> [t6, t7, t8] >> t9