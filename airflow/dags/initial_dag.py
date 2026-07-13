from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
from airflow.providers.cncf.kubernetes.operators.pod import KubernetesPodOperator
import os
import datetime as dt


# Default parameters for all tasks
WORKFLOW_METADATA = [
      {
        "id": "T1",
        "dur": 5,
        "cores": 1,
        "depends_on": [],
        "output_size_gb": 5.0
      },
      {
        "id": "T2",
        "dur": 3,
        "cores": 1,
        "depends_on": [
          "T1"
        ],
        "output_size_gb": 4.3
      },
      {
        "id": "T3",
        "dur": 1,
        "cores": 1,
        "depends_on": [
          "T1"
        ],
        "output_size_gb": 9.6
      },
      {
        "id": "T4",
        "dur": 2,
        "cores": 1,
        "depends_on": [
          "T2",
          "T3"
        ],
        "output_size_gb": 5.4
      },
      {
        "id": "T5",
        "dur": 3,
        "cores": 2,
        "depends_on": [
          "T4"
        ],
        "output_size_gb": 9.6
      },
      {
        "id": "T6",
        "dur": 3,
        "cores": 4,
        "depends_on": [
          "T4"
        ],
        "output_size_gb": 16.0
      },
      {
        "id": "T7",
        "dur": 2,
        "cores": 2,
        "depends_on": [
          "T5",
          "T6"
        ],
        "output_size_gb": 18.7
      }
    ]

with DAG(
    dag_id="initial_dag",
    start_date=datetime(2026, 7, 1), 
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
