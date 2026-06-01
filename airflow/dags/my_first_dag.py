from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime


def say_hello():
    print("Hello from Airflow!")


with DAG(
    dag_id="my_first_dag",
    start_date=datetime(2024, 1, 1),
    schedule="@daily",
    catchup=False,
) as dag:

    task1 = PythonOperator(
        task_id="python_task",
        python_callable=say_hello,
    )

    task2 = BashOperator(
        task_id="bash_task",
        bash_command="echo 'Airflow is running!'",
    )

    task1 >> task2
