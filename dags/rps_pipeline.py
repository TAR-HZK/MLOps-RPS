# Airflow-safe DAG with MLflow + DVC integration
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import os
import subprocess

default_args = {
    "owner": "astro",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

MLFLOW_ARTIFACTS = os.path.join(os.path.dirname(__file__), "..", "include", "mlflow_artifacts")
os.makedirs(MLFLOW_ARTIFACTS, exist_ok=True)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "include", "data", "processed")

with DAG(
    dag_id="rps_pipeline",
    default_args=default_args,
    start_date=datetime(2025, 11, 29),
    schedule="@hourly",
    catchup=False,
    tags=["mlops", "rps"],
) as dag:

    # --------------------
    # Task 1: Fetch Data
    # --------------------
    def fetch_data_task():
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from fetch_data import fetch_weather_data
        return fetch_weather_data()

    fetch_task = PythonOperator(
        task_id="fetch_live_data",
        python_callable=fetch_data_task,
    )

    # --------------------
    # Task 2: Quality Check
    # --------------------
    def quality_check_task(ti):
        raw_file = ti.xcom_pull(task_ids="fetch_live_data")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from quality_check import check_quality
        check_quality(raw_file)

    quality_task = PythonOperator(
        task_id="quality_check",
        python_callable=quality_check_task,
    )

    # --------------------
    # Task 3: Preprocess + DVC
    # --------------------
    def preprocess_task(ti):
        raw_file = ti.xcom_pull(task_ids="fetch_live_data")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from preprocess import preprocess

        # Preprocess
        processed_file = preprocess(raw_file)

        # --------------------
        # DVC Versioning
        # --------------------
        # Only add if processed_dir has files
        if os.listdir(PROCESSED_DIR):
            # Track processed data with DVC
            subprocess.run(["dvc", "add", PROCESSED_DIR], check=True)

            # Stage DVC metadata in Git
            subprocess.run(["git", "add", f"{PROCESSED_DIR}.dvc", ".gitignore"], check=True)

            # Commit to Git (skip if nothing to commit)
            subprocess.run(
                ["git", "commit", "-m", "Add new processed data via Airflow DAG"], 
                check=False
            )

            # Push to DVC remote
            subprocess.run(["dvc", "push"], check=True)

        return processed_file

    preprocess_task_op = PythonOperator(
        task_id="preprocess",
        python_callable=preprocess_task,
    )

    # --------------------
    # Task 4: Train
    # --------------------
    def train_task(ti):
        processed_file = ti.xcom_pull(task_ids="preprocess")
        import sys
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        from train_model import train

        train(processed_file)

    train_task_op = PythonOperator(
        task_id="train_model",
        python_callable=train_task,
    )

    # --------------------
    # DAG Dependencies
    # --------------------
    fetch_task >> quality_task >> preprocess_task_op >> train_task_op
