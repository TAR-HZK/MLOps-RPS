from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
sys.path.append('/home/tarhzk/Documents/FAST/Semester 8-9/MLOps/MLOps-Project')

from src.fetch_data import fetch_weather_data as run_extract
from src.quality_check import quality_check
from etl.transform_weather import run_transform
from utils.dvc_ops import dvc_add_and_push
from src.train_model import train

default_args = {
    "start_date": datetime(2024,1,1),
}

with DAG(
    "rps_openweather_etl_train",
    schedule_interval="@daily",
    catchup=False,
    default_args=default_args,
):

    extract = PythonOperator(
        task_id="extract",
        python_callable=run_extract
    )

    quality = PythonOperator(
        task_id="quality",
        python_callable=quality_check,
        op_args=["{{ ti.xcom_pull('extract') }}"]
    )

    transform = PythonOperator(
        task_id="transform",
        python_callable=run_transform,
        op_args=["{{ ti.xcom_pull('extract') }}"]
    )

    dvc_track = PythonOperator(
        task_id="dvc_track",
        python_callable=dvc_add_and_push,
        op_args=["{{ ti.xcom_pull('transform') }}"]
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=train,
        op_args=["{{ ti.xcom_pull('transform') }}"]
    )

    extract >> quality >> transform >> dvc_track >> train_model

