from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_extractor.tickers import get_sp_500_tickers_from_wikipedia
from src.utils.config import load_config

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
CONFIG = load_config("./configs/main.yaml")


# Default args for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}


# Define DAG
with DAG(
    dag_id='weekly_sp500_ticker_fetch',
    default_args=default_args,
    description='Fetch S&P 500 tickers weekly and save to JSON',
    start_date=datetime(2024, 1, 1),
    schedule_interval='@weekly',
    catchup=False,
    tags=['tickers', 'sp500', 'weekly']
) as dag:

    fetch_and_save_tickers = PythonOperator(
        task_id='fetch_tickers_from_wikipedia',
        python_callable=get_sp_500_tickers_from_wikipedia,
        op_args=[URL, CONFIG]
    )

