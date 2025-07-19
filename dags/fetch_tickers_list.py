from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yaml
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data_extractor.tickers import get_sp_500_tickers_from_wikipedia
from src.utils.config import load_config

CONFIG = load_config("./configs/main.yaml")

# Get the tickers wikipedia url
URL = CONFIG['metadata']['tickers_wikipedia_url']

# Get the tickers list filename pattern from the config
tickers_list_filename_pattern = CONFIG['metadata']['tickers_list_filename']

# Get the tickers list save path from the config
tickers_list_save_path_str = CONFIG['metadata']['tickers_list_save_path']


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
        op_args=[URL, tickers_list_save_path_str, tickers_list_filename_pattern]
    )

