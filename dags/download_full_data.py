from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import yaml
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.data_extractor.tickers import get_most_recent_tickers_file
from src.data_extractor.stocks import download_and_save_stock_data


CONFIG = load_config("./configs/main.yaml")
tickers_folder = CONFIG['metadata']['tickers_list_save_path']
MOST_RECENT_TICKER_FILE = get_most_recent_tickers_file(tickers_folder, CONFIG['metadata']['tickers_list_filename'])

with open(MOST_RECENT_TICKER_FILE, "r") as f:
    tickers_list = json.load(f)

# Default args for the DAG
default_args = {
    'owner': 'airflow',
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}


# Define DAG
with DAG(
    dag_id='download_full_data',
    default_args=default_args,
    description='Download full stock data from given start date to today',
    start_date=datetime(2024, 1, 1),
    schedule_interval=None,
    catchup=False,
    tags=['stock', 'download', 'once']
) as dag:

    fetch_and_save_tickers = PythonOperator(
        task_id='download_full_data',
        python_callable=download_and_save_stock_data,
        op_kwargs={
            'tickers_list': tickers_list,
            'stock_data_start_date': CONFIG['stock']['stock_data_start_date'],
            'exclude_tickers': set(CONFIG['metadata']['exclude_tickers']),
            'data_save_path': CONFIG['stock']['stock_data_save_path'],
        }
    )