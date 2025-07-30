from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# DAG metadata
DAG_ID = "download_full_data"
DAG_DESCRIPTION = "Download full stock data from given start date to today"

def create_dag() -> DAG:
    """
    Creates an Airflow DAG that downloads full historical stock data for all tickers.

    Returns:
        DAG: An Airflow DAG object.
    """
    # Lazy imports to avoid breaking DAG parsing
    import json
    from src.utils.config import load_config
    from src.data.tickers import get_most_recent_tickers_file
    from src.data.stocks import download_and_save_stock_data
    

    # Load configuration
    config = load_config()
    tickers_folder = config["tickers"]["tickers_list_save_path"]
    filename_pattern = config["tickers"]["tickers_list_filename"]
    stock_data_start_date = config["stock"]["stock_data_start_date"]
    exclude_tickers = set(config["tickers"]["exclude_tickers"])
    data_save_path = config["stock"]["stock_data_save_path"]

    most_recent_file = get_most_recent_tickers_file(tickers_folder, filename_pattern)

    with open(most_recent_file, "r") as f:
        tickers_list = json.load(f)

    default_args = {
        "owner": "airflow",
        "retries": 3,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        dag_id=DAG_ID,
        description=DAG_DESCRIPTION,
        default_args=default_args,
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,  # one-time or manually triggered
        catchup=False,
        tags=["stock", "download", "once"]
    )

    with dag:
        PythonOperator(
            task_id="download_full_data",
            python_callable=download_and_save_stock_data,
            op_kwargs={
                "tickers_list": tickers_list,
                "stock_data_start_date": stock_data_start_date,
                "exclude_tickers": exclude_tickers,
                "data_save_path": data_save_path
            }
        )

    return dag

# Required for Airflow to detect the DAG
dag = create_dag()