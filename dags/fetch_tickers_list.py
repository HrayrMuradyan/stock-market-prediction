from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Constants
DAG_ID = "weekly_sp500_ticker_fetch"
DAG_DESCRIPTION = "Fetch S&P 500 tickers weekly and save to JSON"

def create_dag() -> DAG:
    """
    Creates an Airflow DAG that fetches the S&P 500 tickers from Wikipedia weekly
    and saves them to a JSON file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        DAG: An Airflow DAG object.
    """
    from src.utils.config import load_config
    from src.data_extractor.tickers import get_sp_500_tickers_from_wikipedia

    # Load configuration
    config = load_config()
    url = config["metadata"]["tickers_wikipedia_url"]
    save_path = config["metadata"]["tickers_list_save_path"]
    filename_pattern = config["metadata"]["tickers_list_filename"]

    # DAG default arguments
    default_args = {
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    # Define DAG
    dag = DAG(
        dag_id=DAG_ID,
        description=DAG_DESCRIPTION,
        default_args=default_args,
        start_date=datetime(2024, 1, 1),
        schedule_interval="@weekly",
        catchup=False,
        tags=["tickers", "sp500", "weekly"]
    )

    with dag:
        PythonOperator(
            task_id="fetch_tickers_from_wikipedia",
            python_callable=get_sp_500_tickers_from_wikipedia,
            op_args=[url, save_path, filename_pattern],
        )

    return dag

# Get the dag object
dag = create_dag()

