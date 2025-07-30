from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# DAG metadata
DAG_ID = "prepare_data_for_training"
DAG_DESCRIPTION = "Prepare the stock data for training and save to the specified directory."

def create_dag() -> DAG:
    """
    Creates an Airflow DAG that prepares the stock data for training and saves to the specified directory

    Returns:
        DAG: An Airflow DAG object.
    """
    # Lazy imports to avoid breaking DAG parsing
    from src.utils.config import load_config
    from src.training.preparation import prepare_data_for_training
    

    # Load configuration
    config = load_config()


    # Define the default arguments
    default_args = {
        "owner": "airflow",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    }

    dag = DAG(
        dag_id=DAG_ID,
        description=DAG_DESCRIPTION,
        default_args=default_args,
        start_date=datetime(2024, 1, 1),
        schedule_interval=None,  
        catchup=False,
        tags=["stock", "process", "prepare for training"]
    )

    with dag:
        PythonOperator(
            task_id="prepare_data_for_training",
            python_callable=prepare_data_for_training,
            op_kwargs={
                "data_folder_path": config["stock"]["stock_data_save_path"],
                "target_column": config["train_config"]["target"]["target_column"],
                "n_lags": config["train_config"]["feature_engineering"]["n_lags"],
                "validation_size": config["train_config"]["train_test_split"]["validation_size"],
                "prediction_window": config["train_config"]["target"]["prediction_window"],
                "save_path": config["stock"]["stock_data_processed_save_path"],
            }
        )

    return dag

# Required for Airflow to detect the DAG
dag = create_dag()