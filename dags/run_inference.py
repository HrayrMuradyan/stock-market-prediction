from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# DAG metadata
DAG_ID = "run_inference"
DAG_DESCRIPTION = "Predict future data from the last row saved and trained model."

def create_dag() -> DAG:
    """
    Creates an Airflow DAG that predicts the future data from the last row saved and trained model.

    Returns
    -------
    DAG
        The DAG object to be registered by Airflow.
    """
    # Lazy imports to avoid breaking DAG parsing
    from src.inference.predictions import generate_and_save_future_predictions

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
        schedule_interval=None,  
        catchup=False,
        tags=["run", "inference"]
    )

    with dag:
        PythonOperator(
            task_id="run_inference",
            python_callable=generate_and_save_future_predictions,
        )

    return dag

# Required for Airflow to detect the DAG
dag = create_dag()