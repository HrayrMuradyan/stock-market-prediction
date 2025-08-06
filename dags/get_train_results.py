from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# DAG metadata
DAG_ID = "get_train_results"
DAG_DESCRIPTION = "Get train results, evaluation metrics and plots."

def create_dag() -> DAG:
    """
    Creates an Airflow DAG that takes the recently trained model and gets it's results.

    Returns
    -------
    DAG
        The DAG object to be registered by Airflow.
    """
    # Lazy imports to avoid breaking DAG parsing
    from src.evaluation.results import get_train_results

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
        tags=["results", "evaluation"]
    )

    with dag:
        PythonOperator(
            task_id="get_train_results",
            python_callable=get_train_results,
        )

    return dag

# Required for Airflow to detect the DAG
dag = create_dag()