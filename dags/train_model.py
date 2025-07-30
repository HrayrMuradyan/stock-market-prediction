from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))



# DAG metadata
DAG_ID = "train_model"
DAG_DESCRIPTION = "Train a neural network model on processed data."

def create_dag() -> DAG:
    """
    Creates an Airflow DAG that trains a neural network model on processed data.

    Returns:
        DAG: An Airflow DAG object.
    """
    # Lazy imports to avoid breaking DAG parsing
    from src.utils.config import load_config
    from src.training.training_pipeline import train_model
    import src.training.models as models
    import src.training.loss_functions as loss_functions
    from src.data.stocks import read_processed_stock_data
    

    # Load configuration
    config = load_config()

    model_architecture_str = config["train_config"]["model"]["architecture"]
    model_architecture = getattr(models, model_architecture_str)

    X_train, X_valid, y_train, y_valid = read_processed_stock_data(config["stock"]["stock_data_processed_save_path"])

    loss_fn_str = config["train_config"]["optimization"]["loss_fn"]
    loss_fn = getattr(loss_functions, loss_fn_str)

    prediction_window=config["train_config"]["target"]["prediction_window"]
    batch_size=config["train_config"]["optimization"]["batch_size"]
    n_epochs=config["train_config"]["optimization"]["n_epochs"]
    learning_rate=config["train_config"]["optimization"]["learning_rate"]
    shuffle=config["train_config"]["optimization"]["shuffle"]
    model_params=config["train_config"]["model"]["model_params"]
    device=config["train_config"]["general"]["device"]
    save_path=config["train_config"]["model"]["save_path"]


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
        tags=["train", "neural network"]
    )

    with dag:
        PythonOperator(
            task_id="train_model",
            python_callable=train_model,
            op_kwargs={
                "model_architecture": model_architecture, 
                "X_train": X_train, 
                "y_train": y_train, 
                "X_valid": X_valid, 
                "y_valid": y_valid, 
                "loss_fn": loss_fn, 
                "prediction_window": prediction_window, 
                "batch_size": batch_size, 
                "n_epochs": n_epochs, 
                "learning_rate": learning_rate, 
                "shuffle": shuffle, 
                "model_params": model_params, 
                "device": device, 
                "save_path": save_path, 
            }
        )

    return dag

# Required for Airflow to detect the DAG
dag = create_dag()