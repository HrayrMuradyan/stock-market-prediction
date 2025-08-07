from flask import Flask, render_template, abort
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import json
from pathlib import Path
from src.inference.models import get_most_recent_model
from src.utils.config import load_config

app = Flask(__name__)

config = load_config()
latest_model_in_the_config = config['eval_config']['general']['model_train_results_path']
models_folder = Path(config['train_config']['model']['save_path'])
results_save_path = Path(config['eval_config']['general']['results_save_path'])

INFERENCE_DIR = Path(config['deploy_config']['paths']['predictions_save_path'])
LATEST_MODEL = latest_model_in_the_config if latest_model_in_the_config else get_most_recent_model(config['train_config']['model']['save_path']).stem
MODELS_DIR = models_folder / LATEST_MODEL
RESULTS_DIR = results_save_path / LATEST_MODEL

# Run index function when the user visits homepage
@app.route("/")
def index():

    # Get all stock names
    models = get_model_names()

    # If there is at least one model folder, it calls the dashboard() function on the first model in the list
    # it redirects users immediately to the dashboard for the first model available.
    return dashboard(models[0]) if models else "No models found."

# When you visit /dashboard/something, it will call dashboard function with argument "something"
@app.route("/dashboard/<ticker_name>")
def dashboard(ticker_name):
    models = get_model_names()
    if ticker_name not in models:
        return render_template("not_found.html", model=ticker_name, all_models=models), 404

    model_path = MODELS_DIR / ticker_name
    results_path = RESULTS_DIR / ticker_name
    plots_path = results_path / "plots"
    inference_path = INFERENCE_DIR / ticker_name

    metrics = load_json(results_path / "metrics.json")
    params = load_json(model_path / "hyperparams.json")

    # Load plotly HTML strings
    plots = {}
    for f in sorted(plots_path.glob("*.html")):
        with f.open(encoding="utf-8") as plot_file:
            plots[f.stem] = plot_file.read()

    # Get the inference plot path
    inference_plot_path = inference_path / "real_time_inference.html"

    # Add the real-time plot as well
    with inference_plot_path.open(encoding="utf-8") as infer_file:
        plots[inference_plot_path.stem] = infer_file.read()

    # 🔄 Add JSON version of the plot if it exists
    json_plot_path = inference_path / "plot.json"
    if json_plot_path.exists():
        with json_plot_path.open(encoding="utf-8") as json_file:
            plots["real_time_inference_json"] = json.load(json_file)

    return render_template("dashboard.html",
                           model=ticker_name,
                           latest_model=LATEST_MODEL,
                           all_models=models,
                           metrics=metrics,
                           params=params,
                           plots=plots)

def get_model_names():
    return sorted([
        d.name for d in MODELS_DIR.iterdir() if d.is_dir()
    ])

def load_json(path):
    path = Path(path) 
    if path.exists():
        with path.open() as f:
            return json.load(f)
    return {}

if __name__ == "__main__":
    app.run(debug=True)