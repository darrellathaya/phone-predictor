import os
import json
import shutil
import tempfile
import pytest
import warnings
import joblib
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from unittest.mock import patch, Mock, MagicMock
from fastapi.testclient import TestClient
from sklearn.ensemble import RandomForestClassifier
from app.main import app, resolution_to_value, chipset_score, preprocess_data, get_models, setup_experiment, train, MODEL_DIR

# Silence XGBoost warning if needed
warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

client = TestClient(app)

# -----------------------
# Endpoint Tests
# -----------------------

def test_get_index_success(tmp_path):
    meta = {
        "chipset_list": ["Snapdragon 888", "Apple A14"],
        "resolution_list": ["720p", "1080p", "2k+"],
        "available_trained_models": ["RandomForest", "SVM"],
        "best_model_name": "RandomForest"
    }
    os.makedirs("models", exist_ok=True)
    with open("models/meta.json", "w") as f:
        json.dump(meta, f)
    response = client.get("/")
    assert response.status_code == 200
    assert "Snapdragon 888" in response.text


def test_get_index_file_not_found(monkeypatch):
    if os.path.exists("models/meta.json"):
        os.remove("models/meta.json")
    response = client.get("/")
    assert response.status_code == 200
    assert "Gagal memuat metadata awal" in response.text


def test_post_predict_success(tmp_path):
    os.makedirs("models", exist_ok=True)
    meta = {
        "chipset_list": ["Snapdragon 888"],
        "resolution_list": ["1080p"],
        "available_trained_models": ["RandomForest"],
        "best_model_name": "RandomForest",
        "best_model_overall_f1_score": 0.9,
        "model_f1_scores": {"RandomForest": 0.9},
        "label_mapping": {"0": "Low", "1": "Mid", "2": "High"}
    }
    with open("models/meta.json", "w") as f:
        json.dump(meta, f)

    from sklearn.ensemble import RandomForestClassifier
    # Train the dummy model with 4 features to match the server's preprocessing
    dummy_model = RandomForestClassifier()
    dummy_model.fit([[1, 2, 3, 4]], [1])  # One sample with 4 features  # Corrected line: Train with 4 features

    joblib.dump(dummy_model, "models/RandomForest.pkl")

    response = client.post("/", data={
        "ram": 2048,
        "storage": 128,
        "display_resolution": "1080p",
        "chipset": "Snapdragon 888",
        "selected_model_name": "RandomForest"
    })

    # Debugging: Print response details
    print("Response status code:", response.status_code)
    print("Response HTML:", response.text)

    # Assertions
    assert response.status_code == 200
    soup = BeautifulSoup(response.text, "html.parser")
    prediction = soup.find("h5", class_="card-title")
    assert prediction is not None, "Expected <h5> element with class 'card-title' not found in response."
    assert any(label in prediction.text for label in ["Low", "Mid", "High"]), "Unexpected prediction value."


def test_post_predict_missing_model_file():
    if os.path.exists("models/RandomForest.pkl"):
        os.remove("models/RandomForest.pkl")
    meta = {
        "chipset_list": ["Snapdragon 888"],
        "resolution_list": ["1080p"],
        "available_trained_models": ["RandomForest"],
        "best_model_name": "RandomForest",
        "label_mapping": {"0": "Low"}
    }
    with open("models/meta.json", "w") as f:
        json.dump(meta, f)
    response = client.post("/", data={
        "ram": 2048,
        "storage": 128,
        "display_resolution": "1080p",
        "chipset": "Snapdragon 888",
        "selected_model_name": "RandomForest"
    })
    assert response.status_code == 200
    assert "tidak ditemukan" in response.text


# -----------------------
# Utility Function Tests
# -----------------------

def test_resolution_to_value():
    assert resolution_to_value("720p") == 720
    assert resolution_to_value("1080p") == 1080
    assert resolution_to_value("2k+") == 2000
    assert resolution_to_value("unknown") == 720


def test_chipset_score():
    assert chipset_score("Snapdragon 888") == 800
    assert chipset_score("Tensor G3") == 800
    assert chipset_score("Apple A14") == 770
    assert chipset_score("UnknownChipset") == 400


def test_preprocess_data():
    df = pd.DataFrame({
        "ram": [2048],
        "storage": [64],
        "display_resolution": ["1080p"],
        "chipset": ["Snapdragon 888"],
        "price_range": ["Mid"]
    })
    X, y = preprocess_data(df)
    assert "chipset_score" in X.columns
    assert "display_resolution_cat" in X.columns
    assert y.iloc[0] == "Mid"


def test_get_models():
    models = get_models()
    assert "RandomForest" in models
    assert "SVM" in models


# -----------------------
# MLflow Tests
# -----------------------

@patch("app.main.MlflowClient")
def test_setup_experiment(mock_mlflow):
    mock_client = MagicMock()
    mock_client.get_experiment_by_name.return_value = None
    mock_mlflow.return_value = mock_client
    result = setup_experiment("TestExperiment")
    assert mock_client.create_experiment.called

# -----------------------
# Train() Coverage Test
# -----------------------

@pytest.fixture
def sample_training_data(tmp_path):
    df = pd.DataFrame({
        "chipset": ["Snapdragon 888"] * 20,
        "resolution": ["1080p"] * 20,
        "price_range": ["Low"] * 10 + ["Mid"] * 5 + ["High"] * 5,
        "cores": [4]*20,
        "memory": [4]*20,
        "battery": [3000]*20,
        "clock_speed": [2.0]*20
    })
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    os.environ["DATA_PATH"] = str(data_path)

    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MODEL_DIR"] = str(model_dir)

    yield tmp_path

    shutil.rmtree(tmp_path)

@patch("app.main.mlflow.start_run")
@patch("app.main.mlflow.log_param")
@patch("app.main.mlflow.log_metric")
@patch("app.main.setup_experiment", return_value="test_exp_id")
@patch("app.main.joblib.dump")
def test_train_function_executes_to_end(
    mock_dump,
    mock_setup_experiment,
    mock_log_metric,
    mock_log_param,
    mock_start_run,
    sample_training_data,
    capsys
):
    train()
    captured = capsys.readouterr()
    assert "Training complete (from main.py)!" in captured.out
