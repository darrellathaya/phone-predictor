import os
import json
import pytest
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, mock_open, MagicMock

from app.main import (
    app, resolution_to_value, chipset_score, preprocess_data,
    get_models, setup_experiment
)

client = TestClient(app)

# === Unit Tests ===

def test_resolution_to_value():
    assert resolution_to_value("720p") == 720
    assert resolution_to_value("1080p") == 1080
    assert resolution_to_value("2k+") == 2000
    assert resolution_to_value("unknown") == 720

def test_chipset_score():
    assert chipset_score("Snapdragon 8 Gen 2") == 820
    assert chipset_score("Apple A15") == 800
    assert chipset_score("UnknownChip") == 400

def test_preprocess_data():
    df = pd.DataFrame({
        "ram": [2048],
        "storage": [128],
        "display_resolution": ["1080p"],
        "chipset": ["Snapdragon 855"],
        "price_range": ["Mid"]
    })
    X, y = preprocess_data(df)
    assert X.shape == (1, 4)
    assert y.iloc[0] == "Mid"

def test_get_models():
    models = get_models()
    assert "RandomForest" in models
    assert "SVM" in models
    assert "XGBoost" in models

@patch("app.main.MlflowClient")
def test_setup_experiment(mock_client_class):
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "123"
    result = setup_experiment("test")
    assert result == "123"

# === FastAPI Integration Tests ===

@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
    "chipset_list": ["Snapdragon 855"],
    "resolution_list": ["720p", "1080p", "2k+"],
    "available_trained_models": ["RandomForest"],
    "best_model_name": "RandomForest",
    "best_model_overall_f1_score": 0.88,
    "model_f1_scores": {"RandomForest": 0.88},
    "label_mapping": {"0": "Low"}
}))
@patch("app.main.os.path.exists", return_value=True)
@patch("app.main.joblib.load")
def test_post_predict(mock_joblib_load, mock_exists, mock_file):
    mock_model = MagicMock()
    mock_model.predict.return_value = [0]
    mock_joblib_load.return_value = mock_model

    response = client.post("/", data={
        "ram": 2048,
        "storage": 128,
        "display_resolution": "1080p",
        "chipset": "Snapdragon 855",
        "selected_model_name": "RandomForest"
    })
    assert response.status_code == 200
    assert "Low" in response.text

@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
    "chipset_list": ["Snapdragon 855"],
    "resolution_list": ["720p"],
    "available_trained_models": ["SVM"],
    "best_model_name": "SVM"
}))
def test_get_index_success(mock_open_file):
    response = client.get("/")
    assert response.status_code == 200
    assert "Snapdragon 855" in response.text

@patch("builtins.open", side_effect=FileNotFoundError("missing meta"))
def test_get_index_file_not_found(mock_open_file):
    response = client.get("/")
    assert response.status_code == 200
    assert "Gagal memuat metadata awal" in response.text
