import pytest
import numpy as np
import pandas as pd
from unittest import mock
from fastapi.testclient import TestClient
from app.main import (
    app,
    resolution_to_value,
    chipset_score,
    preprocess_data,
    get_models,
    setup_experiment,
)

client = TestClient(app)

# ---------- Fixtures ----------

@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'ram': [2048],
        'storage': [128],
        'display_resolution': ['1080p'],
        'chipset': ['Snapdragon 888'],
        'price_range': ['low']
    })

# ---------- Unit Tests: Utility Functions ----------

def test_resolution_to_value():
    assert resolution_to_value("720p") == 720
    assert resolution_to_value("1080p") == 1080
    assert resolution_to_value("2k+") == 2000
    assert resolution_to_value("unknown") == 720  # default case

def test_chipset_score():
    assert chipset_score("Snapdragon 888") == 800
    assert chipset_score("Apple A14") == 770
    assert chipset_score("Unknown Chipset") == 400

# ---------- Unit Tests: Data Preprocessing ----------

def test_preprocess_data(sample_dataframe):
    X, y = preprocess_data(sample_dataframe)
    assert X.shape == (1, 4)
    assert y.iloc[0] == "low"
    assert 'chipset_score' in X.columns
    assert 'display_resolution_cat' in X.columns

# ---------- Unit Tests: Model Utilities ----------

@mock.patch("app.main.mlflow.MlflowClient")  # Correct patch path
def test_setup_experiment(mock_client):
    mock_instance = mock_client.return_value
    mock_instance.get_experiment_by_name.return_value = None
    mock_instance.create_experiment.return_value = "mock_exp_id"
    result = setup_experiment("MyTestExp")
    assert result == "mock_exp_id"

# ---------- Integration Test: FastAPI Endpoint ----------

@mock.patch("builtins.open", new_callable=mock.mock_open, read_data='''{
    "chipset_list": ["Snapdragon 888"],
    "resolution_list": ["1080p"],
    "available_trained_models": ["SVM"],
    "best_model_name": "SVM",
    "label_mapping": {"0": "low"},
    "model_f1_scores": {"SVM": 0.95},
    "best_model_overall_f1_score": 0.95
}''')
@mock.patch("os.path.exists", return_value=True)
@mock.patch("joblib.load")
def test_predict_price(mock_joblib_load, mock_exists, mock_open):
    mock_model = mock.Mock()
    mock_model.predict.return_value = [0]
    mock_joblib_load.return_value = mock_model

    response = client.post("/", data={
        "ram": 2048,
        "storage": 128,
        "display_resolution": "1080p",
        "chipset": "Snapdragon 888",
        "selected_model_name": "SVM"
    })

    assert response.status_code == 200
    json_response = response.json()
    assert "available_trained_models" in json_response
    assert "best_model_name" in json_response
    assert "chipset_list" in json_response
    assert "resolution_list" in json_response
    assert "label_mapping" in json_response
    assert "model_f1_scores" in json_response
    assert "best_model_overall_f1_score" in json_response
