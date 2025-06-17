import pytest
from fastapi.testclient import TestClient
from app.main import app, train
import json
import pandas as pd
from unittest.mock import patch, mock_open, MagicMock

client = TestClient(app)

def test_get_index_success():
    response = client.get("/")
    assert response.status_code == 200
    assert "chipset_list" in response.text

@patch("app.main.os.path.exists", return_value=False)
@patch("app.main.load_metadata", return_value={"chipset_list": ["Snapdragon 855"], "resolution_list": ["720p", "1080p", "2k+"], "available_trained_models": [], "label_mapping": {}})
def test_model_file_missing(mock_metadata, mock_exists):
    response = client.post("/", data={
        "ram": 2048,
        "storage": 128,
        "display_resolution": "1080p",
        "chipset": "Snapdragon 855",
        "selected_model_name": "RandomForest"
    })
    assert response.status_code == 200
    assert "tidak ditemukan" in response.text or "not found" in response.text

@patch("builtins.open", side_effect=FileNotFoundError("meta.json missing"))
@patch("app.main.os.path.exists", return_value=False)
def test_get_index_file_not_found(mock_exists, mock_open):
    response = client.get("/")
    assert response.status_code == 200
    assert "Gagal memuat metadata awal" in response.text or "meta.json" in response.text

@patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
    "chipset_list": ["Snapdragon 855"],
    "resolution_list": ["720p", "1080p", "2k+"],
    "available_trained_models": ["XGBoost"],
    "label_mapping": {"0": "Low"}
}))
@patch("app.main.os.path.exists", return_value=True)
def test_fallback_selected_model(mock_exists, mock_file):
    response = client.get("/")
    assert response.status_code == 200
    assert "XGBoost" in response.text

@patch("builtins.print")  # suppress logs
@patch("builtins.open", new_callable=mock_open)
@patch("pandas.read_csv")
@patch("app.main.mlflow.start_run")
@patch("app.main.setup_experiment", return_value="123")
def test_train_runs(mock_setup, mock_run, mock_read_csv, mock_open_file, mock_print):
    mock_df = pd.DataFrame({
        "ram": [2048, 4096],
        "storage": [64, 128],
        "display_resolution": ["1080p", "1080p"],
        "chipset": ["Snapdragon 855", "Snapdragon 855"],
        "price_range": ["Low", "Mid"]
    })
    mock_read_csv.return_value = mock_df
    train()  # Ensure it runs without error
