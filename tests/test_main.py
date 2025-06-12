import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.train_model import train

@patch("src.train_model.MlflowClient")
@patch("src.train_model.mlflow.set_experiment")
@patch("src.train_model.mlflow.start_run")
@patch("src.train_model.mlflow.log_metric")
@patch("src.train_model.mlflow.log_param")
@patch("src.train_model.joblib.dump")
@patch("src.train_model.pd.read_csv")
def test_train_function_runs(
    mock_read_csv,
    mock_joblib_dump,
    mock_log_param,
    mock_log_metric,
    mock_start_run,
    mock_set_experiment,
    mock_mlflow_client
):
    # --- Arrange ---
    # Simulate input DataFrame with all three classes
    mock_read_csv.return_value = pd.DataFrame({
        "ram": [4, 6, 8, 4, 12, 8],
        "storage": [64, 128, 256, 64, 512, 256],
        "display_resolution": ["720p", "1080p", "2k+", "720p", "1080p", "2k+"],
        "chipset": ["Snapdragon 8 Gen 3", "Apple A14", "Helio G99", "Tensor G3", "Snapdragon 8 Gen 2", "Dimensity 9200"],
        "price_range": ["Low", "Medium", "High", "Low", "Medium", "High"]
    })

    # Setup mock MLflow client
    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "123"

    # --- Act ---
    train()

    # --- Assert ---
    mock_read_csv.assert_called_once()
    mock_set_experiment.assert_called_once_with("PhonePricePrediction")
    assert mock_start_run.called, "MLflow run not started"
    assert mock_log_param.called, "MLflow parameters not logged"
    assert mock_log_metric.called, "MLflow metrics not logged"
    assert mock_joblib_dump.called, "Model not saved with joblib"