import pytest
import pandas as pd
from unittest.mock import patch
from app.main import train

@patch("app.main.pd.read_csv")
@patch("app.main.joblib.dump")
@patch("app.main.mlflow.log_param")
@patch("app.main.mlflow.log_metric")
@patch("app.main.mlflow.start_run")
@patch("app.main.mlflow.set_experiment")
@patch("app.main.MlflowClient")
def test_train_function_runs_main_script(
    mock_mlflow_client,
    mock_set_experiment,
    mock_start_run,
    mock_log_metric,
    mock_log_param,
    mock_joblib_dump,
    mock_read_csv
):
    # Mock dataset
    mock_read_csv.return_value = pd.DataFrame({
        "ram": [4, 6, 8, 4, 12, 8],
        "storage": [64, 128, 256, 64, 512, 256],
        "display_resolution": ["720p", "1080p", "2k+", "720p", "1080p", "2k+"],
        "chipset": ["Snapdragon 8 Gen 3", "Apple A14", "Helio G99", "Tensor G3", "Snapdragon 8 Gen 2", "Dimensity 9200"],
        "price_range": ["Low", "Medium", "High", "Low", "Medium", "High"]
    })

    mock_client = mock_mlflow_client.return_value
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "123"

    train()

    mock_read_csv.assert_called_once()
    mock_set_experiment.assert_called_once_with("PhonePricePrediction")
    assert mock_start_run.call_count == 2
    mock_log_param.assert_called()
    mock_log_metric.assert_called()
    mock_joblib_dump.assert_called()