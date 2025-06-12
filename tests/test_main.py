import pytest
import pandas as pd
from unittest.mock import patch, MagicMock
from src.train_model import train

@patch("src.train_model.pd.read_csv")
@patch("src.train_model.joblib.dump")
@patch("src.train_model.mlflow.log_param")
@patch("src.train_model.mlflow.log_metric")
@patch("src.train_model.mlflow.start_run")
@patch("src.train_model.mlflow.set_experiment")
@patch("src.train_model.MlflowClient")
def test_train_function_runs(
    mock_mlflow_client,
    mock_set_experiment,
    mock_start_run,
    mock_log_metric,
    mock_log_param,
    mock_joblib_dump,
    mock_read_csv
):
    # Fake input data
    data = {
        "ram": [4, 6, 8, 4],
        "storage": [64, 128, 256, 64],
        "display_resolution": ["720p", "1080p", "2k+", "720p"],
        "chipset": ["Snapdragon 8 Gen 3", "Apple A14", "Helio G99", "Tensor G3"],
        "price_range": ["Low", "Medium", "High", "Low"]
    }
    mock_read_csv.return_value = pd.DataFrame(data)

    # Simulate MLflow experiment setup
    mock_client = MagicMock()
    mock_mlflow_client.return_value = mock_client
    mock_client.get_experiment_by_name.return_value = None
    mock_client.create_experiment.return_value = "123"

    # Act
    train()


    # Assert MLflow and joblib interactions
    assert mock_read_csv.called
    assert mock_set_experiment.called
    assert mock_start_run.called
    assert mock_joblib_dump.called
