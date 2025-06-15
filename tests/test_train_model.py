import pytest
import pandas as pd
from unittest.mock import patch
from src.train_model import (
    train,
    chipset_score,
    preprocess_data,
    resolution_to_value
)

@patch("src.train_model.pd.read_csv")
@patch("src.train_model.joblib.dump")
@patch("src.train_model.mlflow.log_param")
@patch("src.train_model.mlflow.log_metric")
@patch("src.train_model.mlflow.start_run")
@patch("src.train_model.mlflow.set_experiment")
@patch("src.train_model.MlflowClient")
def test_resolution_to_value(mock_client, mock_set_experiment, mock_start_run, mock_log_metric, mock_log_param, mock_dump, mock_read_csv):
    assert resolution_to_value("720p") == 720
    assert resolution_to_value("1080p") == 1080
    assert resolution_to_value("2k+") == 2000
    assert resolution_to_value("unknown") == 720  # default fallback

def test_chipset_score():
    assert chipset_score("Snapdragon 8 Gen 3") == 850
    assert chipset_score("Apple A14") == 770
    assert chipset_score("UnknownChipset") == 400  # default fallback

def test_preprocess_data():
    # Sample data
    data = {
        "ram": [4, 6],
        "storage": [64, 128],
        "display_resolution": ["720p", "1080p"],
        "chipset": ["Snapdragon 8 Gen 3", "Apple A14"],
        "price_range": ["Low", "High"]
    }
    df = pd.DataFrame(data)

    X, y = preprocess_data(df.copy())

    assert X.shape == (2, 4)
    assert y.tolist() == ["Low", "High"]
    assert "display_resolution_cat" in X.columns
    assert "chipset_score" in X.columns