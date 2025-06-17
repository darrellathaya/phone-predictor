import os
import json
import joblib
import pytest
import pandas as pd
from unittest import mock
from src.train_model import resolution_to_value, chipset_score, preprocess_data, train, MODEL_DIR


# === Fixtures & Setup ===
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'ram': [4, 6, 8],
        'storage': [64, 128, 256],
        'display_resolution': ['720p', '1080p', '2k+'],
        'chipset': ['snapdragon 888', 'apple a16', 'unknown'],
        'price_range': ['low', 'medium', 'high']
    })


# === Unit Tests for Helpers ===
def test_resolution_to_value():
    assert resolution_to_value("720p") == 720
    assert resolution_to_value("1080p") == 1080
    assert resolution_to_value("2k+") == 2000
    assert resolution_to_value("unknown") == 720  # default fallback


def test_chipset_score_known():
    assert chipset_score("Snapdragon 8 Gen 3") == 850
    assert chipset_score("apple A14") == 770
    assert chipset_score("Kirin") == 500


def test_chipset_score_unknown():
    assert chipset_score("intel unknown") == 400


def test_preprocess_data(sample_dataframe):
    X, y = preprocess_data(sample_dataframe)
    assert all(col in X.columns for col in ['ram', 'storage', 'display_resolution_cat', 'chipset_score'])
    assert isinstance(y, pd.Series)


# === Integration Test ===
@mock.patch("src.train_model.pd.read_csv")
@mock.patch("src.train_model.mlflow.start_run")
@mock.patch("src.train_model.mlflow.set_experiment")
@mock.patch("src.train_model.mlflow.log_param")
@mock.patch("src.train_model.mlflow.log_metric")
def test_train_function(mock_metric, mock_param, mock_experiment, mock_start_run, mock_read_csv, tmp_path):
    # Prepare fake data
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512, 64, 128, 256, 512],
        'display_resolution': ['720p'] * 8,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 2,
        'price_range': ['low', 'medium', 'high', 'low'] * 2
    })
    mock_read_csv.return_value = df

    # Run training
    train()

    # Check model file
    model_path = os.path.join(MODEL_DIR, "price_range_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    assert os.path.exists(model_path)
    assert os.path.exists(meta_path)

    # Check metadata content
    with open(meta_path) as f:
        meta = json.load(f)
    assert "best_model" in meta
    assert "chipset_list" in meta
    assert isinstance(meta["chipset_list"], list)


# === Negative Test for SMOTE ===
@mock.patch("src.train_model.SMOTE")
@mock.patch("src.train_model.pd.read_csv")
def test_smote_failure(mock_read_csv, mock_smote, capsys):
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512, 64, 128, 256, 512],
        'display_resolution': ['720p'] * 8,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 2,
        'price_range': ['low', 'medium', 'high', 'medium'] * 2
    })
    mock_read_csv.return_value = df
    mock_smote.side_effect = ValueError("Need at least 2 classes")
 
    train()

    captured = capsys.readouterr()
    assert "SMOTE error" in captured.out
