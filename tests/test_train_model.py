import os
import json
import joblib
import pytest
import pandas as pd
from unittest import mock
from train_model import (
    resolution_to_value,
    chipset_score,
    preprocess_data,
    train,
    MODEL_DIR # Pastikan MODEL_DIR ini adalah string "models" atau path yang benar
)


# === Fixtures & Setup ===
@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512] * 3,
        'display_resolution': ['720p'] * 12,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 3,
        'price_range': ['low', 'medium', 'high'] * 4
    })


# === Unit Tests for Helper Functions ===
def test_resolution_to_value():
    assert resolution_to_value("720p") == 720
    assert resolution_to_value("1080p") == 1080
    assert resolution_to_value("2k+") == 2000
    assert resolution_to_value("unknown") == 720


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


# === Integration Test for Training Pipeline ===
@mock.patch("train_model.pd.read_csv")
@mock.patch("train_model.mlflow.start_run")
@mock.patch("train_model.mlflow.set_experiment")
@mock.patch("train_model.mlflow.log_param")
@mock.patch("train_model.mlflow.log_metric")
def test_train_function(mock_log_metric, mock_log_param, mock_set_experiment, mock_start_run, mock_read_csv):
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12], # Data yang cukup untuk SMOTE
        'storage': [64, 128, 256, 512] * 3,
        'display_resolution': ['720p'] * 12,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 3,
        'price_range': ['low', 'medium', 'high'] * 4 # Pastikan ada >1 sampel per kelas untuk SMOTE
    })
    mock_read_csv.return_value = df

    train()

    meta_path = os.path.join(MODEL_DIR, "meta.json")
    model_pkl_path = os.path.join(MODEL_DIR, "price_range_model.pkl")
    assert os.path.exists(model_pkl_path)
    assert os.path.exists(meta_path)

    with open(meta_path) as f:
        meta = json.load(f)
    
    assert "best_model_name" in meta # PERUBAHAN UTAMA DI SINI
    assert "chipset_list" in meta
    assert isinstance(meta["chipset_list"], list)
    assert "available_trained_models" in meta # Pastikan ini juga dicek
    assert "model_f1_scores" in meta # Pastikan ini juga dicek


# === Negative Test for SMOTE Exception Handling ===
@mock.patch("train_model.SMOTE")
@mock.patch("train_model.pd.read_csv")
def test_smote_failure(mock_read_csv, mock_smote, capsys): # Urutan mock harus benar
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512] * 3,
        'display_resolution': ['720p'] * 12,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 3,
        'price_range': ['low'] * 12 # Hanya satu kelas untuk memicu error SMOTE
    })
    mock_read_csv.return_value = df
    
    # Mock instance SMOTE dan metode fit_resample nya
    mock_smote_instance = mock_smote.return_value
    mock_smote_instance.fit_resample.side_effect = ValueError("SMOTE error: Not enough samples in minority class")

    train()

    captured = capsys.readouterr()
    assert "SMOTE error" in captured.out
    assert "Using original data" in captured.out