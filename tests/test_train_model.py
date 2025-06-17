import os
import json
import pytest
import pandas as pd
from unittest import mock
from src.train_model import (
    resolution_to_value,
    chipset_score,
    preprocess_data,
    train,
    MODEL_DIR
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512] * 3,
        'display_resolution': ['720p'] * 12,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 3,
        'price_range': ['low', 'medium', 'high'] * 4
    })


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
    df_copy = sample_dataframe.copy()
    X, y = preprocess_data(df_copy)
    assert all(col in X.columns for col in ['ram', 'storage', 'display_resolution_cat', 'chipset_score'])
    assert isinstance(y, pd.Series)



@mock.patch("src.train_model.pd.read_csv")
@mock.patch("src.train_model.mlflow.start_run")
@mock.patch("src.train_model.mlflow.set_experiment")
@mock.patch("src.train_model.mlflow.log_param")
@mock.patch("src.train_model.mlflow.log_metric")

@mock.patch("src.train_model.mlflow.sklearn.log_model")
def test_train_function(mock_log_model, mock_log_metric, mock_log_param, mock_set_experiment, mock_start_run, mock_read_csv):
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512] * 3,
        'display_resolution': ['720p'] * 12,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 3,
        'price_range': ['low', 'medium', 'high'] * 4
    })
    mock_read_csv.return_value = df

    train()

    model_path = os.path.join(MODEL_DIR, "price_range_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    
    assert os.path.exists(model_path)
    assert os.path.exists(meta_path)

    with open(meta_path) as f:
        meta = json.load(f)
    
    assert "best_model_name" in meta
    assert "chipset_list" in meta
    assert isinstance(meta["chipset_list"], list)
    assert "available_trained_models" in meta
    assert "model_f1_scores" in meta


@mock.patch("src.train_model.SMOTE")
@mock.patch("src.train_model.pd.read_csv")
def test_smote_failure(mock_read_csv, mock_smote, capsys):
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512] * 4,
        'display_resolution': ['720p'] * 16,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 4,
        'price_range': (['low'] * 8) + (['medium'] * 8)
    })
    # -------------------------------------------------------------
    mock_read_csv.return_value = df
    
    mock_smote_instance = mock_smote.return_value
    mock_smote_instance.fit_resample.side_effect = ValueError("SMOTE failed due to a test-induced generic error.")


    train()

    captured = capsys.readouterr()
    assert "SMOTE error" in captured.out
    assert "Using original data" in captured.out