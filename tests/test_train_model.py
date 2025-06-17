import os
import json
# import joblib # Tidak digunakan secara eksplisit di semua tes
import pytest
import pandas as pd
from unittest import mock
from src.train_model import ( # Impor dari src.train_model sudah benar
    resolution_to_value,
    chipset_score,
    preprocess_data,
    train,
    MODEL_DIR # MODEL_DIR dari src.train_model
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
    assert resolution_to_value("unknown") == 720  # default fallback


def test_chipset_score_known():
    assert chipset_score("Snapdragon 8 Gen 3") == 850
    assert chipset_score("apple A14") == 770
    assert chipset_score("Kirin") == 500


def test_chipset_score_unknown():
    assert chipset_score("intel unknown") == 400


def test_preprocess_data(sample_dataframe):
    # Jika preprocess_data memodifikasi dataframe secara inplace, kirim salinannya
    df_copy = sample_dataframe.copy()
    X, y = preprocess_data(df_copy)
    assert all(col in X.columns for col in ['ram', 'storage', 'display_resolution_cat', 'chipset_score'])
    assert isinstance(y, pd.Series)


# === Integration Test for Training Pipeline ===
@mock.patch("src.train_model.pd.read_csv")
@mock.patch("src.train_model.mlflow.start_run")
@mock.patch("src.train_model.mlflow.set_experiment")
@mock.patch("src.train_model.mlflow.log_param")
@mock.patch("src.train_model.mlflow.log_metric")
def test_train_function(mock_log_metric, mock_log_param, mock_set_experiment, mock_start_run, mock_read_csv): # Urutan mock harus sesuai decorator
    # Prepare synthetic balanced dataset
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12], # Data yang cukup
        'storage': [64, 128, 256, 512] * 3,
        'display_resolution': ['720p'] * 12,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 3,
        'price_range': ['low', 'medium', 'high'] * 4 # Pastikan >1 sampel per kelas untuk SMOTE
    })
    mock_read_csv.return_value = df

    # Run training
    train() # Memanggil train dari src.train_model

    # MODEL_DIR di src.train_model.py adalah os.path.join(BASE_DIR, "..", "models")
    # BASE_DIR adalah path ke src/. Jadi, pathnya adalah src/../models -> models/
    # Ini benar jika tes dijalankan dari root proyek.
    model_path = os.path.join(MODEL_DIR, "price_range_model.pkl")
    meta_path = os.path.join(MODEL_DIR, "meta.json")
    
    # Debugging path jika perlu:
    # print(f"CWD: {os.getcwd()}")
    # print(f"MODEL_DIR (from src.train_model): {MODEL_DIR}") # Ini akan menjadi path absolut ke direktori models
    # print(f"Checking model_path: {os.path.abspath(model_path)}")
    # print(f"Checking meta_path: {os.path.abspath(meta_path)}")

    assert os.path.exists(model_path), f"File model tidak ditemukan di {os.path.abspath(model_path)}"
    assert os.path.exists(meta_path), f"File meta tidak ditemukan di {os.path.abspath(meta_path)}"

    # Check metadata content
    with open(meta_path) as f:
        meta = json.load(f)
    
    # --- PERUBAHAN UTAMA DI SINI UNTUK MEMPERBAIKI ASSERTIONERROR ---
    assert "best_model_name" in meta  # Key yang benar adalah 'best_model_name'
    # -----------------------------------------------------------------
    assert "chipset_list" in meta
    assert isinstance(meta["chipset_list"], list)
    # Tambahkan pengecekan untuk key baru lainnya yang ada di meta.json Anda
    assert "available_trained_models" in meta
    assert "model_f1_scores" in meta


# === Negative Test for SMOTE Exception Handling ===
@mock.patch("src.train_model.SMOTE")
@mock.patch("src.train_model.pd.read_csv")
def test_smote_failure(mock_read_csv, mock_smote, capsys): # Urutan mock harus benar
    df = pd.DataFrame({
        'ram': [4, 6, 8, 12, 4, 6, 8, 12, 4, 6, 8, 12],
        'storage': [64, 128, 256, 512] * 3,
        'display_resolution': ['720p'] * 12,
        'chipset': ['snapdragon 888', 'apple a16', 'unknown', 'kirin'] * 3,
        'price_range': ['low'] * 12 # Hanya satu kelas agar SMOTE error jika tidak di-mock dengan benar
                                    # atau jika data sangat sedikit dan SMOTE tidak bisa jalan
    })
    mock_read_csv.return_value = df
    
    # Mock instance SMOTE dan metode fit_resample nya
    # Jika SMOTE() dipanggil di train_model, maka mock_smote.return_value adalah instance
    mock_smote_instance = mock_smote.return_value
    mock_smote_instance.fit_resample.side_effect = ValueError("SMOTE error: Test-induced error") # Pesan error bisa apa saja

    train() # Memanggil train dari src.train_model

    captured = capsys.readouterr()
    assert "SMOTE error" in captured.out
    assert "Using original data" in captured.out # Pastikan pesan fallback ini ada di output