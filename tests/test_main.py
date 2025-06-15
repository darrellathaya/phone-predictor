import pytest
import os
import pandas as pd
import joblib
import json
from unittest.mock import patch, mock_open, MagicMock

# Adjust import based on where your train function lives
from src.train_model import train


@patch("src.train_model.pd.read_csv")
@patch("src.train_model.joblib.dump")
@patch("os.makedirs")
def test_train_function_runs(mock_makedirs, mock_joblib_dump, mock_read_csv, tmp_path):
    # --- Arrange ---
    # Mock data path to use temp directory
    mock_data_path = tmp_path / "data" / "raw"
    mock_data_path.mkdir(parents=True)
    mock_data_file = mock_data_path / "train.csv"
    mock_data_file.write_text("dummy")  # Create dummy file

    with patch("src.train_model.DATA_PATH", str(mock_data_file)):

        # Simulated dataset
        mock_df = pd.DataFrame({
            "ram": [4, 6, 8, 4, 12, 8],
            "storage": [64, 128, 256, 64, 512, 256],
            "display_resolution": ["720p", "1080p", "2k+", "720p", "1080p", "2k+"],
            "chipset": [
                "Snapdragon 8 Gen 3",
                "Apple A14",
                "Helio G99",
                "Tensor G3",
                "Snapdragon 8 Gen 2",
                "Dimensity 9200"
            ],
            "price_range": ["Low", "Medium", "High", "Low", "Medium", "High"]
        })
        mock_read_csv.return_value = mock_df

        # Mock train_test_split output
        X_train = pd.DataFrame([[4, 64, 720, 850], [6, 128, 1080, 800]])
        y_train = pd.Series([0, 1])
        X_test = pd.DataFrame([[8], [256], [2000], [500]])
        y_test = pd.Series([2])

        with patch("src.train_model.train_test_split", return_value=(X_train, X_test, y_train, y_test)):

            # Mock SMOTE fit_resample
            with patch("src.train_model.SMOTE.fit_resample", return_value=(X_train, y_train)):

                # Mock classifiers
                mock_clf = MagicMock()
                mock_clf.fit.return_value = None
                mock_clf.predict.return_value = pd.Series([2])

                with patch("src.train_model.RandomForestClassifier", return_value=mock_clf):
                    with patch("src.train_model.SVC", return_value=mock_clf):
                        with patch("src.train_model.XGBClassifier", return_value=mock_clf):

                            # Run the train function
                            train()

                            # --- Asserts ---

                            # Check data was read
                            mock_read_csv.assert_called_once()

                            # Check models trained
                            assert mock_clf.fit.call_count == 3  # All 3 models called .fit()

                            # Check predictions made
                            assert mock_clf.predict.call_count == 3  # All models predict on test set

                            # Check best model saved
                            mock_joblib_dump.assert_called_once()

                            # Check model dir created
                            mock_makedirs.assert_called_with("models", exist_ok=True)

                            # Check accuracy.txt and meta.json were written
                            assert os.path.exists(os.path.join("models", "accuracy.txt"))
                            assert os.path.exists(os.path.join("models", "meta.json"))

                            # Validate meta.json content
                            with open(os.path.join("models", "meta.json")) as f:
                                meta = json.load(f)
                                assert "chipset_list" in meta
                                assert "resolution_list" in meta
                                assert "label_mapping" in meta