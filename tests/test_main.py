import unittest
import os
import json
import shutil
import tempfile
from unittest.mock import patch, MagicMock

# Import from correct path
import pandas as pd
from app import main  # <-- corrected here


class TestMain(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.data_path = os.path.join(self.test_dir, "data", "raw", "train.csv")
        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)

        self.sample_df = pd.DataFrame({
            'ram': [4, 6, 8],
            'storage': [64, 128, 256],
            'display_resolution': ['720p', '1080p', '2k+'],
            'chipset': ['Snapdragon 8 Gen 3', 'Apple A15', 'Kirin'],
            'price_range': ['Low', 'Medium', 'High']
        })

        self.sample_df.to_csv(self.data_path, index=False)

        self.patcher_data_path = patch('app.main.DATA_PATH', self.data_path)
        self.patcher_data_path.start()

        self.patcher_makedirs = patch('os.makedirs')
        self.patcher_makedirs.start()

    def tearDown(self):
        self.patcher_data_path.stop()
        self.patcher_makedirs.stop()
        shutil.rmtree(self.test_dir)

    def test_resolution_to_value(self):
        self.assertEqual(main.resolution_to_value("720p"), 720)
        self.assertEqual(main.resolution_to_value("1080p"), 1080)
        self.assertEqual(main.resolution_to_value("2k+"), 2000)
        self.assertEqual(main.resolution_to_value("unknown"), 720)

    def test_chipset_score(self):
        self.assertEqual(main.chipset_score("snapdragon 8 gen 3"), 850)
        self.assertEqual(main.chipset_score("apple a18"), 870)
        self.assertEqual(main.chipset_score("exynos"), 650)
        self.assertEqual(main.chipset_score("unknown chipset"), 400)

    def test_preprocess_data(self):
        df = self.sample_df.copy()
        X, y = main.preprocess_data(df)
        self.assertIn('display_resolution_cat', X.columns)
        self.assertIn('chipset_score', X.columns)
        self.assertTrue((X['display_resolution_cat'] == [720, 1080, 2000]).all())
        self.assertTrue((X['chipset_score'] == [850, 800, 500]).all())

    @patch('mlflow.MlflowClient')
    @patch('mlflow.set_tracking_uri')
    @patch('mlflow.set_experiment')
    def test_setup_experiment(self, mock_set_exp, mock_set_uri, mock_client):
        client_instance = mock_client.return_value
        client_instance.get_experiment_by_name.return_value = None
        client_instance.create_experiment.return_value = "test_exp_id"

        exp_id = main.setup_experiment("TestExperiment")
        self.assertEqual(exp_id, "test_exp_id")
        mock_set_exp.assert_called_once_with("TestExperiment")

    @patch('mlflow.start_run')
    @patch('mlflow.log_metric')
    @patch('mlflow.log_param')
    @patch('joblib.dump')
    @patch('app.main.setup_experiment')
    def test_train(self, mock_setup_exp, mock_joblib_dump, mock_log_param, mock_log_metric, mock_start_run):
        mock_setup_exp.return_value = "test_exp_123"

        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = self.sample_df.copy()

            with patch.object(main, 'preprocess_data', wraps=main.preprocess_data) as wrapped_preprocess:
                with patch('sklearn.model_selection.train_test_split', return_value=(
                    pd.DataFrame([[4, 64, 720, 850], [6, 128, 1080, 800]]),
                    pd.DataFrame([[8]][[256]][[2000]][[500]]),  # <-- fixed syntax
                    pd.Series([0, 1]),
                    pd.Series([2])
                )):
                    with patch('imblearn.over_sampling.SMOTE.fit_resample', return_value=(
                        pd.DataFrame([[4, 64, 720, 850], [6, 128, 1080, 800], [4, 64, 720, 850]]),
                        pd.Series([0, 1, 0])
                    )):
                        with patch('sklearn.ensemble.RandomForestClassifier.fit') as mock_rf_fit:
                            with patch('sklearn.svm.SVC.fit'):
                                with patch('xgboost.XGBClassifier.fit'):
                                    main.train()
                                    mock_joblib_dump.assert_called_once()
                                    model_dir = "models"
                                    self.assertTrue(os.path.exists(os.path.join(model_dir, "accuracy.txt")))
                                    self.assertTrue(os.path.exists(os.path.join(model_dir, "meta.json")))

                                    with open(os.path.join(model_dir, "meta.json")) as f:
                                        meta = json.load(f)
                                        self.assertIn("chipset_list", meta)
                                        self.assertIn("resolution_list", meta)
                                        self.assertIn("label_mapping", meta)


if __name__ == "__main__":
    unittest.main()