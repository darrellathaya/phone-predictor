import unittest
import pandas as pd
from main import resolution_to_value, chipset_score, preprocess_data


class TestMainFunctions(unittest.TestCase):

    def test_resolution_to_value(self):
        self.assertEqual(resolution_to_value("720p"), 720)
        self.assertEqual(resolution_to_value("1080p"), 1080)
        self.assertEqual(resolution_to_value("2k+"), 2000)
        self.assertEqual(resolution_to_value("unknown"), 720)  # default fallback

    def test_chipset_score(self):
        self.assertEqual(chipset_score("Snapdragon 8 Gen 3"), 850)
        self.assertEqual(chipset_score("Apple A16"), 830)
        self.assertEqual(chipset_score("Kirin 970"), 500)
        self.assertEqual(chipset_score("Exynos 990"), 650)
        self.assertEqual(chipset_score("Unknown Chip"), 400)  # default fallback

    def test_preprocess_data(self):
        data = {
            'ram': [4, 8],
            'storage': [64, 128],
            'display_resolution': ['720p', '1080p'],
            'chipset': ['Snapdragon 888', 'Apple A14'],
            'price_range': ['Low', 'High']
        }
        df = pd.DataFrame(data)
        X, y = preprocess_data(df)
        self.assertEqual(X.shape[1], 4)  # Should have 4 features
        self.assertEqual(len(y), 2)      # Target column length should match
        self.assertIn("chipset_score", X.columns or [])

    def test_label_mapping_consistency(self):
        data = {
            'price_range': ['Low', 'Medium', 'High']
        }
        df = pd.DataFrame(data)
        label_mapping = {label: idx for idx, label in enumerate(df['price_range'].unique())}
        inverse_mapping = {v: k for k, v in label_mapping.items()}
        self.assertEqual(inverse_mapping[label_mapping['Low']], 'Low')


if __name__ == '__main__':
    unittest.main()
