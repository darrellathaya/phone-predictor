import pytest
import pandas as pd
from src.train_model import resolution_to_value, chipset_score, preprocess_data


def test_resolution_to_value():
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