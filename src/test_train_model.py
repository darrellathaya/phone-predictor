
import os
import pandas as pd
from train import train

def test_train_runs(tmp_path, monkeypatch):
    # Setup fake data directory
    data_dir = tmp_path / "data" / "raw"
    os.makedirs(data_dir)
    df = pd.DataFrame({
        "ram": [4, 6, 8, 4],
        "storage": [64, 128, 256, 64],
        "display_resolution": ["720p", "1080p", "2k+", "720p"],
        "chipset": ["snapdragon 765", "snapdragon 888", "apple a16", "kirin"],
        "price_range": ["budget", "mid", "flagship", "budget"]
    })
    df.to_csv(data_dir / "train.csv", index=False)

    # Monkeypatch constants
    monkeypatch.setattr("train.DATA_PATH", str(data_dir / "train.csv"))
    monkeypatch.setattr("train.MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setattr("train.EXPERIMENT_NAME", "TestExp")

    # Run training
    train()
