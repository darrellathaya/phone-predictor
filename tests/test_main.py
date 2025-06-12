# test_main.py
import os
import json
import joblib
import numpy as np
from fastapi.testclient import TestClient
from main import app, BASE_DIR

client = TestClient(app)

def setup_test_files():
    os.makedirs(BASE_DIR / "models", exist_ok=True)

    # Dummy model that always predicts 0
    class DummyModel:
        def predict(self, X):
            return [0] * len(X)
    joblib.dump(DummyModel(), BASE_DIR / "models" / "price_range_model.pkl")

    # Dummy accuracy
    with open(BASE_DIR / "models" / "accuracy.txt", "w") as f:
        f.write("0.95")

    # Dummy metadata
    meta = {
        "chipset_list": ["Snapdragon 8 Gen 2", "Tensor G3"],
        "resolution_list": ["720p", "1080p", "2k+"],
        "label_mapping": {"0": "Budget", "1": "Midrange", "2": "Flagship"}
    }
    with open(BASE_DIR / "models" / "meta.json", "w") as f:
        json.dump(meta, f)

def test_get_form():
    setup_test_files()
    response = client.get("/")
    assert response.status_code == 200
    assert "name=\"ram\"" in response.text

def test_post_prediction():
    setup_test_files()
    response = client.post("/", data={
        "ram": 8,
        "storage": 128,
        "display_resolution": "1080p",
        "chipset": "Snapdragon 8 Gen 2"
    })
    assert response.status_code == 200
    assert "Budget" in response.text or "prediction" in response.text
