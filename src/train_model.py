import os
import json
import joblib
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import mlflow
from mlflow import MlflowClient

# === Constants ===
DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_DIR = "models"
EXPERIMENT_NAME = "PhonePricePrediction"
os.makedirs(MODEL_DIR, exist_ok=True)


# === Helper functions ===
def resolution_to_value(res_str: str) -> int:
    mapping = {"720p": 720, "1080p": 1080, "2k+": 2000}
    return mapping.get(res_str, 720)

def chipset_score(chipset: str) -> int:
    chipset = chipset.lower()
    scores = {
        'snapdragon 8 gen 3': 850, 'snapdragon 8 gen 2': 820, 'snapdragon 888': 800,
        'snapdragon 855': 730, 'snapdragon 778': 720, 'snapdragon 765': 690,
        'helio g99': 650, 'tensor g4': 830, 'tensor g3': 800, 'tensor g2': 780,
        'tensor': 750, 'apple a18': 870, 'apple a17': 850, 'apple a16': 830,
        'apple a15': 800, 'apple a14': 770, 'apple a13': 740, 'apple a12': 720,
        'apple a11': 690, 'kirin': 500, 'exynos': 650
    }
    for key in scores:
        if key in chipset:
            return scores[key]
    return 400


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df['display_resolution_cat'] = df['display_resolution'].map(resolution_to_value)
    df['chipset_score'] = df['chipset'].map(chipset_score)
    return df[['ram', 'storage', 'display_resolution_cat', 'chipset_score']], df['price_range']


def setup_experiment(experiment_name: str) -> str:
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment:
        print(f"Deleting old experiment: {experiment.experiment_id}")
        client.delete_experiment(experiment.experiment_id)

    artifact_path = os.path.abspath("./mlruns/" + experiment_name)
    experiment_id = client.create_experiment(
        name=experiment_name,
        artifact_location=f"file://{artifact_path}"
    )
    mlflow.set_experiment(experiment_name)
    return experiment_id


def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Converting labels to integers...")
    label_mapping = {label: idx for idx, label in enumerate(df['price_range'].unique())}
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    df['price_range'] = df['price_range'].map(label_mapping)

    print("Preprocessing features...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42))
        ])
    }

    best_score = 0
    best_model = None
    best_name = ""

    print("Setting up MLflow experiment...")
    experiment_id = setup_experiment(EXPERIMENT_NAME)

    for name, model in models.items():
        print(f"Training: {name}")
        with mlflow.start_run(experiment_id=experiment_id, run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)

            if acc > best_score:
                best_score = acc
                best_model = model
                best_name = name

    print(f"Best model: {best_name} (accuracy: {best_score:.4f})")

    print("Saving model and artifacts...")
    joblib.dump(best_model, os.path.join(MODEL_DIR, "price_range_model.pkl"))

    with open(os.path.join(MODEL_DIR, "accuracy.txt"), "w") as f:
        f.write(str(best_score))

    meta = {
        "chipset_list": sorted(df['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"],
        "best_model": best_name,
        "label_mapping": inverse_mapping
    }

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("Training complete and files saved.")


if __name__ == "__main__":
    train()
