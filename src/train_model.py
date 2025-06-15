import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import mlflow  # ✅ Make sure this is present
from mlflow.tracking import MlflowClient


# === Constants ===
DATA_PATH = os.path.join("..", "data", "raw", "train.csv")
MODEL_DIR = os.path.join("..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)


# === Helper Functions ===
def resolution_to_value(res_str: str) -> int:
    return {"720p": 720, "1080p": 1080, "2k+": 2000}.get(res_str, 720)


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


# === Main Training Logic ===
def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Encoding labels...")
    label_mapping = {label: idx for idx, label in enumerate(df['price_range'].unique())}
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    df['price_range'] = df['price_range'].map(label_mapping)

    print("Preprocessing features...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Class distribution before SMOTE:")
    print(y_train.value_counts(normalize=True))

    # Apply SMOTE if appropriate
    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:")
        print(y_train.value_counts(normalize=True))
    except ValueError as e:
        print(f"SMOTE error: {e} - Using original data.")

    models = {
        "RandomForest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42, class_weight='balanced'))
        ]),
        "SVM": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(probability=True, class_weight='balanced', random_state=42))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(eval_metric="mlogloss", random_state=42))
        ])
    }

    best_score = 0
    best_model = None
    best_name = ""

    # === ✅ MLflow logging ===
    mlflow.set_experiment("phone-price-prediction")
    with mlflow.start_run():
        print("Training models...")

        for name, model in models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")

            mlflow.log_metric(f"{name}_accuracy", acc)
            mlflow.log_metric(f"{name}_f1_score", f1)

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

        print(f"Best model: {best_name} (F1-score weighted: {best_score:.4f})")

        mlflow.log_param("best_model", best_name)
        mlflow.log_metric("best_model_f1_score", best_score)

        print("Saving model...")
        model_path = os.path.join(MODEL_DIR, "price_range_model.pkl")
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path)

        print("Saving accuracy and metadata...")
        acc_path = os.path.join(MODEL_DIR, "accuracy.txt")
        with open(acc_path, "w") as f:
            f.write(str(best_score))
        mlflow.log_artifact(acc_path)

        meta = {
            "chipset_list": sorted(df['chipset'].dropna().unique().tolist()),
            "resolution_list": ["720p", "1080p", "2k+"],
            "best_model": best_name,
            "best_model_metric_score": best_score,
            "metric_used": "f1_score_weighted",
            "label_mapping": inverse_mapping
        }
        meta_path = os.path.join(MODEL_DIR, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=4)
        mlflow.log_artifact(meta_path)

        print("Training complete!")