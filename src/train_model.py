import os
import json
import joblib
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import mlflow

# === Constants ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
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

def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[int, str]]:
    label_mapping = {label: idx for idx, label in enumerate(df['price_range'].unique())}
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    df['price_range'] = df['price_range'].map(label_mapping)
    return df, inverse_mapping

def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df['display_resolution_cat'] = df['display_resolution'].map(resolution_to_value)
    df['chipset_score'] = df['chipset'].map(chipset_score)
    return df[['ram', 'storage', 'display_resolution_cat', 'chipset_score']], df['price_range']

def apply_smote(X, y):
    print("Class distribution before SMOTE:")
    print(y.value_counts(normalize=True))
    try:
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)
        print("Class distribution after SMOTE:")
        print(y_res.value_counts(normalize=True))
        return X_res, y_res
    except ValueError as e:
        print(f"SMOTE error: {e} - Using original data.")
        return X, y

def make_pipeline(classifier):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier)
    ])

def get_models():
    return {
        "RandomForest": make_pipeline(RandomForestClassifier(random_state=42, class_weight='balanced')),
        "SVM": make_pipeline(SVC(probability=True, class_weight='balanced', random_state=42)),
        "XGBoost": make_pipeline(XGBClassifier(eval_metric="mlogloss", random_state=42))
    }

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    mlflow.set_experiment("PhonePricePrediction")
    best_score = 0
    best_model = None
    best_name = ""

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
            print("Sample predictions:", preds[:10])
            print("True labels:", y_test[:10].values)

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

    return best_model, best_name, best_score

def save_artifacts(model, score, best_name, inverse_mapping, df):
    joblib.dump(model, os.path.join(MODEL_DIR, "price_range_model.pkl"))
    with open(os.path.join(MODEL_DIR, "accuracy.txt"), "w") as f:
        f.write(str(score))

    meta = {
        "chipset_list": sorted(df['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"],
        "best_model": best_name,
        "best_model_metric_score": score,
        "metric_used": "f1_score_weighted",
        "label_mapping": inverse_mapping
    }

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)

    print("Encoding labels...")
    df_encoded, inverse_mapping = encode_labels(df)

    print("Preprocessing features...")
    X, y = preprocess_data(df_encoded)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Applying SMOTE...")
    X_train, y_train = apply_smote(X_train, y_train)

    print("Training models...")
    models = get_models()
    best_model, best_name, best_score = train_and_evaluate(models, X_train, X_test, y_train, y_test)

    print(f"Best model: {best_name} (F1-score weighted: {best_score:.4f})")
    print("Saving model and metadata...")
    save_artifacts(best_model, best_score, best_name, inverse_mapping, df)

    print("Training complete!")

# Entry point
if __name__ == "__main__":
    train()
