import os
import json
import joblib
import pandas as pd
import numpy as np

from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

import mlflow
from mlflow import MlflowClient

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Mount static file dari templates/static
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Setup templates directory
templates = Jinja2Templates(directory="templates")

# Endpoint default untuk menampilkan index.html (GET)
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    try:
        with open(os.path.join("models", "meta.json"), "r") as f:
            meta = json.load(f)
        chipset_list = meta.get("chipset_list", [])
        resolution_list = meta.get("resolution_list", [])
    except Exception:
        chipset_list = []
        resolution_list = []

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chipset_list": chipset_list,
        "resolution_list": resolution_list,
        "selected_chipset": None,
        "selected_resolution": None,
        "prediction": None,
        "accuracy": None,
        "ram": None,
        "storage": None,
        "error": None
    })

# Endpoint POST untuk prediksi
@app.post("/", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    ram: int = Form(...),
    storage: int = Form(...),
    display_resolution: str = Form(...),
    chipset: str = Form(...)
):
    try:
        model = joblib.load(os.path.join("models", "price_range_model.pkl"))
        with open(os.path.join("models", "meta.json"), "r") as f:
            meta = json.load(f)

        chipset_val = chipset_score(chipset)
        resolution_val = resolution_to_value(display_resolution)
        X_input = np.array([[ram, storage, resolution_val, chipset_val]])

        prediction_idx = model.predict(X_input)[0]
        label_mapping = meta.get("label_mapping", {})
        prediction = label_mapping.get(str(prediction_idx), "Unknown")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "chipset_list": meta.get("chipset_list", []),
            "resolution_list": meta.get("resolution_list", []),
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "prediction": prediction,
            "accuracy": round(meta.get("best_model_metric_score", 0) * 100, 2),
            "ram": ram,
            "storage": storage,
            "error": None
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "chipset_list": [],
            "resolution_list": [],
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "prediction": None,
            "accuracy": None,
            "ram": ram,
            "storage": storage,
            "error": str(e)
        })

# === Constants ===
DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_DIR = "models"
EXPERIMENT_NAME = "PhonePricePrediction"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Helpers ===
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

# === Model Factory ===
def get_models() -> Dict[str, object]:
    return {
        "RandomForest": make_pipeline(
            StandardScaler(),
            RandomForestClassifier(
                random_state=42,
                class_weight='balanced',
                min_samples_leaf=2,
                max_features='sqrt'
            )
        ),
        "SVM": make_pipeline(
            StandardScaler(),
            SVC(
                probability=True,
                class_weight='balanced',
                random_state=42
            )
        ),
        "XGBoost": make_pipeline(
            StandardScaler(),
            XGBClassifier(
                use_label_encoder=False,
                eval_metric="mlogloss",
                random_state=42
            )
        )
    }

# === MLflow Setup ===
def setup_experiment(experiment_name: str) -> str:
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        print(f"Deleting old experiment: {experiment.experiment_id}")
        client.delete_experiment(experiment.experiment_id)

    artifact_path = os.path.abspath(f"./mlruns/{experiment_name}")
    experiment_id = client.create_experiment(
        name=experiment_name,
        artifact_location=f"file://{artifact_path}"
    )
    mlflow.set_experiment(experiment_name)
    return experiment_id

# === Main Training Pipeline ===
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

    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE:")
        print(y_train.value_counts(normalize=True))
    except ValueError as e:
        print(f"SMOTE error: {e} - Using original data.")

    models = get_models()
    best_score = 0
    best_model = None
    best_name = ""

    print("Starting MLflow experiment...")
    experiment_id = setup_experiment(EXPERIMENT_NAME)

    for name, model in models.items():
        print(f"Training {name}...")
        with mlflow.start_run(experiment_id=experiment_id, run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score_weighted", f1)

            if f1 > best_score:
                best_score = f1
                best_model = model
                best_name = name

    print(f"Best model: {best_name} (F1-score weighted: {best_score:.4f})")
    joblib.dump(best_model, os.path.join(MODEL_DIR, "price_range_model.pkl"))

    with open(os.path.join(MODEL_DIR, "accuracy.txt"), "w") as f:
        f.write(str(best_score))

    meta = {
        "chipset_list": sorted(df['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"],
        "best_model": best_name,
        "best_model_metric_score": best_score,
        "metric_used": "f1_score_weighted",
        "label_mapping": inverse_mapping
    }

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print("Training complete!")