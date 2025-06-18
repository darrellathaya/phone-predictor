import os
import json
import joblib
import jinja2
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

app.mount("/static", StaticFiles(directory="templates/static"), name="static")
templates = Jinja2Templates(
    env=jinja2.Environment(
        loader=jinja2.FileSystemLoader("templates"),
        auto_reload=True, 
        autoescape=jinja2.select_autoescape(["html", "xml"])  
    )
)
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)
DATA_PATH = os.path.join("data", "raw", "train.csv")
EXPERIMENT_NAME = "PhonePricePrediction"

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
    for key_pattern, score_val in scores.items():
        if key_pattern in chipset:
            return score_val
    return 400

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    context = {
        "request": request,
        "chipset_list": [],
        "resolution_list": [],
        "available_models": [],
        "selected_model_name": None,
        "best_model_overall_name": None,
        "selected_chipset": None,
        "selected_resolution": None,
        "prediction": None,
        "accuracy_overall_best_model": None,
        "accuracy_selected_model": None,
        "ram": 2048,
        "storage": 128,
        "error": None
    }
    try:
        with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
            meta = json.load(f)
        context["chipset_list"] = meta.get("chipset_list", [])
        context["resolution_list"] = meta.get("resolution_list", [])
        context["available_models"] = meta.get("available_trained_models", [])
        context["best_model_overall_name"] = meta.get("best_model_name")
        default_selection = meta.get("best_model_name")
        if default_selection and default_selection in context["available_models"]:
            context["selected_model_name"] = default_selection
        elif context["available_models"]:
            context["selected_model_name"] = context["available_models"][0]
    except Exception as e:
        context["error"] = "Gagal memuat metadata awal"
        context["available_models"] = ["RandomForest", "SVM", "XGBoost"]
        if context["available_models"]:
            context["selected_model_name"] = context["available_models"][0]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context
    )

@app.post("/", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    ram: int = Form(...),
    storage: int = Form(...),
    display_resolution: str = Form(...),
    chipset: str = Form(...),
    selected_model_name: str = Form(...)
):
    context = {
        "request": request,
        "ram": ram,
        "storage": storage,
        "selected_resolution": display_resolution,
        "selected_chipset": chipset,
        "selected_model_name": selected_model_name,
        "prediction": None,
        "accuracy_overall_best_model": None,
        "accuracy_selected_model": None,
        "best_model_overall_name": None,
        "chipset_list": [],
        "resolution_list": [],
        "available_models": [],
        "error": None
    }

    try:
        with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
            meta = json.load(f)
        context["chipset_list"] = meta.get("chipset_list", [])
        context["resolution_list"] = meta.get("resolution_list", [])
        context["available_models"] = meta.get("available_trained_models", ["RandomForest", "SVM", "XGBoost"])
        context["best_model_overall_name"] = meta.get("best_model_name", "N/A")

        model_filename = f"{selected_model_name}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model '{model_filename}' tidak ditemukan di '{MODEL_DIR}'.")

        loaded_model = joblib.load(model_path)
        chipset_val = chipset_score(chipset)
        resolution_val = resolution_to_value(display_resolution)
        X_input = np.array([[ram, storage, resolution_val, chipset_val]])
        prediction_idx = loaded_model.predict(X_input)[0]
        label_mapping = meta.get("label_mapping", {})
        context["prediction"] = label_mapping.get(str(prediction_idx), "Unknown")

        overall_best_f1_score = meta.get("best_model_overall_f1_score", 0)
        context["accuracy_overall_best_model"] = round(overall_best_f1_score * 100, 2)
        model_f1_scores_dict = meta.get("model_f1_scores", {})
        selected_model_f1 = model_f1_scores_dict.get(selected_model_name, 0)
        context["accuracy_selected_model"] = round(selected_model_f1 * 100, 2)

    except Exception as e:
        context["error"] = str(e)
        try:
            with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f_err:
                meta_err = json.load(f_err)
            context["chipset_list"] = meta_err.get("chipset_list", [])
            context["resolution_list"] = meta_err.get("resolution_list", [])
            context["available_models"] = meta_err.get("available_trained_models", ["RandomForest", "SVM", "XGBoost"])
            context["best_model_overall_name"] = meta_err.get("best_model_name")
        except Exception:
            context["available_models"] = ["RandomForest", "SVM", "XGBoost"]

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context=context
    )


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    df['display_resolution_cat'] = df['display_resolution'].map(resolution_to_value)
    df['chipset_score'] = df['chipset'].map(chipset_score)
    return df[['ram', 'storage', 'display_resolution_cat', 'chipset_score']], df['price_range']

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

def setup_experiment(experiment_name: str) -> str:
    mlflow.set_tracking_uri("file:./mlruns")
    client = MlflowClient()
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment:
        print(f"Deleting old experiment: {experiment.experiment_id} (from main.py)")
        client.delete_experiment(experiment.experiment_id)

    
    experiment_id_str = client.create_experiment(
        name=experiment_name
    )
    mlflow.set_experiment(experiment_name)
    return experiment_id_str

def train():
    print("Loading data (from main.py)...")
    df = pd.read_csv(DATA_PATH)
    df_for_meta = df.copy()

    print("Encoding labels (from main.py)...")
    label_mapping = {label: idx for idx, label in enumerate(df['price_range'].unique())}
    inverse_mapping = {v: k for k, v in label_mapping.items()}
    df['price_range'] = df['price_range'].map(label_mapping)

    print("Preprocessing features (from main.py)...")
    X, y = preprocess_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("Class distribution before SMOTE (from main.py):")
    print(y_train.value_counts(normalize=True))

    try:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print("Class distribution after SMOTE (from main.py):")
        print(y_train.value_counts(normalize=True))
    except ValueError as e:
        print(f"SMOTE error: {e} - Using original data (from main.py).")

    models_dict = get_models()
    best_overall_f1 = 0
    best_model_obj = None
    best_model_name_str = ""
    all_model_f1_scores = {}

    print("Starting MLflow experiment (from main.py)...")
    
    exp_id = setup_experiment(EXPERIMENT_NAME)

    for name, model_pipeline in models_dict.items():
        print(f"Training {name} (from main.py)...")
        with mlflow.start_run(run_name=name):
            model_pipeline.fit(X_train, y_train)
            
            joblib.dump(model_pipeline, os.path.join(MODEL_DIR, f"{name}.pkl"))
            
            preds = model_pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            all_model_f1_scores[name] = f1

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score_weighted", f1)
            

            if f1 > best_overall_f1:
                best_overall_f1 = f1
                best_model_obj = model_pipeline
                best_model_name_str = name

    print(f"Best model: {best_model_name_str} (F1-score weighted: {best_overall_f1:.4f}) (from main.py)")
    joblib.dump(best_model_obj, os.path.join(MODEL_DIR, "price_range_model.pkl"))

    
    accuracy_txt_file = os.path.join(MODEL_DIR, "accuracy.txt")
    if os.path.exists(accuracy_txt_file):
        os.remove(accuracy_txt_file)

    meta_content = {
        "chipset_list": sorted(df_for_meta['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"],
        "best_model_name": best_model_name_str,
        "best_model_overall_f1_score": best_overall_f1,
        "model_f1_scores": all_model_f1_scores,
        "metric_used": "f1_score_weighted",
        "label_mapping": inverse_mapping,
        "available_trained_models": list(models_dict.keys())
    }

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta_content, f, indent=4)

    print("Training complete (from main.py)!")