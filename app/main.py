import os
import json
import joblib
import pandas as pd
import numpy as np
import shutil # Import shutil

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
templates = Jinja2Templates(directory="templates")

DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_DIR = "models"
MLRUNS_DIR_MAIN = "mlruns" # Path ke direktori mlruns dari root
EXPERIMENT_NAME = "PhonePricePrediction"
os.makedirs(MODEL_DIR, exist_ok=True)

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

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    chipset_list_val = []
    resolution_list_val = []
    available_models_val = ["RandomForest", "SVM", "XGBoost"] 
    selected_model_default = available_models_val[0] if available_models_val else None
    best_model_overall_name_val = None

    try:
        with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
            meta = json.load(f)
        chipset_list_val = meta.get("chipset_list", [])
        resolution_list_val = meta.get("resolution_list", [])
        available_models_val = meta.get("available_trained_models", available_models_val)
        best_model_overall_name_val = meta.get("best_model_name")
        
        default_selection = meta.get("best_model_name")
        if default_selection and default_selection in available_models_val:
            selected_model_default = default_selection
        elif available_models_val:
            selected_model_default = available_models_val[0]

    except Exception:
        pass

    return templates.TemplateResponse("index.html", {
        "request": request,
        "chipset_list": chipset_list_val,
        "resolution_list": resolution_list_val,
        "available_models": available_models_val,
        "selected_model_name": selected_model_default,
        "best_model_overall_name": best_model_overall_name_val,
        "selected_chipset": None,
        "selected_resolution": None,
        "prediction": None,
        "accuracy_overall_best_model": None,
        "accuracy_selected_model": None,
        "ram": None,
        "storage": None,
        "error": None
    })

@app.post("/", response_class=HTMLResponse)
async def predict_price(
    request: Request,
    ram: int = Form(...),
    storage: int = Form(...),
    display_resolution: str = Form(...),
    chipset: str = Form(...),
    selected_model_name: str = Form(...)
):
    chipset_list_val = []
    resolution_list_val = []
    available_models_val = ["RandomForest", "SVM", "XGBoost"]
    meta = {}
    accuracy_overall_best_model_val = None
    accuracy_selected_model_val = None
    best_model_overall_name_val = None

    try:
        with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f:
            meta = json.load(f)
        chipset_list_val = meta.get("chipset_list", [])
        resolution_list_val = meta.get("resolution_list", [])
        available_models_val = meta.get("available_trained_models", available_models_val)
        best_model_overall_name_val = meta.get("best_model_name")

        model_filename = f"{selected_model_name}.pkl"
        model_path = os.path.join(MODEL_DIR, model_filename)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"File model '{model_filename}' tidak ditemukan.")
        
        model = joblib.load(model_path)

        chipset_val = chipset_score(chipset)
        resolution_val = resolution_to_value(display_resolution)
        X_input = np.array([[ram, storage, resolution_val, chipset_val]])

        prediction_idx = model.predict(X_input)[0]
        label_mapping = meta.get("label_mapping", {})
        prediction_result = label_mapping.get(str(prediction_idx), "Unknown")

        overall_best_f1_score = meta.get("best_model_overall_f1_score", 0)
        accuracy_overall_best_model_val = round(overall_best_f1_score * 100, 2)
        
        model_f1_scores_dict = meta.get("model_f1_scores", {})
        selected_model_f1 = model_f1_scores_dict.get(selected_model_name, 0)
        accuracy_selected_model_val = round(selected_model_f1 * 100, 2)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "chipset_list": chipset_list_val,
            "resolution_list": resolution_list_val,
            "available_models": available_models_val,
            "selected_model_name": selected_model_name,
            "best_model_overall_name": best_model_overall_name_val,
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "prediction": prediction_result,
            "accuracy_overall_best_model": accuracy_overall_best_model_val,
            "accuracy_selected_model": accuracy_selected_model_val,
            "ram": ram,
            "storage": storage,
            "error": None
        })

    except Exception as e:
        current_selected_model = selected_model_name if 'selected_model_name' in locals() and selected_model_name else (available_models_val[0] if available_models_val else None)
        if not meta:
            try:
                 with open(os.path.join(MODEL_DIR, "meta.json"), "r") as f_err:
                    meta_err = json.load(f_err)
                 chipset_list_val = meta_err.get("chipset_list", chipset_list_val)
                 resolution_list_val = meta_err.get("resolution_list", resolution_list_val)
                 available_models_val = meta_err.get("available_trained_models", available_models_val)
                 best_model_overall_name_val = meta_err.get("best_model_name")
            except Exception:
                pass

        return templates.TemplateResponse("index.html", {
            "request": request,
            "chipset_list": chipset_list_val,
            "resolution_list": resolution_list_val,
            "available_models": available_models_val,
            "selected_model_name": current_selected_model,
            "best_model_overall_name": best_model_overall_name_val,
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "prediction": None,
            "accuracy_overall_best_model": None,
            "accuracy_selected_model": None,
            "ram": ram,
            "storage": storage,
            "error": str(e)
        })

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

def setup_experiment_main(experiment_name: str, mlruns_path: str): # Diubah agar mirip dengan src/train_model.py
    if os.path.exists(mlruns_path):
        print(f"Deleting existing MLruns directory: {mlruns_path} (from main.py train)")
        try:
            shutil.rmtree(mlruns_path)
        except OSError as e:
            print(f"Error deleting MLruns directory {mlruns_path} in main.py: {e}")

    mlflow.set_tracking_uri(f"file:{mlruns_path}")
    client = MlflowClient()
    try:
        experiment_id = client.create_experiment(name=experiment_name)
        mlflow.set_experiment(experiment_name)
        print(f"Created and set new experiment '{experiment_name}' with ID: {experiment_id} (from main.py train)")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e).lower():
            print(f"Experiment '{experiment_name}' already exists (from main.py train). Setting it as active.")
            mlflow.set_experiment(experiment_name)
        else:
            print(f"Critical error creating/setting experiment '{experiment_name}' in main.py: {e}")
            raise

def train(): # Fungsi train di main.py
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
    best_model_obj = None
    best_name = ""
    model_f1_scores_for_meta = {}

    print("Starting MLflow experiment (from main.py)...")
    setup_experiment_main(EXPERIMENT_NAME, MLRUNS_DIR_MAIN)

    for name, model_instance in models.items():
        print(f"Training {name} (from main.py)...")
        with mlflow.start_run(run_name=name):
            model_instance.fit(X_train, y_train)
            preds = model_instance.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')

            model_f1_scores_for_meta[name] = f1

            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score_weighted", f1)
            
            joblib.dump(model_instance, os.path.join(MODEL_DIR, f"{name}.pkl"))
            mlflow.sklearn.log_model(model_instance, artifact_path=name)

            if f1 > best_score:
                best_score = f1
                best_model_obj = model_instance
                best_name = name

    print(f"Best model: {best_name} (F1-score weighted: {best_score:.4f}) (from main.py)")
    joblib.dump(best_model_obj, os.path.join(MODEL_DIR, "price_range_model.pkl"))
    
    accuracy_txt_path = os.path.join(MODEL_DIR, "accuracy.txt")
    if os.path.exists(accuracy_txt_path):
        os.remove(accuracy_txt_path)

    meta_content = {
        "chipset_list": sorted(df['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"],
        "best_model_name": best_name,
        "best_model_overall_f1_score": best_score,
        "model_f1_scores": model_f1_scores_for_meta,
        "metric_used": "f1_score_weighted",
        "label_mapping": inverse_mapping,
        "available_trained_models": list(models.keys())
    }

    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta_content, f, indent=4)

    print("Training complete (from main.py)!")

# if __name__ == "__main__":
# train()