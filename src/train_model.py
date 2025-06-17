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
from mlflow import MlflowClient
import shutil # Import shutil untuk menghapus direktori

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
MLRUNS_DIR = os.path.join(BASE_DIR, "..", "mlruns")
os.makedirs(MODEL_DIR, exist_ok=True)
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

def make_pipeline_func(classifier):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier)
    ])

def get_models():
    return {
        "RandomForest": make_pipeline_func(RandomForestClassifier(random_state=42, class_weight='balanced')),
        "SVM": make_pipeline_func(SVC(probability=True, class_weight='balanced', random_state=42)),
        "XGBoost": make_pipeline_func(XGBClassifier(eval_metric="mlogloss", random_state=42, use_label_encoder=False))
    }

def clean_and_setup_mlflow_experiment(experiment_name: str, mlruns_path: str):
    # Hapus direktori mlruns jika ada untuk memastikan state bersih
    if os.path.exists(mlruns_path):
        print(f"Deleting existing MLruns directory: {mlruns_path}")
        try:
            shutil.rmtree(mlruns_path)
        except OSError as e:
            print(f"Error deleting MLruns directory {mlruns_path}: {e}")
            # Jika penghapusan gagal, ini bisa menjadi masalah. Untuk CI, kita bisa mencoba melanjutkan.
    
    # Buat ulang direktori mlruns (opsional, mlflow akan membuatnya jika tracking URI menunjuk ke sana)
    os.makedirs(mlruns_path, exist_ok=True) # Baris ini tidak esensial jika tracking URI diset dengan benar

    mlflow.set_tracking_uri(f"file:{mlruns_path}") # Set tracking URI ke path absolut atau relatif yang benar
    
    client = MlflowClient()
    # Karena mlruns sudah bersih, kita bisa langsung membuat eksperimen baru.
    # Tidak perlu memeriksa atau menghapus eksperimen dengan nama yang sama lagi.
    try:
        experiment_id = client.create_experiment(name=experiment_name)
        mlflow.set_experiment(experiment_name)
        print(f"Created and set new experiment '{experiment_name}' with ID: {experiment_id}")
    except mlflow.exceptions.MlflowException as e:
        if "already exists" in str(e).lower(): # Menangani kasus jika create_experiment gagal karena sudah ada (seharusnya tidak terjadi setelah rmtree)
            print(f"Experiment '{experiment_name}' already exists. Setting it as active.")
            mlflow.set_experiment(experiment_name)
        else:
            print(f"Critical error creating/setting experiment '{experiment_name}': {e}")
            raise

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    # MLRUNS_DIR didefinisikan di atas, relatif terhadap root proyek jika train_model.py di src/
    # Jika train_model.py di root, MLRUNS_DIR akan menjadi "./mlruns"
    # Untuk CI, pastikan path ini benar berdasarkan CWD.
    # Jika CWD adalah root proyek, maka `os.path.join(BASE_DIR, "..", "mlruns")` menjadi `src/../mlruns` -> `mlruns`
    clean_and_setup_mlflow_experiment(EXPERIMENT_NAME, MLRUNS_DIR)

    best_overall_score = 0
    best_model_obj = None
    best_model_overall_name = ""
    model_f1_scores = {}

    for name, model_pipeline in models.items():
        with mlflow.start_run(run_name=name): # Seharusnya berjalan di bawah eksperimen yang baru dibuat
            print(f"Training {name}...")
            model_pipeline.fit(X_train, y_train)
            
            model_path = os.path.join(MODEL_DIR, f"{name}.pkl")
            joblib.dump(model_pipeline, model_path)
            print(f"Saved {name} model to {model_path}")

            preds = model_pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            
            model_f1_scores[name] = f1

            print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            mlflow.sklearn.log_model(model_pipeline, artifact_path=name) # Gunakan artifact_path, bukan name untuk MLflow >= 2.0

            if f1 > best_overall_score:
                best_overall_score = f1
                best_model_obj = model_pipeline
                best_model_overall_name = name
    
    return best_model_obj, best_model_overall_name, best_overall_score, model_f1_scores

def save_artifacts(best_model_object, best_model_name_val, best_overall_f1_score, all_model_f1_scores, inverse_mapping_dict, df_original):
    joblib.dump(best_model_object, os.path.join(MODEL_DIR, "price_range_model.pkl"))

    meta = {
    "chipset_list": sorted(df_original['chipset'].dropna().unique().tolist()),
    "resolution_list": ["720p", "1080p", "2k+"],
    "best_model": best_model_object.__class__.__name__,  # ✅ Add this line
    "best_model_name": best_model_name_val,
    "best_model_overall_f1_score": best_overall_f1_score,
    "model_f1_scores": all_model_f1_scores,
    "metric_used": "f1_score_weighted",
    "label_mapping": inverse_mapping_dict,
    "available_trained_models": list(get_models().keys())
    }


    with open(os.path.join(MODEL_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)
    
    accuracy_txt_path = os.path.join(MODEL_DIR, "accuracy.txt")
    if os.path.exists(accuracy_txt_path):
        os.remove(accuracy_txt_path)


def train():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df_for_meta = df.copy()

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
    models_to_train = get_models()
    best_model_instance, name_of_best_model, score_of_best_model, all_f1_scores = train_and_evaluate(
        models_to_train, X_train, X_test, y_train, y_test
    )

    print(f"Best model: {name_of_best_model} (F1-score weighted: {score_of_best_model:.4f})")
    print("Saving model and metadata...")
    save_artifacts(best_model_instance, name_of_best_model, score_of_best_model, all_f1_scores, inverse_mapping, df_for_meta)

    print("Training complete!")

if __name__ == "__main__":
    train()