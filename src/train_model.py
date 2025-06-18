import os
import json
import joblib
import pandas as pd
from typing import Tuple, Dict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline # Pastikan ini diimpor
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import mlflow
# from mlflow import MlflowClient # Tidak diperlukan jika hanya menggunakan set_experiment dan logging standar

# === Constants ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
os.makedirs(MODEL_DIR, exist_ok=True)
EXPERIMENT_NAME = "PhonePricePrediction"

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

def make_pipeline_func(classifier): # Diganti nama untuk kejelasan
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", classifier)
    ])

def get_models():
    return {
        "RandomForest": make_pipeline_func(RandomForestClassifier(random_state=42, class_weight='balanced')),
        "SVM": make_pipeline_func(SVC(probability=True, class_weight='balanced', random_state=42)),
    }

def train_and_evaluate(models, X_train, X_test, y_train, y_test):
    # mlflow.set_tracking_uri("file:./mlruns") # Bisa ditambahkan jika perlu, defaultnya juga ./mlruns
    mlflow.set_experiment(EXPERIMENT_NAME)
    best_overall_score = 0
    best_model_obj = None
    best_model_overall_name = ""
    model_f1_scores = {} 

    for name, model_pipeline in models.items():
        with mlflow.start_run(run_name=name):
            print(f"Training {name}...")
            model_pipeline.fit(X_train, y_train)
            
            model_path_pkl = os.path.join(MODEL_DIR, f"{name}.pkl") # Simpan setiap model
            joblib.dump(model_pipeline, model_path_pkl)
            print(f"Saved {name} model to {model_path_pkl}")

            preds = model_pipeline.predict(X_test)
            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds, average='weighted')
            
            model_f1_scores[name] = f1 

            print(f"{name} - Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_weighted", f1)
            # mlflow.sklearn.log_model(model_pipeline, name) # Ini bisa ditambahkan jika Anda ingin log model ke artifact MLflow juga

            if f1 > best_overall_score:
                best_overall_score = f1
                best_model_obj = model_pipeline
                best_model_overall_name = name
    
    return best_model_obj, best_model_overall_name, best_overall_score, model_f1_scores

def save_artifacts(best_model_object, best_model_name_val, best_overall_f1_score, all_model_f1_scores, inverse_mapping_dict, df_original):
    joblib.dump(best_model_object, os.path.join(MODEL_DIR, "price_range_model.pkl")) # Model terbaik default

    meta = {
        "chipset_list": sorted(df_original['chipset'].dropna().unique().tolist()),
        "resolution_list": ["720p", "1080p", "2k+"],
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
    best_model_instance, name_of_best_model, score_of_best_model, all_f1s = train_and_evaluate(
        models_to_train, X_train, X_test, y_train, y_test
    )

    print(f"Best model: {name_of_best_model} (F1-score weighted: {score_of_best_model:.4f})")
    print("Saving model and metadata...")
    save_artifacts(best_model_instance, name_of_best_model, score_of_best_model, all_f1s, inverse_mapping, df_for_meta)

    print("Training complete!")

if __name__ == "__main__":
    train()