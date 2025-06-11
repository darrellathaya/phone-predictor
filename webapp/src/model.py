import os
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn

# Directory where models and related files will be saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
DATA_DIR = os.path.join(BASE_DIR, "..", "data", "raw", "train.csv")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

def convert_resolution(res_str):
    """
    Convert resolution string "widthxheight" to a single integer value by multiplication.
    If conversion fails, return 0.
    Example: "1080x2340" -> 1080*2340
    """
    try:
        w, h = res_str.lower().split('x')
        return int(w) * int(h)
    except:
        return 0

# Dictionary mapping algorithm names to their corresponding sklearn models
ALGORITHMS = {
    "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "gradient_boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42),
}

# Training model function
def train_model(df, algorithm_name):
    """
    Train a classification model on the given dataframe using the specified algorithm.
    
    Args:
        df (pd.DataFrame): DataFrame containing features and target.
        algorithm_name (str): The algorithm key to use ('random_forest', 'gradient_boosting').
    
    Returns:
        pipeline: Trained sklearn Pipeline including scaler and classifier.
        le_chipset: LabelEncoder for chipset feature.
        le_target: LabelEncoder for target variable price_range.
        acc: Accuracy score on validation set.
    """
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm_name}' not supported. Choose from {list(ALGORITHMS.keys())}")

    # Define paths for saving model, encoders, accuracy, etc.
    model_path = os.path.join(MODEL_DIR, f"{algorithm_name}_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, f"{algorithm_name}_chipset_encoder.pkl")
    target_encoder_path = os.path.join(MODEL_DIR, f"{algorithm_name}_target_encoder.pkl")
    accuracy_path = os.path.join(MODEL_DIR, f"{algorithm_name}_accuracy.txt")

    # Preprocess display_resolution: convert string to numeric value
    df['display_resolution'] = df['display_resolution'].apply(convert_resolution)

    # Select features and target
    features = ['ram', 'storage', 'display_resolution', 'chipset']
    X = df[features]
    y = df['price_range']

    # Encode 'chipset' categorical feature into numeric values
    le_chipset = LabelEncoder()
    X.loc[:, 'chipset'] = le_chipset.fit_transform(X['chipset'])

    # Encode target variable (price_range) into numeric classes
    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    # Split dataset into train and validation sets with stratification to preserve class ratios
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize classifier from dictionary based on selected algorithm
    classifier = ALGORITHMS[algorithm_name]

    # Create a pipeline: standard scaling followed by classification
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    # Train the model
    pipeline.fit(X_train, y_train)

    # Predict on validation set
    y_pred = pipeline.predict(X_val)

    # Evaluate performance
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)

    print(f"=== Classification Report for {algorithm_name} ===\n{report}")
    print(f"Accuracy: {acc:.4f}")

    # Remove old files if they exist to avoid conflicts
    for path in [model_path, encoder_path, target_encoder_path, accuracy_path]:
        if os.path.exists(path):
            os.remove(path)

    # Save the trained pipeline, feature encoder, and target encoder
    joblib.dump(pipeline, model_path)
    joblib.dump(le_chipset, encoder_path)
    joblib.dump(le_target, target_encoder_path)

    # Save accuracy score to text file
    with open(accuracy_path, "w") as f:
        f.write(str(acc))

    # Save metadata: original chipset and display_resolution strings for dropdowns or UI
    raw_df = pd.read_csv(DATA_DIR)
    chipset_list = sorted(raw_df['chipset'].dropna().unique().tolist())
    resolution_list = sorted(raw_df['display_resolution'].dropna().unique().tolist())

    meta = {
        "chipset_list": chipset_list,
        "resolution_list": resolution_list
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    print(f"Model, encoders, accuracy, and metadata saved successfully with prefix '{algorithm_name}'.")

    return pipeline, le_chipset, le_target, acc


def load_model(algorithm_name):
    """
    Load a trained model pipeline and encoders for a given algorithm.

    Args:
        algorithm_name (str): The algorithm name to load.

    Returns:
        pipeline: Loaded sklearn Pipeline.
        le_chipset: Loaded LabelEncoder for chipset feature.
        le_target: Loaded LabelEncoder for target variable.
    """
    model_path = os.path.join(MODEL_DIR, f"{algorithm_name}_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, f"{algorithm_name}_chipset_encoder.pkl")
    target_encoder_path = os.path.join(MODEL_DIR, f"{algorithm_name}_target_encoder.pkl")

    pipeline = joblib.load(model_path)
    le_chipset = joblib.load(encoder_path)
    le_target = joblib.load(target_encoder_path)

    return pipeline, le_chipset, le_target

def train_model(df, algorithm_name):
    if algorithm_name not in ALGORITHMS:
        raise ValueError(f"Algorithm '{algorithm_name}' not supported. Choose from {list(ALGORITHMS.keys())}")

    # Define paths
    model_path = os.path.join(MODEL_DIR, f"{algorithm_name}_model.pkl")
    encoder_path = os.path.join(MODEL_DIR, f"{algorithm_name}_chipset_encoder.pkl")
    target_encoder_path = os.path.join(MODEL_DIR, f"{algorithm_name}_target_encoder.pkl")
    accuracy_path = os.path.join(MODEL_DIR, f"{algorithm_name}_accuracy.txt")

    # Preprocess
    df['display_resolution'] = df['display_resolution'].apply(convert_resolution)
    features = ['ram', 'storage', 'display_resolution', 'chipset']
    X = df[features]
    y = df['price_range']

    le_chipset = LabelEncoder()
    X.loc[:, 'chipset'] = le_chipset.fit_transform(X['chipset'])

    le_target = LabelEncoder()
    y = le_target.fit_transform(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    classifier = ALGORITHMS[algorithm_name]
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    # === MLflow: Start experiment run
    mlflow.set_experiment("phone_price_prediction")
    with mlflow.start_run(run_name=f"{algorithm_name}_run"):

        # === MLflow: Log parameters
        mlflow.log_param("algorithm", algorithm_name)
        if hasattr(classifier, 'n_estimators'):
            mlflow.log_param("n_estimators", classifier.n_estimators)
        if hasattr(classifier, 'learning_rate'):
            mlflow.log_param("learning_rate", classifier.learning_rate)

        # Train and evaluate
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        report = classification_report(y_val, y_pred)

        print(f"=== Classification Report for {algorithm_name} ===\n{report}")
        print(f"Accuracy: {acc:.4f}")

        # === MLflow: Log metrics
        mlflow.log_metric("accuracy", acc)

        # === MLflow: Log model
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        # === MLflow: Log additional artifacts (report, encoders, meta.json)
        report_path = os.path.join(MODEL_DIR, f"{algorithm_name}_report.txt")
        with open(report_path, "w") as f:
            f.write(report)
        mlflow.log_artifact(report_path)

        # Save & log encoders and metadata
        joblib.dump(pipeline, model_path)
        joblib.dump(le_chipset, encoder_path)
        joblib.dump(le_target, target_encoder_path)
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(encoder_path)
        mlflow.log_artifact(target_encoder_path)

        with open(accuracy_path, "w") as f:
            f.write(str(acc))
        mlflow.log_artifact(accuracy_path)

        raw_df = pd.read_csv(DATA_DIR)
        chipset_list = sorted(raw_df['chipset'].dropna().unique().tolist())
        resolution_list = sorted(raw_df['display_resolution'].dropna().unique().tolist())

        meta = {
            "chipset_list": chipset_list,
            "resolution_list": resolution_list
        }
        with open(META_PATH, "w") as f:
            json.dump(meta, f)
        mlflow.log_artifact(META_PATH)

        print(f"Model, encoders, accuracy, and metadata saved successfully with prefix '{algorithm_name}'.")

    return pipeline, le_chipset, le_target, acc