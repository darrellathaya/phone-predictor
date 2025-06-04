import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Paths
DATA_PATH = os.path.join("data", "raw", "train.csv")
CLASS_MODEL_PATH = os.path.join("models", "price_range_model.pkl")
REGRESS_MODEL_PATH = os.path.join("models", "price_regression_model.pkl")
ENCODER_PATH = os.path.join("models", "chipset_encoder.pkl")
ACCURACY_PATH = os.path.join("models", "accuracy.txt")
META_PATH = os.path.join("models", "meta.json")
REGRESSION_METRICS_PATH = os.path.join("models", "regression_metrics.txt")


def train():
    print("🚀 Starting model training...")

    df = pd.read_csv(DATA_PATH)

    def convert_resolution(res_str):
        try:
            w, h = res_str.lower().split('x')
            return int(w) * int(h)
        except:
            return 0

    df['display_resolution'] = df['display_resolution'].apply(convert_resolution)

    # Features
    features = ['ram', 'storage', 'display_resolution', 'chipset']
    X = df[features]
    y_class = df['price_range']                 # Classification target
    y_reg = df['price']                         # Regression target (add this column if not present)

    # Encode chipset
    le = LabelEncoder()
    X.loc[:, 'chipset'] = le.fit_transform(X['chipset'])

    # Train/Test split
    X_train, X_val, y_train_class, y_val_class = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    _, X_val_reg, y_train_reg, y_val_reg = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # --- CLASSIFICATION MODEL ---
    class_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    class_pipeline.fit(X_train, y_train_class)
    y_pred_class = class_pipeline.predict(X_val)
    class_acc = accuracy_score(y_val_class, y_pred_class)
    print("✅ Classification Model Trained")
    print(classification_report(y_val_class, y_pred_class))

    # --- REGRESSION MODEL ---
    reg_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    reg_pipeline.fit(X_train, y_train_reg)
    y_pred_reg = reg_pipeline.predict(X_val_reg)
    reg_r2 = r2_score(y_val_reg, y_pred_reg)
    reg_mse = mean_squared_error(y_val_reg, y_pred_reg)
    print("✅ Regression Model Trained")
    print(f"R²: {reg_r2:.4f}, MSE: {reg_mse:.4f}")

    # --- SAVE MODELS & METADATA ---
    joblib.dump(class_pipeline, CLASS_MODEL_PATH)
    joblib.dump(reg_pipeline, REGRESS_MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    # Save accuracy
    with open(ACCURACY_PATH, "w") as f:
        f.write(f"{class_acc:.4f}")

    # Save regression metrics
    with open(REGRESSION_METRICS_PATH, "w") as f:
        f.write(f"R2: {reg_r2:.4f}\nMSE: {reg_mse:.4f}")

    # Save metadata
    raw_df = pd.read_csv(DATA_PATH)
    chipset_list = sorted(raw_df['chipset'].dropna().unique().tolist())
    resolution_list = sorted(raw_df['display_resolution'].dropna().unique().tolist())

    meta = {
        "chipset_list": chipset_list,
        "resolution_list": resolution_list
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    print("💾 All models, encoder, accuracy, and metadata saved.")
    print(f"📍 Files written to: {os.path.abspath('models')}")

if __name__ == "__main__":
    from sklearn.metrics import r2_score, mean_squared_error
    train()