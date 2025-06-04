import pandas as pd
import os
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

DATA_PATH = os.path.join("data", "raw", "train.csv")
MODEL_PATH = os.path.join("models", "price_range_model.pkl")
ENCODER_PATH = os.path.join("models", "chipset_encoder.pkl")
ACCURACY_PATH = os.path.join("models", "accuracy.txt")
META_PATH = os.path.join("models", "meta.json")  # simpan pilihan dropdown

def train():
    df = pd.read_csv(DATA_PATH)

    def convert_resolution(res_str):
        try:
            w, h = res_str.lower().split('x')
            return int(w) * int(h)
        except:
            return 0
    df['display_resolution'] = df['display_resolution'].apply(convert_resolution)

    features = ['ram', 'storage', 'display_resolution', 'chipset']
    X = df[features]
    y = df['price_range']

    le = LabelEncoder()
    X.loc[:, 'chipset'] = le.fit_transform(X['chipset'])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    report = classification_report(y_val, y_pred)
    print("=== Classification Report ===\n", report)
    print(f"Accuracy: {acc:.4f}")
    
    if os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)
    joblib.dump(pipeline, MODEL_PATH)
    joblib.dump(le, ENCODER_PATH)

    # Simpan akurasi
    with open(ACCURACY_PATH, "w") as f:
        f.write(str(acc))

    # Simpan pilihan dropdown chipset dan display_resolution (yang asli string sebelum di convert)
    raw_df = pd.read_csv(DATA_PATH)
    chipset_list = sorted(raw_df['chipset'].dropna().unique().tolist())
    resolution_list = sorted(raw_df['display_resolution'].dropna().unique().tolist())

    meta = {
        "chipset_list": chipset_list,
        "resolution_list": resolution_list
    }

    with open(META_PATH, "w") as f:
        json.dump(meta, f)

    print("Model, encoder, akurasi, dan meta (pilihan dropdown) berhasil disimpan.")

if __name__ == "__main__":
    train()
