from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import numpy as np
import json
import glob

app = FastAPI()
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "..", "templates"))


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
META_PATH = os.path.join(MODEL_DIR, "meta.json")

# Load meta dropdown choices once
with open(META_PATH, "r") as f:
    meta = json.load(f)
chipset_list = meta.get("chipset_list", [])
resolution_list = meta.get("resolution_list", [])

def convert_resolution(res_str):
    try:
        w, h = res_str.lower().split('x')
        return int(w) * int(h)
    except:
        return 0

def find_best_model():
    """
    Scans the models directory for all *_model.pkl and corresponding accuracy files,
    returns the best model path, chipset encoder path, target encoder path, and accuracy
    """
    best_acc = -1
    best_model_path = None
    best_chipset_encoder_path = None
    best_target_encoder_path = None

    # Pattern to match model files like random_forest_model.pkl, gradient_boosting_model.pkl, etc
    model_files = glob.glob(os.path.join(MODEL_DIR, "*_model.pkl"))

    for model_path in model_files:
        # Extract prefix: e.g. 'random_forest' from 'random_forest_model.pkl'
        prefix = os.path.basename(model_path).replace("_model.pkl", "")

        accuracy_path = os.path.join(MODEL_DIR, f"{prefix}_accuracy.txt")
        chipset_encoder_path = os.path.join(MODEL_DIR, f"{prefix}_chipset_encoder.pkl")
        target_encoder_path = os.path.join(MODEL_DIR, f"{prefix}_target_encoder.pkl")

        if os.path.exists(accuracy_path) and os.path.exists(chipset_encoder_path) and os.path.exists(target_encoder_path):
            with open(accuracy_path, "r") as f:
                try:
                    acc = float(f.read())
                    if acc > best_acc:
                        best_acc = acc
                        best_model_path = model_path
                        best_chipset_encoder_path = chipset_encoder_path
                        best_target_encoder_path = target_encoder_path
                except:
                    pass

    if best_model_path is None:
        raise Exception("No valid models found.")

    return best_model_path, best_chipset_encoder_path, best_target_encoder_path, best_acc

# Load best model and encoders at startup
try:
    MODEL_PATH, ENCODER_PATH, TARGET_ENCODER_PATH, BEST_ACCURACY = find_best_model()
    model = joblib.load(MODEL_PATH)
    chipset_encoder = joblib.load(ENCODER_PATH)
    target_encoder = joblib.load(TARGET_ENCODER_PATH)
except Exception as e:
    print(f"Error loading models: {e}")
    model = None
    chipset_encoder = None
    target_encoder = None
    BEST_ACCURACY = None


@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    acc = None
    if BEST_ACCURACY is not None:
        acc = round(BEST_ACCURACY * 100, 2)
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None,
        "error": None,
        "accuracy": acc,
        "chipset_list": chipset_list,
        "resolution_list": resolution_list
    })

@app.post("/", response_class=HTMLResponse)
def form_post(
    request: Request,
    ram: int = Form(...),
    storage: int = Form(...),
    display_resolution: str = Form(...),
    chipset: str = Form(...)
):
    if model is None or chipset_encoder is None or target_encoder is None:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "error": "Model not loaded properly.",
            "accuracy": None,
            "chipset_list": chipset_list,
            "resolution_list": resolution_list,
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "ram": ram,
            "storage": storage
        })
    try:
        display_res_value = convert_resolution(display_resolution)
        chipset_encoded = chipset_encoder.transform([chipset])[0]

        input_data = np.array([[ram, storage, display_res_value, chipset_encoded]])

        prediction_encoded = model.predict(input_data)[0]

        # Decode numeric prediction back to original label
        prediction = target_encoder.inverse_transform([prediction_encoded])[0]

        acc = round(BEST_ACCURACY * 100, 2) if BEST_ACCURACY is not None else None

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": prediction,
            "error": None,
            "accuracy": acc,
            "chipset_list": chipset_list,
            "resolution_list": resolution_list,
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "ram": ram,
            "storage": storage
        })
    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": None,
            "error": str(e),
            "accuracy": None,
            "chipset_list": chipset_list,
            "resolution_list": resolution_list,
            "selected_chipset": chipset,
            "selected_resolution": display_resolution,
            "ram": ram,
            "storage": storage
        })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
