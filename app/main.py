from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import joblib
import os
import numpy as np
import json

app = FastAPI()
templates = Jinja2Templates(directory="templates")

MODEL_PATH = os.path.join("models", "price_range_model.pkl")
ENCODER_PATH = os.path.join("models", "chipset_encoder.pkl")
ACCURACY_PATH = os.path.join("models", "accuracy.txt")
META_PATH = os.path.join("models", "meta.json")

model = joblib.load(MODEL_PATH)
chipset_encoder = joblib.load(ENCODER_PATH)

# Load meta dropdown choices
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

@app.get("/", response_class=HTMLResponse)
def form_get(request: Request):
    acc = None
    try:
        with open(ACCURACY_PATH, "r") as f:
            acc = round(float(f.read()) * 100, 2)
    except:
        acc = None
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
    display_resolution: str = Form(...),  # ambil string dari dropdown
    chipset: str = Form(...)
):
    try:
        display_res_value = convert_resolution(display_resolution)
        chipset_encoded = chipset_encoder.transform([chipset])[0]

        input_data = np.array([[ram, storage, display_res_value, chipset_encoded]])

        prediction = model.predict(input_data)[0]
        label_map = {
            0: "Low Cost",
            1: "Medium Cost",
            2: "High Cost",
            3: "Very High Cost"
        }

        with open(ACCURACY_PATH, "r") as f:
            acc = round(float(f.read()) * 100, 2)

        return templates.TemplateResponse("index.html", {
            "request": request,
            "prediction": label_map[prediction],
            "error": None,
            "accuracy": acc,
            "chipset_list": chipset_list,
            "resolution_list": resolution_list,
            # untuk tetap menampilkan pilihan yang dipilih di form
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
