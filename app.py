import os
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Rainfall Prediction Web App")

# Ensure static & templates directory mapping
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# File paths
MODEL_PATH = "models/rainfall_model.pkl"
SCALER_PATH = "models/scaler.pkl"

# Load Model and Scaler
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Model and Scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model/scaler: {e}")
    model = None
    scaler = None

class WeatherInput(BaseModel):
    Temp3pm: float
    Humidity3pm: float
    WindSpeed3pm: float
    Pressure3pm: float

# 1. UI Home Page Endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": None
    })

# 2. UI Form Submission Endpoint
@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    Temp3pm: float = Form(...),
    Humidity3pm: float = Form(...),
    WindSpeed3pm: float = Form(...),
    Pressure3pm: float = Form(...)
):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")

    # Prepare input and scale
    input_data = np.array([[Temp3pm, Humidity3pm, WindSpeed3pm, Pressure3pm]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if prediction == 1:
        result_text = "🌧️ Rain Expected Tomorrow! Carry an umbrella."
        result_class = "rain"
    else:
        result_text = "☀️ No Rain Expected Tomorrow. Enjoy the clear day!"
        result_class = "no-rain"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result_text,
        "result_class": result_class
    })

# 3. REST API Endpoint (For JSON access / Swagger)
@app.post("/predict")
def predict_api(data: WeatherInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")

    input_data = np.array([[data.Temp3pm, data.Humidity3pm, data.WindSpeed3pm, data.Pressure3pm]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    return {"RainTomorrow": "Yes" if prediction == 1 else "No"}