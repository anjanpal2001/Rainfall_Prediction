import os
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Rainfall Prediction API")

# Static and Template Configuration
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load Trained Model & Scaler
MODEL_PATH = "models/rainfall_model.pkl"
SCALER_PATH = "models/scaler.pkl"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

class WeatherInput(BaseModel):
    Temp3pm: float
    Humidity3pm: float
    WindSpeed3pm: float
    Pressure3pm: float

# Home Page (UI)
@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Form Submission endpoint (UI Result)
@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    Temp3pm: float = Form(...),
    Humidity3pm: float = Form(...),
    WindSpeed3pm: float = Form(...),
    Pressure3pm: float = Form(...)
):
    input_data = np.array([[Temp3pm, Humidity3pm, WindSpeed3pm, Pressure3pm]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    
    if prediction == 1:
        result_text = "🌧️ Expect Rain Tomorrow!"
        result_class = "rain"
    else:
        result_text = "☀️ No Rain Tomorrow. Enjoy the Sunshine!"
        result_class = "no-rain"

    return templates.TemplateResponse("index.html", {
        "request": request,
        "prediction": result_text,
        "result_class": result_class
    })

# JSON API Endpoint
@app.post("/predict")
def predict_json(data: WeatherInput):
    input_data = np.array([[data.Temp3pm, data.Humidity3pm, data.WindSpeed3pm, data.Pressure3pm]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    result = "Yes" if prediction == 1 else "No"
    return {"RainTomorrow": result}