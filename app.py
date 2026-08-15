import os
import joblib
import numpy as np
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

app = FastAPI(title="Rainfall Prediction Web App")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Mount static files and templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

MODEL_PATH = os.path.join(BASE_DIR, "models", "rainfall_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")

model = None
scaler = None

# Load model and scaler safely
try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("✅ Model and Scaler loaded successfully.")
    else:
        print("⚠️ Model or Scaler file missing on server.")
except Exception as e:
    print(f"❌ Error loading files: {e}")

class WeatherInput(BaseModel):
    Temp3pm: float
    Humidity3pm: float
    WindSpeed3pm: float
    Pressure3pm: float

# UI Home Page Endpoint
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={"prediction": None}
    )

# UI Form Submission Endpoint
@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    Temp3pm: float = Form(...),
    Humidity3pm: float = Form(...),
    WindSpeed3pm: float = Form(...),
    Pressure3pm: float = Form(...)
):
    if model is None or scaler is None:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"prediction": "Error: Model or Scaler not loaded on server.", "result_class": "rain"}
        )

    try:
        input_data = np.array([[float(Temp3pm), float(Humidity3pm), float(WindSpeed3pm), float(Pressure3pm)]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]

        if prediction == 1:
            result_text = "🌧️ Rain Expected Tomorrow! Carry an umbrella."
            result_class = "rain"
        else:
            result_text = "☀️ No Rain Expected Tomorrow. Enjoy the clear day!"
            result_class = "no-rain"

        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"prediction": result_text, "result_class": result_class}
        )
    except Exception as e:
        return templates.TemplateResponse(
            request=request,
            name="index.html",
            context={"prediction": f"Prediction Error: {str(e)}", "result_class": "rain"}
        )

# REST API Endpoint (Swagger / JSON)
@app.post("/predict")
def predict_api(data: WeatherInput):
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model or Scaler not loaded.")

    input_data = np.array([[data.Temp3pm, data.Humidity3pm, data.WindSpeed3pm, data.Pressure3pm]])
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    return {"RainTomorrow": "Yes" if prediction == 1 else "No"}