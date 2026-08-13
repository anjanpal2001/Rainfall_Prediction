import os
import boto3
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app=FastAPI(title="Rainfall Prediction API",version="1.0",description="API for predicting rainfall using a trained Random Forest model.")
S3_BUCKET=os.getenv("S3_BUCKET_NAME","rainfall-prediction-bucket-2026")
AWS_REGION=os.getenv("AWS_REGION","us-east-1")
LOCAL_MODEL_PATH="models/rainfall_model.pkl"
class WeatherInput(BaseModel):
    Temp3pm: float
    Humidity3pm: float
    WindSpeed3pm: float
    Pressure3pm: float
def download_model_from_s3():
    if not os.path.exists(LOCAL_MODEL_PATH):
        os.makedirs("models", exist_ok=True)
        print("Downloading the model from S3...")
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3.download_file(S3_BUCKET, "models/rainfall_model.pkl", LOCAL_MODEL_PATH)
        print("Model downloaded successfully.")

@app.get("/")
def home():
    return {"message": "Welcome to the Rainfall Prediction API. Use the /predict endpoint to get predictions."}

@app.post("/predict")
def predict_rainfall(input_data:WeatherInput):
    try:
        download_model_from_s3()
        model=joblib.load(LOCAL_MODEL_PATH)
        features=[[
            input_data.Temp3pm,
            input_data.Humidity3pm,
            input_data.WindSpeed3pm,
            input_data.Pressure3pm
        ]]
        prediction=model.predict(features)[0]
        probability=model.predict_proba(features)[0][1]
        result="Yes(Rain Expected)" if prediction==1 else "No(Rain Not Expected)"
        return {
           "Rain Prediction": result,
           "Probability of Rain": round(probability, 4) ,
           "InputData": input_data.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {e}")
    