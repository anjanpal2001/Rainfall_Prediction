import os
import pandas as pd
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import boto3
from dotenv import load_dotenv

# load aws_ keys from .env
load_dotenv()

# ১. make conform to create folder

os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
csv_file_path = "data/rainfall_data.csv"

# ২. Creation of data set

if not os.path.exists(csv_file_path):
    print("Creating local dataset for training...")
    np.random.seed(42)
    n_samples = 8000
    
    data = {
        'Temp3pm': np.random.uniform(10, 42, n_samples),
        'Humidity3pm': np.random.uniform(15, 95, n_samples),
        'WindSpeed3pm': np.random.uniform(5, 55, n_samples),
        'Pressure3pm': np.random.uniform(995, 1030, n_samples),
        'RainTomorrow': np.random.choice(['Yes', 'No'], size=n_samples, p=[0.3, 0.7])
    }
    
    df_new = pd.DataFrame(data)
    df_new.to_csv(csv_file_path, index=False)
    print(f"Dataset created and saved to {csv_file_path}")

# 3.Load the data sert and process
df = pd.read_csv(csv_file_path)

features = ['Temp3pm', 'Humidity3pm', 'WindSpeed3pm', 'Pressure3pm']

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_filtered = df[features + ['RainTomorrow']].dropna()
df_filtered['RainTomorrow'] = df_filtered['RainTomorrow'].map({'Yes': 1, 'No': 0})

# 4. model training 
X_train, X_test, y_train, y_test = train_test_split(
    df_filtered[features], df_filtered['RainTomorrow'], test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# 5. save the model
local_model_path = "models/rainfall_model.pkl"
joblib.dump(model, local_model_path)
print("Model trained and saved successfully at", local_model_path)

# upload the model to s3 bucket
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "rainfall-prediction-bucket-2026")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
# S3 BUCKET PUSH 
def upload_to_s3():
    try:
        s3 = boto3.client('s3', region_name=AWS_REGION)
        s3.upload_file(local_model_path, S3_BUCKET, "models/rainfall_model.pkl")
        print(f"Model uploaded to S3 Bucket ({S3_BUCKET}) successfully.")
    except Exception as e:
        print(f"Error occurred while uploading to S3: {e}")

if __name__ == "__main__":
    upload_to_s3()