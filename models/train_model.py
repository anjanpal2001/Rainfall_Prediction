import os
import pandas as pd
import joblib
import numpy as np
import boto3
from dotenv import load_dotenv

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score

# Load environment variables
load_dotenv()

# 1. Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("data", exist_ok=True)
csv_file_path = "data/rainfall_data.csv"

# 2. Create sample dataset if it does not exist
# 2. Creation of realistic dataset
if not os.path.exists(csv_file_path):
    print("Creating realistic local dataset for training...")
    np.random.seed(42)
    n_samples = 20000
    
    temp = np.random.uniform(10, 55, n_samples)
    humidity = np.random.uniform(15, 97, n_samples)
    wind = np.random.uniform(5, 55, n_samples)
    pressure = np.random.uniform(995, 1030, n_samples)
    
    # iN genral logic that if humidity and wind speed increases and pressure decrease probabu=ility of rain is increases
    rain_score = (
        0.06 * (humidity - 50) + 
        0.03 * (wind - 25) - 
        0.08 * (pressure - 1012) + 
        np.random.normal(0, 0.6, n_samples)
    )
    
    # creating probability usin Sigmoid function
    prob = 1 / (1 + np.exp(-rain_score))
    rain_tomorrow = ['Yes' if p > 0.5 else 'No' for p in prob]
    
    data = {
        'Temp3pm': temp,
        'Humidity3pm': humidity,
        'WindSpeed3pm': wind,
        'Pressure3pm': pressure,
        'RainTomorrow': rain_tomorrow
    }
    
    pd.DataFrame(data).to_csv(csv_file_path, index=False)
    print(f"Dataset created and saved to {csv_file_path}")

# 3. Load dataset and preprocess
df = pd.read_csv(csv_file_path)
features = ['Temp3pm', 'Humidity3pm', 'WindSpeed3pm', 'Pressure3pm']

for col in features:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_filtered = df[features + ['RainTomorrow']].dropna()
df_filtered['RainTomorrow'] = df_filtered['RainTomorrow'].map({'Yes': 1, 'No': 0})
X = df_filtered[features]
y = df_filtered['RainTomorrow']

# Standard Scaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Define 6 ML Models and Hyperparameter Grids
models_and_params = {
    'RandomForest': (
        RandomForestClassifier(random_state=42),
        {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=42),
        {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    ),
    'AdaBoost': (
        AdaBoostClassifier(random_state=42),
        {'n_estimators': [50, 100], 'learning_rate': [0.01, 0.1]}
    ),
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [3, 5, 10], 'criterion': ['gini', 'entropy']}
    ),
    'LogisticRegression': (
        LogisticRegression(random_state=42),
        {'C': [0.1, 1.0, 10.0]}
    ),
    'SupportVectorMachine': (
        SVC(probability=True, random_state=42),
        {'C': [0.1, 1.0, 10.0], 'kernel': ['rbf', 'linear']}
    ),
    'RidgeClassifier':(
        RidgeClassifier(random_state=42),
        {"alpha":[0.01,0.1,1,10],"solver":['auto', 'svd', 'cholesky', 'lsqr']}
    )
}

best_model = None
best_accuracy = 0.0
best_model_name = ""

print("---------- Starting Hyperparameter Tuning on 7 ML Models ----------")
for name, (model, params) in models_and_params.items():
    grid = GridSearchCV(model, params, cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train, y_train)
    tuned_model = grid.best_estimator_
    y_pred = tuned_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"Model: {name:<22} | Best Params: {grid.best_params_} | Test Accuracy: {acc * 100:.4f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = tuned_model
        best_model_name = name

print("\n" + "="*50)
print(f"BEST MODEL : {best_model_name} with Accuracy: {best_accuracy:.4f}")
print("="*50)

# File paths for saving
local_model_path = "models/rainfall_model.pkl"
scaler_path = "models/scaler.pkl"

# Save the best trained model and scaler objects
joblib.dump(best_model, local_model_path)
joblib.dump(scaler, scaler_path)
print("Model and Scaler successfully saved locally.")

# S3 configuration
S3_BUCKET = os.getenv("S3_BUCKET_NAME", "rainfall-prediction-bucket-2026")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")

def upload_to_s3():
    if not S3_BUCKET:
        raise ValueError("S3_BUCKET_NAME environment variable is not set.")
    
    s3 = boto3.client('s3', region_name=AWS_REGION)
    
    #* Check if bucket exists, create if missing
    try:
        s3.head_bucket(Bucket=S3_BUCKET)
        print(f"Bucket '{S3_BUCKET}' exists.")
    except Exception:
        print(f"Bucket '{S3_BUCKET}' does not exist. Creating it now...")
        if AWS_REGION == "us-east-1":
            s3.create_bucket(Bucket=S3_BUCKET)
        else:
            s3.create_bucket(
                Bucket=S3_BUCKET,
                CreateBucketConfiguration={'LocationConstraint': AWS_REGION}
            )
        print(f"Bucket '{S3_BUCKET}' created successfully.")

    #! Upload model and scaler files to S3
    try:
        s3.upload_file(local_model_path, S3_BUCKET, "models/rainfall_model.pkl")
        print(f"Successfully uploaded {local_model_path} to S3 Bucket '{S3_BUCKET}'.")
        s3.upload_file(scaler_path, S3_BUCKET, "models/scaler.pkl")
        print(f"Successfully uploaded {scaler_path} to S3 Bucket '{S3_BUCKET}'.")
    except Exception as e:
        print(f"Critical Upload Failure: {e}")
        raise e

if __name__ == "__main__":
    upload_to_s3()