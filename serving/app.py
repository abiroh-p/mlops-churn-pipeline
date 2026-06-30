import os
import mlflow
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_URI = "models:/churn-classifier/Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
model = mlflow.sklearn.load_model(MODEL_URI)
from prometheus_fastapi_instrumentator import Instrumentator

app = FastAPI(title="Churn Prediction API")
Instrumentator().instrument(app).expose(app)

class CustomerFeatures(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    df = pd.DataFrame([customer.dict()])
    prob = model.predict_proba(df)[0, 1]
    pred = int(prob >= 0.5)
    return {"churn_prediction": pred, "churn_probability": round(float(prob), 4)}