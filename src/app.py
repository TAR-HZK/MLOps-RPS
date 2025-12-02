from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os
import uvicorn
import joblib
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Histogram

# Initialize FastAPI
app = FastAPI(title="RPS Weather Forecaster 🌦️")

# ---------------------------------------------------------
# MONITORING SETUP
# ---------------------------------------------------------
# 1. Automatic Instrumentation (Latency, Request Count, Errors)
instrumentator = Instrumentator().instrument(app).expose(app)

# 2. Custom Metrics (Drift Detection)
# We track the distribution of our predictions. 
# If this histogram shifts significantly, we have "Prediction Drift".
PREDICTION_GAUGE = Histogram(
    "prediction_output", 
    "Forecasted Temperature (6h)", 
    buckets=[0, 5, 10, 15, 20, 25, 30, 35, 40]
)

class WeatherInput(BaseModel):
    hour: int
    temperature: float
    temp_lag1: float
    temp_rolling3: float

model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = "include/models/latest_model.pkl" 
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"✅ Model loaded from {model_path}")
    else:
        print("⚠ Model not found!")

@app.get("/")
def home():
    return {"message": "RPS Weather API is Live! 🚀"}

@app.post("/predict")
def predict(data: WeatherInput):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    
    # --- LOG METRIC FOR GRAFANA ---
    PREDICTION_GAUGE.observe(prediction)
    
    return {
        "input": data.dict(),
        "forecast_6h": round(prediction, 2)
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)