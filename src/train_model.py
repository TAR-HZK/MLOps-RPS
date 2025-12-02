import pandas as pd
import os
import mlflow
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# 1. Setup Dynamic Paths (Works on Docker AND GitHub)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "include", "models")
MLFLOW_DIR = os.path.join(BASE_DIR, "include", "mlflow_artifacts")

def train(file_path):
    print(f"🚂 Starting training using data: {file_path}")
    df = pd.read_csv(file_path)

    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # -------------------------------
    # SMART MLFLOW SETUP
    # -------------------------------
    # If we are on GitHub Actions, use Dagshub Remote. 
    # If we are Local (Docker), use the File Store.
    dagshub_uri = os.getenv("MLFLOW_TRACKING_URI")
    
    if dagshub_uri:
        print(f"🌍 Detected Remote MLflow: {dagshub_uri}")
        mlflow.set_tracking_uri(dagshub_uri)
    else:
        print(f"🏠 Detected Local Environment. Logging to: {MLFLOW_DIR}")
        os.makedirs(MLFLOW_DIR, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
        
    mlflow.set_experiment("rps_experiment")

    # Close any active run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    model_path = os.path.join(MODEL_DIR, f"rf_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")

    # ... (Rest of logic: Case 1 Dummy, Case 2 Real) ...
    # Copy the EXACT same logic as before for training below
    
    # CASE 1: Dummy
    if len(df) < 10:
        print(f"⚠ Not enough data ({len(df)} rows). Saving DUMMY MODEL.")
        dummy_model = {"model_type": "dummy", "mean_temp": df["temperature"].mean()}
        joblib.dump(dummy_model, model_path)
        return model_path

    # CASE 2: Real Training
    features = ["hour", "temperature", "temp_lag1", "temp_rolling3"]
    target = "target_6h" if "target_6h" in df.columns else "temperature"
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)

        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        
        # Log Model & Artifacts
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        print(f"✔ Trained model. MAE: {mae:.2f} | R2: {r2:.2f}")

    return model_path

if __name__ == "__main__":
    import sys
    # Handle CLI args or default
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        default_path = os.path.join(BASE_DIR, "include", "data", "processed", "processed_data.csv")
        train(default_path)