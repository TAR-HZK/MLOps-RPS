import pandas as pd
import os
import mlflow
import joblib
import numpy as np  # <--- Essential import for RMSE calculation
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

# 1. Setup Absolute Paths (Docker Safe)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "include", "models")
MLFLOW_DIR = os.path.join(BASE_DIR, "include", "mlflow_artifacts")

def train(file_path):
    print(f"🚂 Starting training using data: {file_path}")
    df = pd.read_csv(file_path)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MLFLOW_DIR, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("rps_experiment")

    # Close any active run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    model_path = os.path.join(MODEL_DIR, f"rf_model_{datetime.now().strftime('%Y%m%d%H%M%S')}.pkl")

    # -------------------------------
    # CASE 1: Not enough data → Dummy Model
    # -------------------------------
    if len(df) < 10:
        print(f"⚠ Not enough data ({len(df)} rows). Saving DUMMY MODEL.")
        dummy_model = {
            "model_type": "dummy",
            "prediction_rule": "always_return_mean_temp",
            "mean_temperature": df["temperature"].mean() if "temperature" in df else 0
        }
        joblib.dump(dummy_model, model_path)
        return model_path

    # -------------------------------
    # CASE 2: Train Real Model
    # -------------------------------
    
    # Define Features
    features = ["hour", "temperature", "temp_lag1", "temp_rolling3"]
    
    # Check if necessary columns exist
    missing_cols = [c for c in features if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing feature columns: {missing_cols}")

    # DEFINE TARGET
    if "target_6h" in df.columns:
        target = "target_6h"
        print("🎯 Target: Forecasting 6 hours ahead (target_6h)")
    else:
        print("⚠ 'target_6h' not found! Falling back to 'temperature' (Nowcasting) for testing.")
        target = "temperature"

    X = df[features]
    y = df[target]

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict & metrics
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        
        # --- FIXED RMSE CALCULATION ---
        # Old way (deprecated): mean_squared_error(..., squared=False)
        # New way (compatible with all versions):
        mse = mean_squared_error(y_test, preds)
        rmse = np.sqrt(mse)
        
        r2 = r2_score(y_test, preds)

        # Log MLflow
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("target_variable", target)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_artifact(file_path)

        # Save & log model
        joblib.dump(model, model_path)
        mlflow.log_artifact(model_path)

        # Feature importance
        feat_imp = pd.DataFrame({
            "feature": X.columns,
            "importance": model.feature_importances_
        }).sort_values(by="importance", ascending=False)
        
        feat_imp_path = os.path.join(MODEL_DIR, "feature_importance.csv")
        feat_imp.to_csv(feat_imp_path, index=False)
        mlflow.log_artifact(feat_imp_path)

        print(f"✔ Trained real model saved to {model_path}")
        print(f"📊 MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

    return model_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        train(sys.argv[1])
    else:
        default_path = os.path.join(BASE_DIR, "include", "data", "processed", "processed_data.csv")
        if os.path.exists(default_path):
            train(default_path)
        else:
            print("❌ No input file provided and default processed file not found.")