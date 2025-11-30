import pandas as pd
import os
import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import datetime

PROCESSED_DIR = "include/data/processed"
MODEL_DIR = "include/models"
MLFLOW_DIR = "include/mlflow_artifacts"

def train(file_path):
    df = pd.read_csv(file_path)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(MLFLOW_DIR, exist_ok=True)

    # MLflow setup
    mlflow.set_tracking_uri(f"file:{MLFLOW_DIR}")
    mlflow.set_experiment("rps_experiment")

    # Close any active run
    if mlflow.active_run() is not None:
        mlflow.end_run()

    model_path = f"{MODEL_DIR}/rf_model_{os.path.basename(file_path).replace('.csv','.pkl')}"

    # -------------------------------
    # CASE 1: Not enough data → Dummy Model
    # -------------------------------
    if len(df) < 2:
        print(f"⚠ Not enough data ({len(df)} rows). Saving DUMMY MODEL.")
        dummy_model = {
            "model_type": "dummy",
            "prediction_rule": "always_return_mean_temp",
            "mean_temperature": df["temperature"].mean() if "temperature" in df else 0
        }
        joblib.dump(dummy_model, model_path)

        with mlflow.start_run(run_name=os.path.basename(file_path)):
            mlflow.log_param("model_type", "dummy")
            mlflow.log_param("reason", "insufficient_data")
            mlflow.log_metric("rows_seen", len(df))
            mlflow.log_artifact(model_path)
            mlflow.log_artifact(file_path)

        print(f"Dummy model saved → {model_path}")
        return model_path

    # -------------------------------
    # CASE 2: Train Real Model
    # -------------------------------
    X = df[["hour", "temp_lag1", "temp_rolling3"]]
    y = df["temperature"]

    test_size = 0.2 if len(df) > 5 else 0.5
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    run_name = f"{os.path.basename(file_path)}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    with mlflow.start_run(run_name=run_name):
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict & metrics
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        r2 = r2_score(y_test, preds)

        # Log MLflow
        mlflow.log_param("model_type", "RandomForestRegressor")
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
        feat_imp_path = f"{MODEL_DIR}/feature_importance_{os.path.basename(file_path).replace('.csv','.csv')}"
        feat_imp.to_csv(feat_imp_path, index=False)
        mlflow.log_artifact(feat_imp_path)

        print(f"✔ Trained real model saved to {model_path}")

    return model_path


if __name__ == "__main__":
    import sys
    train(sys.argv[1])
