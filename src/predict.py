import joblib
import pandas as pd

MODEL_DIR = "include/models"

def predict(model_file, X):
    model = joblib.load(model_file)
    preds = model.predict(X)
    return preds

if __name__ == "__main__":
    import sys
    file_path = sys.argv[1]
    df = pd.read_csv(file_path)
    X = df[["hour", "temp_lag1", "temp_rolling3"]]
    model_file = f"{MODEL_DIR}/rf_model_{file_path.split('/')[-1].replace('.csv','.pkl')}"
    print(predict(model_file, X))

