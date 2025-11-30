import pandas as pd
import os
from datetime import datetime

RAW_DIR = "include/data/raw"
PROCESSED_DIR = "include/data/processed"

def preprocess(file_path):
    df = pd.read_csv(file_path)
    
    # Feature engineering
    df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
    df["temp_lag1"] = df["temperature"].shift(1)
    df["temp_rolling3"] = df["temperature"].rolling(3).mean()
    
    # Fill NaNs after feature creation
    df.fillna(method="bfill", inplace=True)
    
    # Save processed
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filename = f"{PROCESSED_DIR}/processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved processed data to {filename}")
    return filename

if __name__ == "__main__":
    import sys
    preprocess(sys.argv[1])

