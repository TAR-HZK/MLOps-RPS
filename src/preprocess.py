import pandas as pd
import os
import glob
from datetime import datetime
import sweetviz as sv  # <--- Using Sweetviz (Matches your Dockerfile)

# Docker-safe paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, "include", "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "include", "data", "processed")

def preprocess():
    print("🔄 Merging raw data to build history...")
    
    # 1. Load ALL raw files (We need history to predict the future)
    all_files = glob.glob(os.path.join(RAW_DIR, "*.csv"))
    if not all_files:
        raise Exception("❌ No raw data found! Run 'python generate_fake_data.py' first.")
    
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)
    
    # 2. Sort by Time (Crucial for time-series)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    # 3. Feature Engineering (The Input X)
    df["hour"] = df["timestamp"].dt.hour
    df["temp_lag1"] = df["temperature"].shift(1)
    df["temp_rolling3"] = df["temperature"].rolling(3).mean()
    
    # 4. Create the FUTURE TARGET (The Output y)
    # We want to predict temperature 6 hours from now.
    # So, for the row at 12:00, the target is the temp at 18:00.
    df["target_6h"] = df["temperature"].shift(-6)
    
    # 5. Drop rows we can't use (Missing targets or lags)
    df_clean = df.dropna().reset_index(drop=True)
    
    if len(df_clean) == 0:
        print("⚠ Not enough data to generate training pairs yet.")
        # Fallback to prevent crash during testing
        df_clean = df.fillna(method='bfill').fillna(method='ffill')

    # 6. Save Processed CSV
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filename = os.path.join(PROCESSED_DIR, "processed_data.csv")
    df_clean.to_csv(filename, index=False)
    print(f"✅ Saved processed dataset with {len(df_clean)} rows.")

    # 7. Generate Quality Report (Sweetviz)
    print("📊 Generating Sweetviz Quality Report...")
    report_filename = filename.replace(".csv", "_report.html")
    
    # Create and save report
    report = sv.analyze(df_clean)
    report.show_html(filepath=report_filename, open_browser=False, layout='vertical')
    
    print(f"📄 Saved Quality Report to {report_filename}")
    return filename

if __name__ == "__main__":
    preprocess()