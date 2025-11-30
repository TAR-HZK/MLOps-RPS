import pandas as pd
import sys

def check_quality(file_path):
    df = pd.read_csv(file_path)
    
    key_columns = ["temperature", "humidity", "pressure", "wind_speed"]
    null_percentage = df[key_columns].isnull().mean()
    
    for col, pct in null_percentage.items():
        if pct > 0.01:
            print(f"Data quality check failed: {col} has {pct*100:.2f}% nulls")
            sys.exit(1)
    
    print("Data quality check passed")
    return True

if __name__ == "__main__":
    import sys
    check_quality(sys.argv[1])

