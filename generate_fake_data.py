import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import random

# Path setup
RAW_DATA_DIR = "include/data/raw"
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def generate_data():
    print("Generating 48 hours of fake historical data...")
    
    # Start from 48 hours ago
    start_time = datetime.now() - timedelta(hours=48)
    city = "London"
    
    generated_files = []

    for i in range(50): # Generate 50 points to be safe
        current_time = start_time + timedelta(hours=i)
        
        # 1. Simulate Temperature (Sine wave for day/night cycle + noise)
        # Peak at 2PM (14:00), lowest at 2AM. Mean 15°C, Amplitude 5°C.
        hour = current_time.hour
        base_temp = 15 + 5 * np.sin((hour - 8) * np.pi / 12) 
        temp = round(base_temp + random.uniform(-1.5, 1.5), 2)
        
        # 2. Simulate other metrics
        humidity = random.randint(40, 90)
        pressure = random.randint(990, 1020)
        wind_speed = round(random.uniform(0, 15), 2)
        weather_main = random.choice(["Clear", "Clouds", "Rain", "Drizzle"])

        # 3. Create DataFrame matching your format exactly
        record = {
            "timestamp": current_time,
            "city": city,
            "temperature": temp,
            "humidity": humidity,
            "pressure": pressure,
            "wind_speed": wind_speed,
            "weather_main": weather_main
        }
        df = pd.DataFrame([record])

        # 4. Save individual CSVs (simulating separate API calls)
        filename = f"{RAW_DATA_DIR}/data_{current_time.strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(filename, index=False)
        generated_files.append(filename)

    print(f"✅ Generated {len(generated_files)} CSV files in {RAW_DATA_DIR}")

if __name__ == "__main__":
    generate_data()
