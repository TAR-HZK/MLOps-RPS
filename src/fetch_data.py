import requests
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
load_dotenv()


RAW_DATA_DIR = "include/data/raw"
API_KEY = os.getenv("OPENWEATHER_API_KEY")  # better to store in .env
CITY = "London"
BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

def fetch_weather_data():
    params = {"q": CITY, "appid": API_KEY, "units": "metric"}
    response = requests.get(BASE_URL, params=params, timeout=10)
    if response.status_code != 200:
        raise Exception(f"API request failed with status {response.status_code}: {response.text}")
    data = response.json()

    
    # Flatten data into a dict
    record = {
        "timestamp": datetime.now(),
        "city": CITY,
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind_speed": data["wind"]["speed"],
        "weather_main": data["weather"][0]["main"],
    }

    df = pd.DataFrame([record])
    
    # Save CSV
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    filename = f"{RAW_DATA_DIR}/data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved raw data to {filename}")
    return filename

if __name__ == "__main__":
    fetch_weather_data()

