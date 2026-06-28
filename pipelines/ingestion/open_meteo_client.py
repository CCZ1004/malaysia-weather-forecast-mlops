import httpx
from datetime import datetime, date
from typing import Optional


# City coordinates
CITIES = {
    "KL": {"name": "Kuala Lumpur", "latitude": 3.1390, "longitude": 101.6869},
    "Kemaman": {"name": "Kemaman", "latitude": 4.2333, "longitude": 103.4167},
    "Penang": {"name": "Penang", "latitude": 5.4141, "longitude": 100.3288},
    "JB": {"name": "Johor Bahru", "latitude": 1.4927, "longitude": 103.7414},
    "KK": {"name": "Kota Kinabalu", "latitude": 5.9804, "longitude": 116.0735},
}

# Variables to fetch from Open-Meteo
HOURLY_VARIABLES = [
    "temperature_2m",
    "precipitation",
    "relative_humidity_2m",
    "windspeed_10m",
    "cloud_cover",
]

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"

import time

def fetch_historical(
    city_key: str,
    start_date: date,
    end_date: date,
    retries: int = 3,
) -> list[dict]:
    city = CITIES[city_key]

    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kuala_Lumpur",
    }

    for attempt in range(retries):
        try:
            response = httpx.get(HISTORICAL_URL, params=params, timeout=30)
            response.raise_for_status()
            return _parse_hourly(response.json(), city_key)
        except httpx.HTTPStatusError as e:
            if attempt < retries - 1:
                wait = 10 * (attempt + 1)
                print(f"Error fetching {city_key}: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise


def fetch_recent(city_key: str, past_days: int = 7) -> list[dict]:
    """
    Fetch recent + upcoming hourly weather data for a city.
    Uses Open-Meteo Forecast API (no API key required).

    Args:
        city_key: One of KL, Kemaman, Penang, JB, KK
        past_days: How many past days to include (max 92)

    Returns:
        List of dicts, one per hour
    """
    city = CITIES[city_key]

    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "Asia/Kuala_Lumpur",
        "past_days": past_days,
        "forecast_days": 1,
    }

    response = httpx.get(FORECAST_URL, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    return _parse_hourly(data, city_key)


def _parse_hourly(data: dict, city_key: str) -> list[dict]:
    """
    Parse Open-Meteo hourly response into a flat list of dicts.
    Each dict represents one hour of weather data for one city.
    """
    hourly = data["hourly"]
    timestamps = hourly["time"]
    records = []

    for i, ts in enumerate(timestamps):
        records.append({
            "city": city_key,
            "timestamp": datetime.fromisoformat(ts).isoformat(),
            "temperature_2m": hourly["temperature_2m"][i],
            "precipitation": hourly["precipitation"][i],
            "humidity": hourly["relative_humidity_2m"][i],
            "windspeed_10m": hourly["windspeed_10m"][i],
            "cloud_cover": hourly["cloud_cover"][i],
        })

    return records


if __name__ == "__main__":
    # Quick test — fetch last 2 days for KL
    records = fetch_recent("KL", past_days=2)
    print(f"Fetched {len(records)} records for KL")
    print(records[0])
    print(records[-1])