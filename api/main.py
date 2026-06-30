import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
RAW_DATASET = os.getenv("BQ_DATASET_RAW", "raw_weather")

app = FastAPI(
    title="Malaysia Weather Forecast API",
    description="Multi-variable hourly weather forecasts for 5 Malaysian cities.",
    version="2.0.0",
)

MODELS_PATH = Path(os.getenv("MODELS_PATH", str(Path(__file__).resolve().parents[1] / "models")))
STATIC_PATH = Path(__file__).resolve().parent / "static"
CITIES = ["KL", "Kemaman", "Penang", "JB", "KK"]

# variable_name -> BigQuery raw column name
VARIABLES = {
    "temperature": "temperature_2m",
    "precipitation": "precipitation",
    "humidity": "humidity",
    "windspeed": "windspeed_10m",
}

app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# Loaded at startup: prophet_models[variable][city], xgb_models[variable][city]
prophet_models: dict[str, dict] = {v: {} for v in VARIABLES}
xgb_models: dict[str, dict] = {v: {} for v in VARIABLES}


@app.on_event("startup")
def load_models():
    for variable_name in VARIABLES:
        for city in CITIES:
            prophet_path = MODELS_PATH / f"prophet_{variable_name}_{city}.joblib"
            xgb_path     = MODELS_PATH / f"xgb_{variable_name}_{city}.joblib"
            if prophet_path.exists() and xgb_path.exists():
                prophet_models[variable_name][city] = joblib.load(prophet_path)
                xgb_models[variable_name][city] = joblib.load(xgb_path)
            else:
                print(f"WARNING: Missing models for {variable_name}/{city}")
    loaded_count = sum(len(c) for c in prophet_models.values())
    print(f"Loaded {loaded_count} prophet models and matching xgb models.")


def fetch_recent_actuals(city: str, hours: int = 168) -> pd.DataFrame:
    """Fetch recent actuals for all 4 variables from BigQuery."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT timestamp, temperature_2m, precipitation, humidity, windspeed_10m
            FROM `{PROJECT_ID}.{RAW_DATASET}.hourly`
            WHERE city = '{city}'
            ORDER BY timestamp DESC
            LIMIT {hours}
        """
        df = client.query(query).to_dataframe()
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Warning: Could not fetch recent actuals for {city}: {e}")
        return pd.DataFrame()


def compute_xgb_features(
    future_timestamps: pd.DatetimeIndex,
    recent_df: pd.DataFrame,
    target_col: str,
) -> pd.DataFrame:
    """
    Compute XGBoost features for one target variable across future timestamps.
    Uses real lag/rolling values from recent_df where available, NaN beyond the lag window.
    """
    lag_windows = [1, 3, 6, 24, 168]
    rows = []

    for ts in future_timestamps:
        row = {}

        for lag in lag_windows:
            target_ts = ts - timedelta(hours=lag)
            col_name = f"{target_col}_lag_{lag}h"
            if not recent_df.empty:
                match = recent_df[recent_df["timestamp"] <= target_ts]
                if not match.empty:
                    closest = match.iloc[-1]
                    if abs((closest["timestamp"] - target_ts).total_seconds()) <= 3600:
                        row[col_name] = closest[target_col]
                    else:
                        row[col_name] = np.nan
                else:
                    row[col_name] = np.nan
            else:
                row[col_name] = np.nan

        if not recent_df.empty:
            past = recent_df[recent_df["timestamp"] < ts]
            row[f"{target_col}_rolling_mean_6h"]  = past.tail(6)[target_col].mean()  if len(past) >= 6  else np.nan
            row[f"{target_col}_rolling_std_6h"]   = past.tail(6)[target_col].std()   if len(past) >= 6  else np.nan
            row[f"{target_col}_rolling_mean_24h"] = past.tail(24)[target_col].mean() if len(past) >= 24 else np.nan
            if target_col == "precipitation":
                row["precipitation_rolling_sum_24h"] = past.tail(24)[target_col].sum() if len(past) >= 24 else np.nan
        else:
            row[f"{target_col}_rolling_mean_6h"]  = np.nan
            row[f"{target_col}_rolling_std_6h"]   = np.nan
            row[f"{target_col}_rolling_mean_24h"] = np.nan
            if target_col == "precipitation":
                row["precipitation_rolling_sum_24h"] = np.nan

        row["hour_sin"]  = np.sin(2 * np.pi * ts.hour / 24)
        row["hour_cos"]  = np.cos(2 * np.pi * ts.hour / 24)
        row["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

        rows.append(row)

    return pd.DataFrame(rows)


def predict_variable(
    variable_name: str,
    target_col: str,
    city: str,
    future_timestamps: pd.DatetimeIndex,
    recent_df: pd.DataFrame,
) -> np.ndarray:
    prophet = prophet_models[variable_name][city]
    xgb     = xgb_models[variable_name][city]

    future_df = pd.DataFrame({"ds": future_timestamps.tz_localize(None)})
    forecast = prophet.predict(future_df)
    prophet_pred = forecast["yhat"].values

    xgb_features = compute_xgb_features(future_timestamps, recent_df, target_col)
    xgb_correction = xgb.predict(xgb_features)

    final_pred = prophet_pred + xgb_correction

    # Clip to physically valid ranges
    if variable_name == "precipitation":
        final_pred = np.clip(final_pred, 0, None)
    elif variable_name == "humidity":
        final_pred = np.clip(final_pred, 0, 100)
    elif variable_name == "windspeed":
        final_pred = np.clip(final_pred, 0, None)

    return final_pred


def get_condition(precipitation: float, humidity: float) -> tuple[str, str]:
    """Derive a simple weather condition label from precipitation and humidity."""
    if precipitation > 1.0:
        return "Rainy", "🌧"
    elif precipitation > 0.1:
        return "Light Rain", "🌦"
    elif humidity > 85:
        return "Cloudy", "☁️"
    elif humidity > 65:
        return "Partly Cloudy", "⛅"
    else:
        return "Sunny", "☀️"


# --- Schemas ---
class PredictionPoint(BaseModel):
    timestamp: str
    temperature_c: float
    precipitation_mm: float
    humidity_pct: float
    windspeed_kmh: float
    condition: str
    condition_icon: str


class PredictResponse(BaseModel):
    city: str
    forecast_horizon_hours: int
    predictions: list[PredictionPoint]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    cities_loaded: list[str]
    variables_loaded: list[str]


# --- Endpoints ---
@app.get("/")
def root():
    return FileResponse(str(STATIC_PATH / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health():
    loaded_cities = [c for c in CITIES if all(c in prophet_models[v] for v in VARIABLES)]
    return {
        "status": "ok",
        "cities_loaded": loaded_cities,
        "variables_loaded": list(VARIABLES.keys()),
    }


@app.get("/predict", response_model=PredictResponse)
def predict(city: str = "KL", hours: int = 24):
    if city not in CITIES:
        raise HTTPException(status_code=400, detail=f"City '{city}' not supported. Choose from {CITIES}")
    if hours < 1 or hours > 168:
        raise HTTPException(status_code=400, detail="hours must be between 1 and 168")
    if city not in prophet_models["temperature"]:
        raise HTTPException(status_code=503, detail=f"Models for {city} not loaded.")

    now = pd.Timestamp.now(tz="UTC").floor("h")
    future_timestamps = pd.date_range(start=now, periods=hours, freq="h", tz="UTC")

    recent_df = fetch_recent_actuals(city, hours=168)

    results = {}
    for variable_name, target_col in VARIABLES.items():
        results[variable_name] = predict_variable(
            variable_name, target_col, city, future_timestamps, recent_df
        )

    predictions = []
    for i, ts in enumerate(future_timestamps):
        precip = float(results["precipitation"][i])
        humid  = float(results["humidity"][i])
        condition, icon = get_condition(precip, humid)

        predictions.append(PredictionPoint(
            timestamp=ts.isoformat(),
            temperature_c=round(float(results["temperature"][i]), 2),
            precipitation_mm=round(precip, 2),
            humidity_pct=round(humid, 1),
            windspeed_kmh=round(float(results["windspeed"][i]), 2),
            condition=condition,
            condition_icon=icon,
        ))

    return PredictResponse(
        city=city,
        forecast_horizon_hours=hours,
        predictions=predictions,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )