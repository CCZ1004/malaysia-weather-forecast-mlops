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
    description="Hourly temperature forecasts for 5 Malaysian cities.",
    version="1.0.0",
)

MODELS_PATH = Path(os.getenv("MODELS_PATH", str(Path(__file__).resolve().parents[1] / "models")))
STATIC_PATH = Path(__file__).resolve().parent / "static"
CITIES = ["KL", "Kemaman", "Penang", "JB", "KK"]

app.mount("/static", StaticFiles(directory=str(STATIC_PATH)), name="static")

# Load all models at startup
prophet_models = {}
xgb_models = {}

@app.on_event("startup")
def load_models():
    for city in CITIES:
        prophet_path = MODELS_PATH / f"prophet_{city}.joblib"
        xgb_path     = MODELS_PATH / f"xgb_{city}.joblib"
        if prophet_path.exists() and xgb_path.exists():
            prophet_models[city] = joblib.load(prophet_path)
            xgb_models[city]     = joblib.load(xgb_path)
            print(f"Loaded models for {city}")
        else:
            print(f"WARNING: Models not found for {city}")


def fetch_recent_actuals(city: str, hours: int = 168) -> pd.DataFrame:
    """Fetch recent actual temperature data from BigQuery for lag computation."""
    try:
        client = bigquery.Client(project=PROJECT_ID)
        query = f"""
            SELECT timestamp, temperature_2m, precipitation
            FROM `{PROJECT_ID}.{RAW_DATASET}.hourly`
            WHERE city = '{city}'
            ORDER BY timestamp DESC
            LIMIT {hours}
        """
        df = client.query(query).to_dataframe()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Warning: Could not fetch recent actuals for {city}: {e}")
        return pd.DataFrame()


def compute_xgb_features(future_timestamps: pd.DatetimeIndex, recent_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute XGBoost features for future timestamps.
    Uses real lag values where available, NaN beyond the lag window.
    """
    lag_windows = [1, 3, 6, 24, 168]
    rows = []

    for i, ts in enumerate(future_timestamps):
        row = {}

        # Lag features — look back in recent actuals
        for lag in lag_windows:
            target_ts = ts - timedelta(hours=lag)
            if not recent_df.empty:
                match = recent_df[recent_df["timestamp"] <= target_ts]
                if not match.empty:
                    closest = match.iloc[-1]
                    if abs((closest["timestamp"] - target_ts).total_seconds()) <= 3600:
                        row[f"temp_lag_{lag}h"] = closest["temperature_2m"]
                    else:
                        row[f"temp_lag_{lag}h"] = np.nan
                else:
                    row[f"temp_lag_{lag}h"] = np.nan
            else:
                row[f"temp_lag_{lag}h"] = np.nan

        # Rolling features from recent actuals
        if not recent_df.empty:
            past = recent_df[recent_df["timestamp"] < ts]
            row["temp_rolling_mean_6h"]   = past.tail(6)["temperature_2m"].mean() if len(past) >= 6 else np.nan
            row["temp_rolling_std_6h"]    = past.tail(6)["temperature_2m"].std()  if len(past) >= 6 else np.nan
            row["temp_rolling_mean_24h"]  = past.tail(24)["temperature_2m"].mean() if len(past) >= 24 else np.nan
            row["precip_rolling_sum_24h"] = past.tail(24)["precipitation"].sum()   if len(past) >= 24 else np.nan
        else:
            row["temp_rolling_mean_6h"]   = np.nan
            row["temp_rolling_std_6h"]    = np.nan
            row["temp_rolling_mean_24h"]  = np.nan
            row["precip_rolling_sum_24h"] = np.nan

        # Cyclical features — always available
        row["hour_sin"]  = np.sin(2 * np.pi * ts.hour / 24)
        row["hour_cos"]  = np.cos(2 * np.pi * ts.hour / 24)
        row["month_sin"] = np.sin(2 * np.pi * ts.month / 12)
        row["month_cos"] = np.cos(2 * np.pi * ts.month / 12)

        rows.append(row)

    return pd.DataFrame(rows)


# --- Schemas ---
class PredictionPoint(BaseModel):
    timestamp: str
    temperature_c: float


class PredictResponse(BaseModel):
    city: str
    forecast_horizon_hours: int
    predictions: list[PredictionPoint]
    generated_at: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: list[str]


# --- Endpoints ---
@app.get("/")
def root():
    return FileResponse(str(STATIC_PATH / "index.html"))


@app.get("/health", response_model=HealthResponse)
def health():
    return {
        "status": "ok",
        "models_loaded": list(prophet_models.keys()),
    }


@app.get("/predict", response_model=PredictResponse)
def predict(city: str = "KL", hours: int = 24):
    if city not in CITIES:
        raise HTTPException(
            status_code=400,
            detail=f"City '{city}' not supported. Choose from {CITIES}"
        )
    if hours < 1 or hours > 168:
        raise HTTPException(
            status_code=400,
            detail="hours must be between 1 and 168"
        )
    if city not in prophet_models:
        raise HTTPException(
            status_code=503,
            detail=f"Model for {city} not loaded."
        )

    prophet = prophet_models[city]
    xgb     = xgb_models[city]

    # Generate future timestamps
    now = pd.Timestamp.now().floor("h")
    future_timestamps = pd.date_range(start=now, periods=hours, freq="h")

    # Prophet forecast
    future_df = pd.DataFrame({"ds": future_timestamps})
    forecast = prophet.predict(future_df)
    prophet_pred = forecast["yhat"].values

    # Fetch recent actuals from BigQuery for real lag features
    recent_df = fetch_recent_actuals(city, hours=168)

    # Compute XGBoost features with real lags where available
    xgb_features = compute_xgb_features(future_timestamps, recent_df)

    # XGBoost residual correction
    xgb_correction = xgb.predict(xgb_features)
    final_pred = prophet_pred + xgb_correction

    predictions = [
        PredictionPoint(
            timestamp=ts.isoformat(),
            temperature_c=round(float(temp), 2),
        )
        for ts, temp in zip(future_timestamps, final_pred)
    ]

    return PredictResponse(
        city=city,
        forecast_horizon_hours=hours,
        predictions=predictions,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )