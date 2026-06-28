import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

app = FastAPI(
    title="Malaysia Weather Forecast API",
    description="Hourly temperature forecasts for 5 Malaysian cities.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory="api/static"), name="static")

@app.get("/")
def root():
    return FileResponse("api/static/index.html")

MODELS_PATH = Path(os.getenv("MODELS_PATH", "/app/models"))
CITIES = ["KL", "Kemaman", "Penang", "JB", "KK"]

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

    # XGBoost residual correction
    # For future hours we use cyclical features only (no lag features available)
    xgb_features = pd.DataFrame({
        "temp_lag_1h":             [np.nan] * hours,
        "temp_lag_3h":             [np.nan] * hours,
        "temp_lag_6h":             [np.nan] * hours,
        "temp_lag_24h":            [np.nan] * hours,
        "temp_lag_168h":           [np.nan] * hours,
        "temp_rolling_mean_6h":    [np.nan] * hours,
        "temp_rolling_std_6h":     [np.nan] * hours,
        "temp_rolling_mean_24h":   [np.nan] * hours,
        "precip_rolling_sum_24h":  [np.nan] * hours,
        "hour_sin":  np.sin(2 * np.pi * future_timestamps.hour / 24),
        "hour_cos":  np.cos(2 * np.pi * future_timestamps.hour / 24),
        "month_sin": np.sin(2 * np.pi * future_timestamps.month / 12),
        "month_cos": np.cos(2 * np.pi * future_timestamps.month / 12),
    })

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