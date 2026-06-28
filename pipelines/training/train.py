import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from pathlib import Path
from dotenv import load_dotenv
from prophet import Prophet
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "features.csv"
MODELS_PATH = Path(__file__).resolve().parents[2] / "models"
MODELS_PATH.mkdir(exist_ok=True)

CITIES = ["KL", "Kemaman", "Penang", "JB", "KK"]

# Time-based split — no random shuffle to avoid leakage
TRAIN_END = "2023-12-31"
VAL_END   = "2024-06-30"
# Test: 2024-07-01 to 2024-12-31


def load_data(city: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df[df["city"] == city].sort_values("timestamp").reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame):
    train = df[df["timestamp"] <= TRAIN_END]
    val   = df[(df["timestamp"] > TRAIN_END) & (df["timestamp"] <= VAL_END)]
    test  = df[df["timestamp"] > VAL_END]
    return train, val, test


def train_prophet(train: pd.DataFrame) -> Prophet:
    prophet_df = train[["timestamp", "temperature_2m"]].rename(
        columns={"timestamp": "ds", "temperature_2m": "y"}
    )
    # Remove timezone info — Prophet doesn't support tz-aware timestamps
    prophet_df["ds"] = prophet_df["ds"].dt.tz_localize(None)

    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_mode="multiplicative",
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True,
    )
    model.fit(prophet_df)
    return model

def get_prophet_predictions(model: Prophet, df: pd.DataFrame) -> np.ndarray:
    future = df[["timestamp"]].rename(columns={"timestamp": "ds"})
    future["ds"] = future["ds"].dt.tz_localize(None)
    forecast = model.predict(future)
    return forecast["yhat"].values


def get_xgb_features(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [
        "temp_lag_1h", "temp_lag_3h", "temp_lag_6h",
        "temp_lag_24h", "temp_lag_168h",
        "temp_rolling_mean_6h", "temp_rolling_std_6h",
        "temp_rolling_mean_24h", "precip_rolling_sum_24h",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]
    return df[feature_cols]


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


def train_city(city: str):
    print(f"\n{'='*50}")
    print(f"Training models for {city}")
    print(f"{'='*50}")

    df = load_data(city)
    train, val, test = split_data(df)

    print(f"Train: {len(train)} rows | Val: {len(val)} rows | Test: {len(test)} rows")

    with mlflow.start_run(run_name=f"{city}_ensemble"):
        # --- Step 1: Train Prophet baseline ---
        print("Training Prophet...")
        prophet_model = train_prophet(train)

        # Get Prophet predictions on train + val for residual calculation
        train_prophet_pred = get_prophet_predictions(prophet_model, train)
        val_prophet_pred   = get_prophet_predictions(prophet_model, val)

        # --- Step 2: Compute residuals ---
        train_residuals = train["temperature_2m"].values - train_prophet_pred

        # --- Step 3: Train XGBoost on residuals ---
        print("Training XGBoost on residuals...")
        X_train = get_xgb_features(train)
        X_val   = get_xgb_features(val)

        xgb_model = XGBRegressor(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        xgb_model.fit(
            X_train, train_residuals,
            eval_set=[(X_val, val["temperature_2m"].values - val_prophet_pred)],
            verbose=False,
        )

        # --- Step 4: Ensemble predictions ---
        val_xgb_residual = xgb_model.predict(X_val)
        val_ensemble_pred = val_prophet_pred + val_xgb_residual

        # --- Step 5: Evaluate ---
        prophet_metrics  = evaluate(val["temperature_2m"].values, val_prophet_pred)
        ensemble_metrics = evaluate(val["temperature_2m"].values, val_ensemble_pred)

        print(f"Prophet  — MAE: {prophet_metrics['mae']} | RMSE: {prophet_metrics['rmse']} | MAPE: {prophet_metrics['mape']}%")
        print(f"Ensemble — MAE: {ensemble_metrics['mae']} | RMSE: {ensemble_metrics['rmse']} | MAPE: {ensemble_metrics['mape']}%")

        # --- Step 6: Log to MLflow ---
        mlflow.log_params({
            "city": city,
            "train_end": TRAIN_END,
            "val_end": VAL_END,
            "prophet_changepoint_prior": 0.05,
            "prophet_seasonality_mode": "multiplicative",
            "xgb_n_estimators": 200,
            "xgb_max_depth": 4,
            "xgb_learning_rate": 0.05,
        })

        mlflow.log_metrics({
            f"prophet_mae":   prophet_metrics["mae"],
            f"prophet_rmse":  prophet_metrics["rmse"],
            f"ensemble_mae":  ensemble_metrics["mae"],
            f"ensemble_rmse": ensemble_metrics["rmse"],
            f"ensemble_mape": ensemble_metrics["mape"],
        })

        # --- Step 7: Save models ---
        prophet_path = MODELS_PATH / f"prophet_{city}.joblib"
        xgb_path     = MODELS_PATH / f"xgb_{city}.joblib"

        joblib.dump(prophet_model, prophet_path)
        joblib.dump(xgb_model, xgb_path)

        mlflow.log_artifact(str(prophet_path))
        mlflow.log_artifact(str(xgb_path))

        print(f"Models saved to {MODELS_PATH}")

    return ensemble_metrics


def train_all():
    mlflow.set_experiment("malaysia-weather-forecast")

    all_metrics = {}
    for city in CITIES:
        metrics = train_city(city)
        all_metrics[city] = metrics

    print(f"\n{'='*50}")
    print("SUMMARY — Validation MAE per city")
    print(f"{'='*50}")
    for city, m in all_metrics.items():
        print(f"{city:12s} MAE: {m['mae']} | RMSE: {m['rmse']} | MAPE: {m['mape']}%")

    avg_mae = np.mean([m["mae"] for m in all_metrics.values()])
    print(f"\nAverage MAE across all cities: {avg_mae:.4f}")


if __name__ == "__main__":
    train_all()