import os
import pandas as pd
import numpy as np
import mlflow
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

# Map target variable -> (column name, lag/rolling feature prefix)
TARGETS = {
    "temperature": "temperature_2m",
    "precipitation": "precipitation",
    "humidity": "humidity",
    "windspeed": "windspeed_10m",
}

# Time-based split — extended through 2025
TRAIN_END = "2024-12-31"
VAL_END   = "2025-06-30"
# Test: 2025-07-01 onward


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


def train_prophet(train: pd.DataFrame, target_col: str) -> Prophet:
    prophet_df = train[["timestamp", target_col]].rename(
        columns={"timestamp": "ds", target_col: "y"}
    )
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


def get_xgb_feature_columns(target_var: str) -> list[str]:
    """Build the feature column list for a given target variable."""
    cols = []
    for lag in [1, 3, 6, 24, 168]:
        cols.append(f"{target_var}_lag_{lag}h")
    cols.append(f"{target_var}_rolling_mean_6h")
    cols.append(f"{target_var}_rolling_std_6h")
    cols.append(f"{target_var}_rolling_mean_24h")
    if target_var == "precipitation":
        cols.append("precipitation_rolling_sum_24h")
    cols += ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    return cols


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    # Avoid divide-by-zero for variables that can be 0 (e.g. precipitation)
    nonzero_mask = actual != 0
    if nonzero_mask.sum() > 0:
        mape = np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100
    else:
        mape = float("nan")
    return {"mae": round(mae, 4), "rmse": round(rmse, 4), "mape": round(mape, 4)}


def train_city_variable(city: str, variable_name: str, target_col: str, df: pd.DataFrame):
    print(f"\n--- {city} / {variable_name} ---")

    train, val, test = split_data(df)
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")

    with mlflow.start_run(run_name=f"{city}_{variable_name}_ensemble"):
        # Step 1: Prophet baseline
        prophet_model = train_prophet(train, target_col)
        train_prophet_pred = get_prophet_predictions(prophet_model, train)
        val_prophet_pred   = get_prophet_predictions(prophet_model, val)

        # Step 2: Residuals
        train_residuals = train[target_col].values - train_prophet_pred

        # Step 3: XGBoost on residuals
        feature_cols = get_xgb_feature_columns(target_col)
        X_train = train[feature_cols]
        X_val   = val[feature_cols]

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
            eval_set=[(X_val, val[target_col].values - val_prophet_pred)],
            verbose=False,
        )

        # Step 4: Ensemble
        val_xgb_residual = xgb_model.predict(X_val)
        val_ensemble_pred = val_prophet_pred + val_xgb_residual

        # Clip predictions to physically valid ranges
        if variable_name == "precipitation":
            val_ensemble_pred = np.clip(val_ensemble_pred, 0, None)
        elif variable_name == "humidity":
            val_ensemble_pred = np.clip(val_ensemble_pred, 0, 100)
        elif variable_name == "windspeed":
            val_ensemble_pred = np.clip(val_ensemble_pred, 0, None)

        # Step 5: Evaluate
        prophet_metrics  = evaluate(val[target_col].values, val_prophet_pred)
        ensemble_metrics = evaluate(val[target_col].values, val_ensemble_pred)

        print(f"Prophet  — MAE: {prophet_metrics['mae']} | RMSE: {prophet_metrics['rmse']}")
        print(f"Ensemble — MAE: {ensemble_metrics['mae']} | RMSE: {ensemble_metrics['rmse']}")

        # Step 6: Log to MLflow
        mlflow.log_params({
            "city": city,
            "variable": variable_name,
            "train_end": TRAIN_END,
            "val_end": VAL_END,
        })
        mlflow.log_metrics({
            "prophet_mae":  prophet_metrics["mae"],
            "prophet_rmse": prophet_metrics["rmse"],
            "ensemble_mae":  ensemble_metrics["mae"],
            "ensemble_rmse": ensemble_metrics["rmse"],
        })

        # Step 7: Save models
        prophet_path = MODELS_PATH / f"prophet_{variable_name}_{city}.joblib"
        xgb_path     = MODELS_PATH / f"xgb_{variable_name}_{city}.joblib"
        joblib.dump(prophet_model, prophet_path)
        joblib.dump(xgb_model, xgb_path)
        mlflow.log_artifact(str(prophet_path))
        mlflow.log_artifact(str(xgb_path))

    return ensemble_metrics


def train_all():
    mlflow.set_experiment("malaysia-weather-forecast-multivariable")

    summary = {}
    for city in CITIES:
        df = load_data(city)
        summary[city] = {}
        for variable_name, target_col in TARGETS.items():
            metrics = train_city_variable(city, variable_name, target_col, df)
            summary[city][variable_name] = metrics

    print(f"\n{'='*60}")
    print("SUMMARY — Validation MAE per city / variable")
    print(f"{'='*60}")
    for city, vars_metrics in summary.items():
        for variable_name, m in vars_metrics.items():
            print(f"{city:10s} {variable_name:14s} MAE: {m['mae']:.4f}")


if __name__ == "__main__":
    train_all()