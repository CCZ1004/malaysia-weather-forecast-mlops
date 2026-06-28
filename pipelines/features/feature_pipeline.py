import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from google.cloud import bigquery

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
RAW_DATASET = os.getenv("BQ_DATASET_RAW")
FEATURE_DATASET = os.getenv("BQ_DATASET_FEATURES")
TABLE = "hourly"


def fetch_raw(client: bigquery.Client, city: str) -> pd.DataFrame:
    query = f"""
        SELECT
            city,
            timestamp,
            temperature_2m,
            precipitation,
            humidity,
            windspeed_10m,
            cloud_cover
        FROM `{PROJECT_ID}.{RAW_DATASET}.{TABLE}`
        WHERE city = '{city}'
        ORDER BY timestamp ASC
    """
    return client.query(query).to_dataframe()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # --- Lag features ---
    for lag in [1, 3, 6, 24, 168]:
        df[f"temp_lag_{lag}h"] = df["temperature_2m"].shift(lag)

    # --- Rolling features ---
    df["temp_rolling_mean_6h"]   = df["temperature_2m"].rolling(6).mean()
    df["temp_rolling_std_6h"]    = df["temperature_2m"].rolling(6).std()
    df["temp_rolling_mean_24h"]  = df["temperature_2m"].rolling(24).mean()
    df["precip_rolling_sum_24h"] = df["precipitation"].rolling(24).sum()

    # --- Cyclical encoding ---
    df["hour_sin"]  = np.sin(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["timestamp"].dt.hour / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["timestamp"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["timestamp"].dt.month / 12)

    # --- Drop rows with NaN from lag/rolling ---
    df = df.dropna()

    return df


def create_feature_table(client: bigquery.Client):
    table_id = f"{PROJECT_ID}.{FEATURE_DATASET}.{TABLE}"

    schema = [
        bigquery.SchemaField("city",                    "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("timestamp",               "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("temperature_2m",          "FLOAT64"),
        bigquery.SchemaField("temp_lag_1h",             "FLOAT64"),
        bigquery.SchemaField("temp_lag_3h",             "FLOAT64"),
        bigquery.SchemaField("temp_lag_6h",             "FLOAT64"),
        bigquery.SchemaField("temp_lag_24h",            "FLOAT64"),
        bigquery.SchemaField("temp_lag_168h",           "FLOAT64"),
        bigquery.SchemaField("temp_rolling_mean_6h",    "FLOAT64"),
        bigquery.SchemaField("temp_rolling_std_6h",     "FLOAT64"),
        bigquery.SchemaField("temp_rolling_mean_24h",   "FLOAT64"),
        bigquery.SchemaField("precip_rolling_sum_24h",  "FLOAT64"),
        bigquery.SchemaField("hour_sin",                "FLOAT64"),
        bigquery.SchemaField("hour_cos",                "FLOAT64"),
        bigquery.SchemaField("month_sin",               "FLOAT64"),
        bigquery.SchemaField("month_cos",               "FLOAT64"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",
    )
    table.clustering_fields = ["city"]

    client.create_table(table, exists_ok=True)
    print(f"Table {table_id} ready.")


def insert_features(client: bigquery.Client, df: pd.DataFrame):
    table_id = f"{PROJECT_ID}.{FEATURE_DATASET}.{TABLE}"

    # Convert timestamps to string for BigQuery JSON insert
    df["timestamp"] = df["timestamp"].astype(str)

    records = df[[
        "city", "timestamp", "temperature_2m",
        "temp_lag_1h", "temp_lag_3h", "temp_lag_6h",
        "temp_lag_24h", "temp_lag_168h",
        "temp_rolling_mean_6h", "temp_rolling_std_6h",
        "temp_rolling_mean_24h", "precip_rolling_sum_24h",
        "hour_sin", "hour_cos", "month_sin", "month_cos",
    ]].to_dict(orient="records")

    batch_size = 500
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        errors = client.insert_rows_json(table_id, batch)
        if errors:
            print(f"Errors in batch {i//batch_size + 1}: {errors}")

    print(f"Inserted {total} feature rows for {df['city'].iloc[0]}.")


def run_pipeline():
    client = bigquery.Client(project=PROJECT_ID)
    create_feature_table(client)

    cities = ["KL", "Kemaman", "Penang", "JB", "KK"]

    for city in cities:
        print(f"\nProcessing {city}...")
        df = fetch_raw(client, city)
        print(f"Fetched {len(df)} raw rows.")
        df = engineer_features(df)
        print(f"Engineered {len(df)} feature rows.")
        insert_features(client, df)

    print("\nFeature pipeline complete.")


if __name__ == "__main__":
    run_pipeline()