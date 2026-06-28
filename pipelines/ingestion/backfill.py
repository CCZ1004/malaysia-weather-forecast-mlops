import os
from datetime import date
from dotenv import load_dotenv
from google.cloud import bigquery
from open_meteo_client import CITIES, fetch_historical

from pathlib import Path
load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET_RAW")
TABLE = "hourly"

# Backfill range — 4 years of historical data
START_DATE = date(2020, 1, 1)
END_DATE = date(2024, 12, 31)


def create_table_if_not_exists(client: bigquery.Client):
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    schema = [
        bigquery.SchemaField("city",           "STRING",    mode="REQUIRED"),
        bigquery.SchemaField("timestamp",      "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("temperature_2m", "FLOAT64"),
        bigquery.SchemaField("precipitation",  "FLOAT64"),
        bigquery.SchemaField("humidity",       "FLOAT64"),
        bigquery.SchemaField("windspeed_10m",  "FLOAT64"),
        bigquery.SchemaField("cloud_cover",    "INT64"),
        bigquery.SchemaField("ingested_at",    "TIMESTAMP"),
    ]

    table = bigquery.Table(table_id, schema=schema)

    # Partition by date for cheaper queries
    table.time_partitioning = bigquery.TimePartitioning(
        type_=bigquery.TimePartitioningType.DAY,
        field="timestamp",
    )
    table.clustering_fields = ["city"]

    client.create_table(table, exists_ok=True)
    print(f"Table {table_id} ready.")


def insert_rows(client: bigquery.Client, records: list[dict]):
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    for r in records:
        r["ingested_at"] = now

    # Insert in batches of 500 rows
    batch_size = 500
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        errors = client.insert_rows_json(table_id, batch)
        if errors:
            print(f"Errors in batch {i//batch_size + 1}: {errors}")
        else:
            print(f"Batch {i//batch_size + 1}/{(total + batch_size - 1)//batch_size} inserted ({len(batch)} rows)")

def backfill():
    client = bigquery.Client(project=PROJECT_ID)
    create_table_if_not_exists(client)

    # Skip cities already backfilled
    already_done = ["KL"]  # Add "KL" here since it was already inserted
    
    for city_key in CITIES:
        if city_key in already_done:
            print(f"Skipping {city_key} — already backfilled.")
            continue
        print(f"\nBackfilling {city_key} from {START_DATE} to {END_DATE}...")
        records = fetch_historical(city_key, START_DATE, END_DATE)
        print(f"Fetched {len(records)} records.")
        insert_rows(client, records)
        print(f"Done: {city_key}")

    print("\nBackfill complete.")


if __name__ == "__main__":
    backfill()