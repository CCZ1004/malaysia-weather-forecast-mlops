import os
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
from open_meteo_client import CITIES, fetch_historical

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET_RAW")
TABLE = "hourly"

# Gap-fill range — 2025 only, since 2020-2024 already loaded
START_DATE = date(2025, 1, 1)
END_DATE = date(2025, 12, 31)


def insert_rows(client: bigquery.Client, records: list[dict]):
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    for r in records:
        r["ingested_at"] = now

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

    for city_key in CITIES:
        print(f"\nGap-filling {city_key} from {START_DATE} to {END_DATE}...")
        records = fetch_historical(city_key, START_DATE, END_DATE)
        print(f"Fetched {len(records)} records.")
        insert_rows(client, records)
        print(f"Done: {city_key}")

    print("\n2025 gap-fill complete.")


if __name__ == "__main__":
    backfill()