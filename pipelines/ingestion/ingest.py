import os
from datetime import datetime, timezone
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
from open_meteo_client import CITIES, fetch_recent

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
DATASET = os.getenv("BQ_DATASET_RAW")
TABLE = "hourly"


def ingest():
    client = bigquery.Client(project=PROJECT_ID)
    table_id = f"{PROJECT_ID}.{DATASET}.{TABLE}"
    now = datetime.now(timezone.utc).isoformat()

    for city_key in CITIES:
        print(f"Ingesting {city_key}...")

        # Fetch last 2 days to catch any missed hours
        records = fetch_recent(city_key, past_days=2)

        # Add ingested_at timestamp
        for r in records:
            r["ingested_at"] = now

        # Insert in batches
        batch_size = 500
        for i in range(0, len(records), batch_size):
            batch = records[i:i + batch_size]
            errors = client.insert_rows_json(table_id, batch)
            if errors:
                print(f"Errors inserting {city_key} batch: {errors}")

        print(f"Done: {city_key} — {len(records)} rows inserted.")

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest()