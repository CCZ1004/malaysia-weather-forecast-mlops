import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import bigquery
import pandas as pd

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROJECT_ID = os.getenv("GCP_PROJECT_ID")
FEATURE_DATASET = os.getenv("BQ_DATASET_FEATURES")

def export():
    client = bigquery.Client(project=PROJECT_ID)

    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{FEATURE_DATASET}.hourly`
        ORDER BY city, timestamp ASC
    """

    print("Fetching features from BigQuery...")
    df = client.query(query).to_dataframe()
    print(f"Fetched {len(df)} rows across {df['city'].nunique()} cities.")

    output_path = Path(__file__).resolve().parents[2] / "data" / "features.csv"
    output_path.parent.mkdir(exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    export()