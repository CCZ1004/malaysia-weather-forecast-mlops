from google.cloud import bigquery
from dotenv import load_dotenv
import os

load_dotenv()
client = bigquery.Client(project=os.getenv('GCP_PROJECT_ID'))
table_id = f"{os.getenv('GCP_PROJECT_ID')}.feature_store.hourly"

try:
    client.delete_table(table_id)
    print('Deleted existing table')
except Exception as e:
    print(f'No table to delete or error: {e}')