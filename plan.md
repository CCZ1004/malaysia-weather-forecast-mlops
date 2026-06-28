# Malaysia Weather Forecast MLOps Pipeline — Detailed Project Plan

**Project Name:** `malaysia-weather-forecast-mlops`  
**Author:** Chiu Chang Ze  
**Goal:** End-to-end MLOps pipeline for multi-city Malaysian weather forecasting — data engineering, time series ML, experiment tracking, drift monitoring, and full observability — deployed on GCP via GitHub Actions CI/CD.

---

## 1. Project Overview

### Problem Statement

Build a production-grade MLOps system that ingests historical and real-time weather data for major Malaysian cities, trains and serves a time series forecasting model, and continuously monitors for data drift and prediction degradation — deployed serverlessly on Google Cloud Run with automated CI/CD via GitHub Actions.

### Why This Project

- Demonstrates time series ML (distinct from tabular classification in churn/fraud projects)
- Natural concept drift (monsoon vs. dry season) makes drift monitoring realistic and non-trivial
- Multi-location forecasting shows scalability thinking
- Open-Meteo is free, no API key, years of historical data — zero setup friction
- GCP free tier covers the entire stack: Cloud Run, GCS, BigQuery, Cloud Scheduler

### Target Cities

| City | Latitude | Longitude | Region |
|---|---|---|---|
| Kuala Lumpur | 3.1390 | 101.6869 | West |
| Kemaman | 4.2333 | 103.4167 | East Coast |
| Penang | 5.4141 | 100.3288 | North |
| Johor Bahru | 1.4927 | 103.7414 | South |
| Kota Kinabalu | 5.9804 | 116.0735 | Sabah |

### Forecast Target

**Primary:** Hourly temperature forecast (next 24 hours)  
**Secondary:** Daily precipitation probability (next 7 days)

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│                                                                  │
│  Open-Meteo Historical API  ──►  backfill.py (one-shot script)   │
│  Open-Meteo Forecast API    ──►  ingestion service               │
│                                        │                         │
│                             Cloud Scheduler (hourly trigger)     │
│                                        │                         │
│                                   BigQuery                       │
│                              (dataset: raw_weather)              │
└──────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────┐
│                       FEATURE LAYER                              │
│                                                                  │
│  Feature Pipeline (Cloud Scheduler: daily trigger)               │
│  - Lag features (1h, 3h, 6h, 12h, 24h, 48h, 168h)               │
│  - Rolling stats (mean, std, min, max over 6h/24h/72h)           │
│  - Cyclical encoding (hour_sin/cos, month_sin/cos)               │
│  - Malaysian public holiday flags                                │
│                        │                                         │
│                   BigQuery (dataset: feature_store)              │
└──────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────┐
│                        MODEL LAYER                               │
│                                                                  │
│  Training Pipeline (Cloud Scheduler: weekly trigger)             │
│  - Prophet  ──► baseline (seasonality + trend)                   │
│  - XGBoost  ──► residual model (learns what Prophet misses)      │
│  - Final prediction = Prophet forecast + XGBoost correction      │
│                                                                  │
│  MLflow Tracking (self-hosted on Cloud Run)                      │
│  - Artifacts stored in GCS bucket                                │
│  - Model registry: Staging → Production promotion                │
└──────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────┐
│                       SERVING LAYER                              │
│                                                                  │
│  FastAPI  ──►  Docker image  ──►  Google Cloud Run               │
│  - GET  /health                                                   │
│  - GET  /predict?city=KL&hours=24                                │
│  - POST /retrain                                                  │
│  - GET  /model/info                                              │
│  - GET  /metrics  (Prometheus scrape endpoint)                   │
│                                                                  │
│  Streamlit UI  ──►  separate Cloud Run service                   │
│  - Calls FastAPI for real-time forecasts                         │
│  - Displays forecast charts + drift status                       │
└──────────────────────────────────────────────────────────────────┘
                                    │
┌──────────────────────────────────────────────────────────────────┐
│                     MONITORING LAYER                             │
│                                                                  │
│  Evidently AI (Cloud Scheduler: daily trigger)                   │
│  - Data drift report (feature distribution shift)                │
│  - Prediction drift report (MAE degradation)                     │
│  - Reports saved as HTML to GCS bucket                           │
│                                                                  │
│  Prometheus + Grafana Cloud                                      │
│  - Request latency, error rate, throughput                       │
│  - Rolling MAE per city                                          │
│  - Drift score gauge with alert threshold                        │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Deployment Pipeline

The deployment pattern mirrors the Telco Churn project: every `git push` to `main` triggers a full CI/CD run via GitHub Actions.

```
Developer pushes code
        │
        ▼
GitHub Actions (CI/CD)
  ├── 1. Run tests (pytest)
  ├── 2. Build Docker image
  ├── 3. Push to Google Artifact Registry
  └── 4. Deploy to Google Cloud Run
        │
        ▼
Cloud Run (FastAPI)          Cloud Run (Streamlit UI)
        │                              │
        └──────────── calls ───────────┘
                          │
                     BigQuery + GCS
                     (data + artifacts)
```

### GitHub Actions Workflow (`.github/workflows/deploy.yml`)

```yaml
name: Deploy Weather Forecast API

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v1
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v1

      - name: Run tests
        run: pytest tests/ -v

      - name: Build and push Docker image
        run: |
          gcloud builds submit \
            --tag asia-southeast1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/weather-repo/weather-api:${{ github.sha }}

      - name: Deploy to Cloud Run
        run: |
          gcloud run deploy weather-forecast-api \
            --image asia-southeast1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/weather-repo/weather-api:${{ github.sha }} \
            --region asia-southeast1 \
            --platform managed \
            --allow-unauthenticated \
            --set-env-vars GCP_PROJECT=${{ secrets.GCP_PROJECT_ID }}
```

### GCP Region

`asia-southeast1` (Singapore) — lowest latency for Malaysia.

---

## 4. GCP Services & Free Tier Usage

| GCP Service | Role | Free Tier Limit |
|---|---|---|
| **Cloud Run** | FastAPI + Streamlit serving | 2M requests/month, 360K vCPU-seconds |
| **Cloud Storage (GCS)** | MLflow artifacts, Evidently HTML reports | 5 GB storage, 1 GB egress/month |
| **BigQuery** | Raw weather, feature store, predictions | 10 GB storage, 1 TB queries/month |
| **Cloud Scheduler** | Ingestion, feature pipeline, retraining triggers | 3 jobs free/month |
| **Artifact Registry** | Docker image storage | 0.5 GB free |
| **Cloud Build** | Build Docker images in CI/CD | 120 build-minutes/day |

Everything fits within free tier for a portfolio project with light traffic.

---

## 5. Repository Structure

```
malaysia-weather-forecast-mlops/
│
├── .github/
│   └── workflows/
│       ├── deploy.yml          # FastAPI CI/CD
│       └── deploy_ui.yml       # Streamlit CI/CD
│
├── api/                        # FastAPI inference service
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py
│   ├── predictor.py            # Loads model from GCS via MLflow
│   └── schemas.py              # Pydantic request/response models
│
├── ui/                         # Streamlit frontend
│   ├── Dockerfile
│   ├── requirements.txt
│   └── app.py
│
├── pipelines/                  # Data + ML pipelines (triggered by Cloud Scheduler)
│   ├── ingestion/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── backfill.py         # One-shot historical data pull (run manually once)
│   │   ├── ingest.py           # Hourly ingestion job
│   │   └── open_meteo_client.py
│   │
│   ├── features/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── feature_pipeline.py
│   │
│   ├── training/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   ├── train.py
│   │   ├── prophet_model.py
│   │   ├── xgboost_model.py
│   │   └── evaluate.py
│   │
│   └── monitoring/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── drift_monitor.py
│
├── infra/
│   ├── bigquery/
│   │   └── schema.py           # BigQuery table schema definitions
│   ├── gcs/
│   │   └── setup_buckets.py    # One-shot bucket creation script
│   └── scheduler/
│       └── create_jobs.sh      # Cloud Scheduler job creation commands
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_experiments.ipynb
│   └── 04_drift_analysis.ipynb
│
├── tests/
│   ├── test_ingestion.py
│   ├── test_features.py
│   ├── test_predictor.py
│   └── test_api.py
│
├── docker-compose.yml          # Local dev environment
├── .env.example
└── README.md
```

---

## 6. BigQuery Schema

### Table: `raw_weather.hourly`

```python
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
# Partitioned by DATE(timestamp), clustered by city
```

### Table: `feature_store.hourly`

```python
schema = [
    bigquery.SchemaField("city",                    "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("timestamp",               "TIMESTAMP", mode="REQUIRED"),
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
    bigquery.SchemaField("temperature_2m",          "FLOAT64"),
]
# Partitioned by DATE(timestamp), clustered by city
```

### Table: `predictions.hourly`

```python
schema = [
    bigquery.SchemaField("city",           "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("forecast_time",  "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("predicted_temp", "FLOAT64",   mode="REQUIRED"),
    bigquery.SchemaField("actual_temp",    "FLOAT64"),
    bigquery.SchemaField("model_version",  "STRING"),
    bigquery.SchemaField("predicted_at",   "TIMESTAMP"),
]
```

---

## 7. GCS Bucket Structure

```
gs://weather-mlops-artifacts/
├── mlflow/                     # MLflow artifact store
│   └── <experiment_id>/<run_id>/artifacts/
│       ├── prophet_kl/
│       ├── xgboost_kl/
│       └── ...
│
└── reports/                    # Evidently drift reports
    └── YYYY-MM-DD/
        ├── data_drift_KL.html
        ├── data_drift_Kemaman.html
        └── prediction_drift_KL.html
```

---

## 8. Model Design

### Why Prophet + XGBoost Ensemble?

Prophet handles trend and seasonality well but struggles with sudden changes (e.g., thunderstorm onset). XGBoost on the residuals learns the non-linear patterns Prophet misses. The ensemble is interpretable, fast to train, and outperforms either model alone on Malaysian weather data where monsoon transitions are abrupt.

### Training Strategy

Time-based split — no random shuffle to avoid data leakage:

- **Train:** Jan 2020 – Dec 2023 (4 years)
- **Validation:** Jan 2024 – Jun 2024 (6 months, used for hyperparameter tuning)
- **Test:** Jul 2024 – Dec 2024 (6 months, held out until final evaluation)

One set of models trained per city (5 cities × 2 models = 10 models registered in MLflow).

### Evaluation Metrics

| Metric | Description |
|---|---|
| MAE | Mean Absolute Error — primary metric |
| RMSE | Root Mean Squared Error |
| MAPE | Mean Absolute Percentage Error |
| Coverage | % of actuals within 80% prediction interval |

### Retraining Trigger

- **Scheduled:** Weekly via Cloud Scheduler (every Sunday 2 AM MYT)
- **Drift-triggered:** Evidently JS divergence score > 0.2 on temperature distribution
- **Manual:** POST `/retrain` endpoint call

---

## 9. FastAPI Endpoints

### GET `/health`
```json
{
  "status": "ok",
  "model_versions": { "KL": "v3", "Kemaman": "v3", "Penang": "v2" }
}
```

### GET `/predict?city=KL&hours=24`
```json
{
  "city": "KL",
  "forecast_horizon_hours": 24,
  "predictions": [
    {
      "timestamp": "2024-11-01T08:00:00Z",
      "temperature_c": 29.4,
      "lower_bound": 27.1,
      "upper_bound": 31.7
    }
  ],
  "model_version": "v3",
  "generated_at": "2024-11-01T07:00:00Z"
}
```

### POST `/retrain`
Triggers training pipeline asynchronously. Returns job ID for status polling.

### GET `/model/info`
Returns active model metadata from MLflow registry (version, training date, validation MAE).

### GET `/metrics`
Prometheus scrape endpoint — request count, latency histogram, model MAE gauge, drift score gauge.

---

## 10. Monitoring Design

### Evidently Reports (Daily via Cloud Scheduler)

**Data Drift Report:**
- Reference window: same calendar period last year (controls for seasonality)
- Current window: last 7 days
- Features monitored: temperature_2m, precipitation, humidity, windspeed
- Method: Jensen-Shannon divergence
- Alert threshold: JS > 0.2

**Prediction Drift Report:**
- Compares predicted vs. actual temperature distribution (last 7 days)
- Rolling MAE trend — alert if 7-day MAE degrades >15% vs. training MAE

Reports saved as HTML to `gs://weather-mlops-artifacts/reports/YYYY-MM-DD/` and linked from the Streamlit dashboard.

### Prometheus Metrics

| Metric | Type | Description |
|---|---|---|
| `weather_api_requests_total` | Counter | Total requests by city and endpoint |
| `weather_api_latency_seconds` | Histogram | Request latency distribution |
| `weather_model_mae` | Gauge | Rolling 7-day MAE per city |
| `weather_drift_score` | Gauge | Latest JS divergence score per city |
| `weather_predictions_total` | Counter | Total predictions served |

### Grafana Dashboard Panels

1. Request rate over time (by city)
2. P50 / P95 latency
3. Rolling 7-day MAE per city
4. Temperature drift score with alert threshold line
5. Predicted vs. actual temperature overlay (last 24h)
6. Retraining events timeline

---

## 11. Local Development

Docker Compose is used for local development only — it replicates the GCP services locally so you can develop without incurring cloud costs.

```yaml
# docker-compose.yml (local dev only)
services:
  api:          # FastAPI — mirrors Cloud Run
  ui:           # Streamlit — mirrors Cloud Run
  mlflow:       # MLflow tracking server (local)
  minio:        # GCS substitute (S3-compatible, local artifact store)
  postgres:     # BigQuery substitute (local only)
  prometheus:
  grafana:
```

In production everything except Prometheus/Grafana runs on GCP. Grafana Cloud free tier replaces the local Grafana container in production.

---

## 12. Cloud Scheduler Jobs

```bash
# Hourly ingestion
gcloud scheduler jobs create http ingest-weather \
  --schedule="0 * * * *" \
  --uri="https://weather-ingestion-<hash>-as.a.run.app/run" \
  --time-zone="Asia/Kuala_Lumpur"

# Daily feature pipeline
gcloud scheduler jobs create http feature-pipeline \
  --schedule="30 0 * * *" \
  --uri="https://weather-features-<hash>-as.a.run.app/run" \
  --time-zone="Asia/Kuala_Lumpur"

# Weekly retraining
gcloud scheduler jobs create http retrain \
  --schedule="0 2 * * SUN" \
  --uri="https://weather-training-<hash>-as.a.run.app/run" \
  --time-zone="Asia/Kuala_Lumpur"

# Daily drift monitoring
gcloud scheduler jobs create http drift-monitor \
  --schedule="0 6 * * *" \
  --uri="https://weather-monitoring-<hash>-as.a.run.app/run" \
  --time-zone="Asia/Kuala_Lumpur"
```

---

## 13. Environment Variables

```env
# GCP
GCP_PROJECT_ID=your-project-id
GCP_REGION=asia-southeast1
GCS_BUCKET=weather-mlops-artifacts
BQ_DATASET_RAW=raw_weather
BQ_DATASET_FEATURES=feature_store
BQ_DATASET_PREDICTIONS=predictions

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000          # local
# MLFLOW_TRACKING_URI=https://mlflow-<hash>-as.a.run.app  # production

# Model
TARGET_CITIES=KL,Kemaman,Penang,JB,KK
DRIFT_ALERT_THRESHOLD=0.2
MAE_DEGRADATION_THRESHOLD=0.15

# Ports (local dev only)
FASTAPI_PORT=8000
STREAMLIT_PORT=8501
MLFLOW_PORT=5000
```

---

## 14. Development Milestones

### Week 1 — Data Foundation
- [ ] Set up GCP project, enable APIs (BigQuery, GCS, Cloud Run, Scheduler, Artifact Registry)
- [ ] Create GCS buckets and BigQuery datasets via `infra/` scripts
- [ ] Write `open_meteo_client.py` (historical + forecast endpoints)
- [ ] Write `backfill.py` — pull 4 years of hourly data for 5 cities into BigQuery
- [ ] Write `ingest.py` — hourly ingestion job, containerised and deployed to Cloud Run
- [ ] Set up Cloud Scheduler for hourly ingestion trigger
- [ ] Verify data in BigQuery, write `01_eda.ipynb`

### Week 2 — Feature Engineering + Baseline Model
- [ ] Write `feature_pipeline.py` (reads BigQuery raw → writes BigQuery feature_store)
- [ ] Deploy feature pipeline as Cloud Run job, schedule daily
- [ ] Write `02_feature_engineering.ipynb` to validate feature quality
- [ ] Train Prophet baseline per city, log runs to MLflow (local)
- [ ] Evaluate on validation set, document MAE per city
- [ ] Write `03_model_experiments.ipynb`

### Week 3 — Ensemble + MLflow + FastAPI + CI/CD
- [ ] Train XGBoost residual model, log to MLflow
- [ ] Implement ensemble: Prophet forecast + XGBoost correction
- [ ] Register models in MLflow Model Registry (GCS-backed artifacts)
- [ ] Build FastAPI (`main.py`, `predictor.py`, `schemas.py`)
- [ ] Write `tests/test_api.py`
- [ ] Set up GitHub Actions workflow (`deploy.yml`)
- [ ] First successful Cloud Run deployment via `git push`

### Week 4 — Streamlit UI + Monitoring + Full Stack
- [ ] Build Streamlit UI with forecast charts and drift report links
- [ ] Deploy Streamlit to Cloud Run with its own `deploy_ui.yml` workflow
- [ ] Implement Evidently drift reports, save HTML to GCS
- [ ] Expose Prometheus metrics from FastAPI
- [ ] Set up Grafana Cloud free tier, connect Prometheus
- [ ] Build Grafana dashboard (all 6 panels)
- [ ] Deploy training + monitoring pipelines to Cloud Run, schedule via Cloud Scheduler
- [ ] End-to-end test: data in BigQuery → prediction via Cloud Run API → drift report in GCS

### Polish Week — Portfolio Ready
- [ ] Write comprehensive README with architecture diagram (Mermaid)
- [ ] Add architecture PNG to README (export from draw.io or Excalidraw)
- [ ] Record demo GIF: Streamlit UI → live forecast → Grafana dashboard
- [ ] Add `CONTRIBUTING.md` and `CHANGELOG.md`
- [ ] Ensure all secrets are in GitHub Secrets, no hardcoded credentials
- [ ] Push to GitHub with meaningful commit history

---

## 15. What This Project Demonstrates to Recruiters

| Skill Area | Evidence |
|---|---|
| Time series ML | Lag features, cyclical encoding, no-leakage temporal split, multi-step forecast |
| MLOps maturity | Feature store → model registry → versioned serving → drift monitoring |
| Cloud engineering | GCP (Cloud Run, BigQuery, GCS, Scheduler, Artifact Registry) |
| CI/CD | GitHub Actions → Artifact Registry → Cloud Run (same pattern as Telco Churn) |
| Data engineering | Scheduled ingestion, BigQuery partitioned tables, idempotent writes |
| Software engineering | Clean service separation, Pydantic schemas, pytest, Docker multi-stage builds |
| Observability | Prometheus metrics, Grafana dashboards, automated Evidently drift alerts |
| Domain relevance | Malaysian cities, monsoon seasonality as a real and meaningful drift signal |

---

*Last updated: June 2026*