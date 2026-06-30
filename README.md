# 🌦️ Malaysia Weather Forecast MLOps Pipeline

A production-grade MLOps system that forecasts hourly temperature, precipitation, humidity, and wind speed for 5 Malaysian cities — deployed serverlessly on Google Cloud Run with automated CI/CD via GitHub Actions.

**Live demo:** `https://weather-forecast-api-XXXXXXXX-uc.a.run.app`

---

## 🚀 Overview

This project demonstrates an end-to-end machine learning lifecycle for multi-variable time series forecasting — from raw data ingestion through to a deployed, monitored API with a live dashboard. It combines Prophet (for trend and seasonality) with XGBoost (for residual correction using real-time lag features) to forecast 4 weather variables across 5 cities, surfaced through a custom dashboard with derived weather condition icons.

---

## 🛠 Tech Stack

* **Data Source:** Open-Meteo API (historical archive + forecast endpoints, no API key required)
* **Data Warehouse:** Google BigQuery (partitioned, clustered tables)
* **Modelling:** Prophet (baseline) + XGBoost (residual correction) ensemble
* **Experiment Tracking:** MLflow (local)
* **API:** FastAPI with Pydantic validation
* **Frontend:** HTML + JavaScript + Chart.js (served directly from FastAPI)
* **Containerisation:** Docker
* **CI/CD:** GitHub Actions
* **Deployment:** Google Cloud Run (serverless)
* **Image Registry:** Google Artifact Registry

---

## 🏗 System Architecture

```
Open-Meteo API
      │
      ▼
Ingestion Scripts (backfill.py / ingest.py)
      │
      ▼
BigQuery — raw_weather.hourly
(~265,000 rows · 2020–2025 · 5 cities)
      │
      ▼
Feature Pipeline (feature_pipeline.py)
  - Lag features: 1h, 3h, 6h, 24h, 168h
  - Rolling stats: mean, std (6h, 24h)
  - Cyclical encoding: hour, month (sin/cos)
      │
      ▼
BigQuery — feature_store.hourly
      │
      ▼
Training Pipeline (train.py)
  - Prophet baseline per variable per city
  - XGBoost residual correction
  - MLflow experiment tracking
  - 20 ensemble models (4 variables × 5 cities)
      │
      ▼
GitHub Actions (on every git push)
  ├── Build Docker image
  ├── Push to Artifact Registry
  └── Deploy to Cloud Run
      │
      ▼
FastAPI on Cloud Run
  - Loads all 20 models at startup
  - Fetches recent actuals from BigQuery at request time
  - Computes real-time lag features for accurate corrections
  - Serves combined multi-variable forecast + derived weather condition
      │
      ▼
Dashboard UI (HTML/JS/Chart.js)
  - Current conditions card
  - 24-hour forecast strip
  - Tabbed charts per variable
```

---

## 📊 Forecast Targets

| Variable | Unit | Approach |
|---|---|---|
| Temperature | °C | Prophet + XGBoost ensemble |
| Precipitation | mm | Prophet + XGBoost ensemble |
| Humidity | % | Prophet + XGBoost ensemble |
| Wind Speed | km/h | Prophet + XGBoost ensemble |

**Cities covered:** Kuala Lumpur, Kemaman, Penang, Johor Bahru, Kota Kinabalu

**Weather condition** (Sunny / Partly Cloudy / Cloudy / Light Rain / Rainy) is derived from predicted precipitation and humidity thresholds, displayed as icons throughout the UI.

---

## 🧠 Why Prophet + XGBoost?

Prophet captures long-term seasonality and trend well (daily, weekly, and yearly patterns in Malaysian weather) but struggles to react to current conditions — it only knows what a given date *usually* looks like.

XGBoost is trained on the **residuals** between Prophet's predictions and actual values, using lag and rolling features. At prediction time, the API pulls the last 168 hours of actual weather from BigQuery and computes real lag features — so if there's an unusual cold front or heatwave happening right now, XGBoost can correct Prophet's climatological baseline to reflect what's actually occurring, rather than just what's typical for that date.

**Validation results (MAE):**

| City | Temperature (°C) | Precipitation (mm) | Humidity (%) | Wind Speed (km/h) |
|---|---|---|---|---|
| Kuala Lumpur | 0.49 | 0.34 | 3.10 | 1.20 |
| Kemaman | 0.83 | 0.34 | 4.61 | 2.44 |
| Penang | 0.55 | 0.37 | 2.93 | 1.41 |
| Johor Bahru | 0.73 | 0.27 | 5.24 | 1.44 |
| Kota Kinabalu | 0.69 | 0.34 | 3.31 | 1.31 |

Trained on 2020–2024 data, validated on Jan–Jun 2025, time-based split (no shuffling, no data leakage).

---

## 📋 Key Features

* **Multi-variable forecasting** — temperature, precipitation, humidity, and wind speed predicted simultaneously
* **Real-time lag features** — predictions are corrected using actual recent weather data pulled from BigQuery at request time, not just historical seasonality
* **Derived weather conditions** — sunny/cloudy/rainy icons computed from predicted precipitation and humidity
* **Automated CI/CD** — every `git push` triggers build, containerisation, and deployment to Cloud Run with zero manual steps
* **Interactive dashboard** — current conditions hero card, 24-hour forecast strip, and tabbed charts for each variable
* **Production-ready API** — FastAPI with automatic Swagger documentation at `/docs`

---

## 📁 Repository Structure

```
malaysia-weather-forecast-mlops/
│
├── .github/workflows/
│   └── deploy.yml                  # CI/CD: build, push, deploy to Cloud Run
│
├── api/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py                     # FastAPI app — multi-variable prediction
│   └── static/
│       └── index.html              # Dashboard UI (HTML/JS/Chart.js)
│
├── pipelines/
│   ├── ingestion/
│   │   ├── open_meteo_client.py    # Open-Meteo API wrapper with retry logic
│   │   ├── backfill.py             # One-shot historical data load
│   │   └── ingest.py               # Hourly ingestion job
│   │
│   ├── features/
│   │   └── feature_pipeline.py     # Lag, rolling, cyclical feature engineering
│   │
│   └── training/
│       ├── export_features.py      # BigQuery → local CSV export
│       └── train.py                # Prophet + XGBoost training for all variables
│
├── models/                          # 20 trained model pairs (committed to repo)
├── data/                            # Local feature CSVs (gitignored)
└── credentials/                     # GCP service account key (gitignored)
```

---

## 🚀 How to Run Locally

### 1. Clone the repo

```bash
git clone https://github.com/CCZ1004/malaysia-weather-forecast-mlops.git
cd malaysia-weather-forecast-mlops
```

### 2. Set up a virtual environment

```bash
python -m venv weather
weather\Scripts\activate          # Windows
source weather/bin/activate       # Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r api/requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the project root:

```env
GCP_PROJECT_ID=your-project-id
BQ_DATASET_RAW=raw_weather
BQ_DATASET_FEATURES=feature_store
BQ_DATASET_PREDICTIONS=predictions
GOOGLE_APPLICATION_CREDENTIALS=credentials/gcp-key.json
```

### 5. Run the API locally

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080` for the dashboard, or `http://127.0.0.1:8080/docs` for the Swagger API docs.

---

## ☁️ Deployment

Deployment is fully automated. Every push to `main` triggers:

1. **GitHub Actions** checks out the code
2. **Docker image** is built from `api/Dockerfile` (includes all 20 trained models)
3. Image is pushed to **Google Artifact Registry**
4. **Cloud Run** deploys the new image, replacing the previous revision

No manual deployment steps required — `git push origin main` is the only command needed to ship changes.

---

## 🔌 API Endpoints

### `GET /health`
Returns service status and which cities/variables have models loaded.

### `GET /predict?city={city}&hours={hours}`
Returns a multi-variable forecast for the requested city and horizon (1–168 hours).

```json
{
  "city": "KL",
  "forecast_horizon_hours": 24,
  "predictions": [
    {
      "timestamp": "2026-06-30T08:00:00+00:00",
      "temperature_c": 28.4,
      "precipitation_mm": 0.0,
      "humidity_pct": 78.2,
      "windspeed_kmh": 9.3,
      "condition": "Partly Cloudy",
      "condition_icon": "⛅"
    }
  ],
  "generated_at": "2026-06-30T07:00:00.123456+00:00"
}
```

### `GET /docs`
Interactive Swagger UI for testing all endpoints.

---

## 🔭 Roadmap

- [ ] Cloud Scheduler for automated daily ingestion + feature refresh
- [ ] Evidently AI drift monitoring (data drift + prediction drift reports)
- [ ] Weather condition classifier trained directly on labelled data (vs. current rule-based derivation)
- [ ] City comparison view in the dashboard
- [ ] Prometheus + Grafana observability stack

---

## 👤 Author

**Chiu Chang Ze**  
GitHub: [@CCZ1004](https://github.com/CCZ1004)