# DB Delay Tracker

A real-time Deutsche Bahn train delay prediction system using online machine learning. Scrapes live data from the DB Timetables API every 60 seconds, predicts arrival delays for 12 stations in the Stuttgart S-Bahn network, and displays everything on a live Streamlit dashboard.

## Screenshots

> Add screenshots of your dashboard to `docs/screenshots/` and reference them here.

| Overall Network View | Station Detail |
|---|---|
| ![Overview](docs/screenshots/overview.png) | ![Station](docs/screenshots/station.png) |

## Notebooks

Step-by-step explanations — open with nbviewer if GitHub preview fails:

| Notebook | nbviewer |
|---|---|
| 01 — Data & the DB API | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Murpppp/db-delay-tracker/blob/main/notebooks/01_data_and_api.ipynb) |
| 02 — Feature Engineering | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Murpppp/db-delay-tracker/blob/main/notebooks/02_feature_engineering.ipynb) |
| 03 — Model Training & Eval | [![nbviewer](https://raw.githubusercontent.com/jupyter/design/master/logos/Badges/nbviewer_badge.svg)](https://nbviewer.org/github/Murpppp/db-delay-tracker/blob/main/notebooks/03_model_training_and_eval.ipynb) |

## Architecture

```
DB Timetables API  (plan + fchg endpoints)
        │
        ▼
   scraper.py       fetch XML for 12 stations every 60s
        │
        ▼
  pipeline.py       extract features → predict → learn → store
        │
   ┌────┴────┐
   ▼         ▼
PostgreSQL   model.pkl     (HoeffdingAdaptiveTreeRegressor)
   │
   ▼
dashboard.py         Streamlit live dashboard (auto-refresh 60s)
```

## Tech Stack

| Layer | Technology |
|---|---|
| Online ML | [River](https://riverml.xyz/) — `HoeffdingAdaptiveTreeRegressor` |
| Database | PostgreSQL + psycopg2 |
| Dashboard | Streamlit + Plotly |
| Data source | Deutsche Bahn Timetables API v1 |
| Language | Python 3.10+ |

## Features (29 total)

The model uses 29 features across 5 groups:

- **Time** — hour, weekday, minutes since midnight
- **Cyclical encodings** — sin/cos for hour, minute, weekday, week, month (avoids discontinuities at midnight/Monday/January)
- **Real-time signal** — departure delay at current station (`dep_delay_min`, `dep_delay_known`)
- **Route topology** — route length, position in route (0–1), route identity hash (from `ppth`)
- **Upstream propagation** — last confirmed delay for this train at any prior station (`upstream_delay_min`)

## Model

Uses River's `HoeffdingAdaptiveTreeRegressor` — an online decision tree that:
- Updates with every confirmed arrival (`learn_one`)
- Never requires retraining from scratch
- Adapts to concept drift (seasonal patterns, timetable changes) over time
- Saves a checkpoint (`model.pkl`) every 500 learned trips

## Results (after ~1 week of live learning)

| Metric | Baseline (CSV) | Live |
|---|---|---|
| MAE | ~5.4 min | ~2.5 min |
| Within ±5 min | ~72% | ~88% |
| R² | negative | ~0.4+ |

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL (running locally or remotely)
- Deutsche Bahn Timetables API key — get one at [developers.deutschebahn.com](https://developers.deutschebahn.com)

### Installation

```bash
git clone https://github.com/your-username/db-delay-tracker
cd db-delay-tracker
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your credentials
```

### Database Setup

```sql
CREATE DATABASE db_delays;
```

Tables (`trips` and `predictions`) are created automatically on first pipeline run.

### Running

**Step 1 — Train base model** (requires a historical CSV export named `trips_final.csv` or `trips.csv`):
```bash
python train_base.py
```
Produces `model.pkl` and `model_eval.json`.

**Step 2 — Start the live pipeline:**
```bash
python pipeline.py
```

**Step 3 — Launch the dashboard** (in a separate terminal):
```bash
streamlit run dashboard.py
```

### Retraining on live data

After collecting a few days of live data, export from PostgreSQL and retrain:

```sql
COPY (SELECT * FROM trips ORDER BY scheduled_arr)
TO '/path/to/trips_final.csv' CSV HEADER;
```

Then re-run `train_base.py`. The new model will have seen all the route and upstream delay features from the start.

## Project Structure

```
db-delay-tracker/
├── scraper.py          DB API fetching + station list
├── preprocessing.py    Feature engineering (shared by pipeline + train_base)
├── pipeline.py         Live scraping loop + online learning
├── train_base.py       One-time base training from CSV
├── dashboard.py        Streamlit dashboard
├── requirements.txt
├── .env.example
└── notebooks/
    ├── 01_data_and_api.ipynb         How the DB API works
    ├── 02_feature_engineering.ipynb  Why each feature was chosen
    └── 03_model_training_and_eval.ipynb  Training + interpreting results
```

## Stations Tracked

All in the Stuttgart / Vaihingen–Stuttgart S-Bahn corridor:

Vaihingen (Enz) · Sersheim · Sachsenheim · Bietigheim-Bissingen · Asperg ·
Stuttgart-Zuffenhausen · Stuttgart-Feuerbach · Stuttgart Hbf · Ludwigsburg ·
Stuttgart Stadtmitte · Stuttgart Feuersee · Stuttgart Universität

## Notes

- IC/ICE trains are excluded (different delay patterns, separate network)
- The predictions table keeps all raw predictions (one per scrape cycle per trip); the dashboard deduplicates with `DISTINCT ON` for accurate metrics
- `learned_keys` prevents the same trip arrival from being learned twice across cycles
