"""
pipeline.py — Runs continuously; predicts delays and learns from live data.

Usage:
    python pipeline.py          # requires model.pkl from train_base.py

Every 60 seconds:
  - Fetches plan + changes for all 12 stations
  - Skips IC/ICE
  - Upserts trips into PostgreSQL
  - Predicts delay with the River model
  - Stores predictions in PostgreSQL
  - Learns when actual_arr is known
  - Saves model.pkl every 500 learn_one() calls
"""

import json
import logging
import os
import pickle
import sys
import time
from collections import deque
from datetime import datetime

import psycopg2
from dotenv import load_dotenv

from preprocessing import SKIP_TYPES, build_features, compute_delay_min
from scraper import (
    BASE, HEADERS, STATIONS,
    build_changes_map, fetch_changes, fetch_plan, parse_time,
)

load_dotenv()

MODEL_PATH = "model.pkl"
LOG_PATH   = "model_log.jsonl"
SAVE_EVERY = 500

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="[%H:%M:%S]",
)
log = logging.getLogger(__name__)

DB_CONFIG = {
    "host":     os.getenv("DB_HOST", "localhost"),
    "port":     int(os.getenv("DB_PORT", 5432)),
    "dbname":   os.getenv("DB_NAME", "db_delays"),
    "user":     os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASS"),
}

# ── Schema ────────────────────────────────────────────────────────────────────

_CREATE_TRIPS = """
CREATE TABLE IF NOT EXISTS trips (
    id              BIGSERIAL PRIMARY KEY,
    train_id        TEXT        NOT NULL,
    train_type      TEXT,
    line            TEXT,
    station_name    TEXT,
    station_eva     TEXT,
    scheduled_dep   TIMESTAMPTZ,
    scheduled_arr   TIMESTAMPTZ,
    actual_dep      TIMESTAMPTZ,
    actual_arr      TIMESTAMPTZ,
    platform_sched  TEXT,
    platform_actual TEXT,
    cancelled       BOOLEAN     DEFAULT FALSE,
    direction       TEXT,
    ppth            TEXT,
    scraped_at      TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (train_id, station_eva, scheduled_dep)
);
"""

_CREATE_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    id                  BIGSERIAL   PRIMARY KEY,
    train_id            TEXT        NOT NULL,
    station_eva         TEXT        NOT NULL,
    scheduled_arr       TIMESTAMPTZ,
    predicted_delay_min FLOAT,
    actual_delay_min    FLOAT,
    predicted_at        TIMESTAMPTZ DEFAULT NOW()
);
"""

_INSERT_TRIP = """
INSERT INTO trips (
    train_id, train_type, line, station_name, station_eva,
    scheduled_dep, scheduled_arr, actual_dep, actual_arr,
    platform_sched, platform_actual, cancelled, direction, ppth, scraped_at
) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
ON CONFLICT (train_id, station_eva, COALESCE(scheduled_dep, scheduled_arr)) DO UPDATE SET
    actual_dep      = EXCLUDED.actual_dep,
    actual_arr      = EXCLUDED.actual_arr,
    platform_actual = EXCLUDED.platform_actual,
    cancelled       = EXCLUDED.cancelled,
    ppth            = EXCLUDED.ppth,
    scraped_at      = NOW();
"""

_INSERT_PRED = """
INSERT INTO predictions
    (train_id, station_eva, scheduled_arr, predicted_delay_min, actual_delay_min)
VALUES (%s, %s, %s, %s, %s);
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _setup_tables(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(_CREATE_TRIPS)
        cur.execute(_CREATE_PREDICTIONS)
        cur.execute("ALTER TABLE trips ADD COLUMN IF NOT EXISTS ppth TEXT;")
    conn.commit()
    log.info("DB tables ready.")


def _extract_trip(station_name: str, eva: str, s_elem, changes: dict) -> dict:
    trip_id = s_elem.get("id")
    tl      = s_elem.find("tl")
    dp      = s_elem.find("dp")
    ar      = s_elem.find("ar")
    ref     = dp if dp is not None else ar

    train_type = tl.get("c") if tl is not None else None
    line       = ref.get("l")         if ref is not None else None
    ppth_full  = ref.get("ppth", "")  if ref is not None else ""
    direction  = ppth_full.split("|")[-1] if ppth_full else None
    sched_dep  = parse_time(dp.get("pt")) if dp is not None else None
    sched_arr  = parse_time(ar.get("pt")) if ar is not None else None
    plat_sched = ref.get("pp")        if ref is not None else None

    actual_dep = actual_arr = plat_actual = None
    cancelled  = False

    chg = changes.get(trip_id)
    if chg is not None:
        cdp = chg.find("dp")
        car = chg.find("ar")
        if cdp is not None:
            actual_dep  = parse_time(cdp.get("ct"))
            plat_actual = cdp.get("cp")
            cancelled   = cdp.get("cs") == "c"
        if car is not None:
            actual_arr = parse_time(car.get("ct"))
            if plat_actual is None:
                plat_actual = car.get("cp")
            if not cancelled:
                cancelled = car.get("cs") == "c"

    return {
        "train_id":       trip_id,
        "train_type":     train_type,
        "line":           line,
        "station_name":   station_name,
        "station_eva":    eva,
        "scheduled_dep":  sched_dep,
        "scheduled_arr":  sched_arr,
        "actual_dep":     actual_dep,
        "actual_arr":     actual_arr,
        "platform_sched": plat_sched,
        "platform_actual":plat_actual,
        "cancelled":      cancelled,
        "direction":      direction,
        "ppth":           ppth_full,
    }


def _save_model(model) -> None:
    tmp = MODEL_PATH + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(model, f)
    os.replace(tmp, MODEL_PATH)  # atomic overwrite


def _append_log(trips_learned: int, e100: deque, e500: deque) -> None:
    mae100 = sum(e100) / len(e100) if e100 else 0.0
    mae500 = sum(e500) / len(e500) if e500 else 0.0
    entry = {
        "ts":            datetime.now().isoformat(timespec="seconds"),
        "trips_learned": trips_learned,
        "mae_100":       round(mae100, 4),
        "mae_500":       round(mae500, 4),
    }
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def _log_summary(trips_learned: int, e100: deque, e500: deque) -> None:
    mae100 = sum(e100) / len(e100) if e100 else 0.0
    mae500 = sum(e500) / len(e500) if e500 else 0.0
    log.info(
        f"=== Model Update #{trips_learned} ===\n"
        f"  Rolling MAE (last  100): {mae100:.2f} min\n"
        f"  Rolling MAE (last  500): {mae500:.2f} min\n"
        f"  Total trips learned    : {trips_learned}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    if not os.path.exists(MODEL_PATH):
        sys.exit(f"ERROR: {MODEL_PATH} not found — run train_base.py first.")

    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    log.info(f"Model loaded from {MODEL_PATH}")

    conn = psycopg2.connect(**DB_CONFIG)
    _setup_tables(conn)

    trips_learned = 0
    errors_100: deque = deque(maxlen=100)
    errors_500: deque = deque(maxlen=500)
    learned_keys: set = set()   # prevents re-learning the same arrival
    train_delays: dict = {}     # last known delay per train_id for upstream propagation

    log.info("Pipeline running — Ctrl+C to stop.\n")

    try:
        while True:
            cycle_count = 0

            for station_name, eva in STATIONS.items():
                try:
                    plan    = fetch_plan(eva)
                    changes = build_changes_map(fetch_changes(eva))
                except Exception as exc:
                    log.warning(f"  ✗ {station_name}: fetch error — {exc}")
                    continue

                station_count = 0
                for s_elem in plan.findall("s"):
                    try:
                        row = _extract_trip(station_name, eva, s_elem, changes)

                        if row["train_type"] in SKIP_TYPES:
                            continue

                        row["upstream_delay_min"] = train_delays.get(row["train_id"])
                        features = build_features(row)
                        raw_pred = model.predict_one(features)
                        pred     = max(-30.0, min(120.0, raw_pred))
                        delay    = compute_delay_min(row["scheduled_arr"], row["actual_arr"])
                        if delay is not None:
                            train_delays[row["train_id"]] = delay

                        with conn.cursor() as cur:
                            cur.execute(_INSERT_TRIP, (
                                row["train_id"],   row["train_type"],    row["line"],
                                row["station_name"], row["station_eva"],
                                row["scheduled_dep"], row["scheduled_arr"],
                                row["actual_dep"],    row["actual_arr"],
                                row["platform_sched"], row["platform_actual"],
                                row["cancelled"],  row["direction"], row["ppth"],
                            ))
                        conn.commit()

                        with conn.cursor() as cur:
                            cur.execute(_INSERT_PRED, (
                                row["train_id"], row["station_eva"],
                                row["scheduled_arr"], pred, delay,
                            ))
                        conn.commit()

                        if delay is not None and -120.0 <= delay <= 120.0:
                            key = (row["train_id"], eva, str(row["scheduled_arr"]))
                            if key not in learned_keys:
                                model.learn_one(features, delay)
                                learned_keys.add(key)
                                trips_learned += 1
                                err = abs(delay - pred)
                                errors_100.append(err)
                                errors_500.append(err)
                                mae_100 = sum(errors_100) / len(errors_100)

                                log.info(
                                    f"  learn #{trips_learned:,} | "
                                    f"actual={delay:+.1f}min pred={pred:+.1f}min | "
                                    f"MAE(100)={mae_100:.2f}min"
                                )

                                if trips_learned % SAVE_EVERY == 0:
                                    _save_model(model)
                                    _append_log(trips_learned, errors_100, errors_500)
                                    _log_summary(trips_learned, errors_100, errors_500)

                        station_count += 1
                        cycle_count   += 1

                    except Exception as exc:
                        log.warning(f"    Trip error ({station_name}): {exc}")
                        try:
                            conn.rollback()
                        except Exception:
                            pass

                log.info(f"  ✓ {station_name}: {station_count} trips")

            log.info(
                f"Cycle done — {cycle_count} trips | "
                f"learned total: {trips_learned:,} | "
                f"sleeping 60s ..."
            )
            time.sleep(60)

    except KeyboardInterrupt:
        log.info("Stopped by user — saving model ...")
    finally:
        _save_model(model)
        conn.close()
        log.info("Model saved and DB connection closed.")


if __name__ == "__main__":
    main()
