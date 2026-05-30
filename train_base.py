"""
train_base.py — Run once to train the model on historical CSV data.

Usage:
    python train_base.py

Expects one of: trips_final.csv, trips.csv
Produces:
    model.pkl       — trained HoeffdingAdaptiveTreeRegressor
    model_eval.json — evaluation metrics on the last 3 days of data
"""

import json
import os
import pickle
import sys

import pandas as pd
from river import metrics, tree

from preprocessing import SKIP_TYPES, build_features, compute_delay_min

CSV_CANDIDATES = ["trips_final.csv", "trips.csv"]
MODEL_PATH     = "model.pkl"
EVAL_PATH      = "model_eval.json"


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in ["scheduled_dep", "scheduled_arr", "actual_dep", "actual_arr", "scraped_at"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")
    return df


def main() -> None:
    csv_path = next((p for p in CSV_CANDIDATES if os.path.exists(p)), None)
    if csv_path is None:
        sys.exit(f"ERROR: CSV not found. Expected one of: {CSV_CANDIDATES}")

    print(f"Loading {csv_path} ...")
    df = load_csv(csv_path)
    print(f"  Total rows: {len(df):,}")

    df = df[~df["train_type"].isin(SKIP_TYPES)].reset_index(drop=True)
    print(f"  After IC/ICE filter: {len(df):,} rows")

    df = df.sort_values("scheduled_arr", na_position="last").reset_index(drop=True)

    max_date = df["scheduled_arr"].max()
    cutoff   = max_date - pd.Timedelta(days=3)
    train_df = df[df["scheduled_arr"] <= cutoff].reset_index(drop=True)
    test_df  = df[df["scheduled_arr"] >  cutoff].reset_index(drop=True)

    print(f"  Max date : {max_date}")
    print(f"  Cutoff   : {cutoff}  (last 3 days → test)")
    print(f"  Train    : {len(train_df):,} rows")
    print(f"  Test     : {len(test_df):,} rows\n")

    model = tree.HoeffdingAdaptiveTreeRegressor()

    print("Training ...")
    trained = skipped = 0
    for _, row in train_df.iterrows():
        features = build_features(row.to_dict())
        delay    = compute_delay_min(row.get("scheduled_arr"), row.get("actual_arr"))
        if delay is None:
            skipped += 1
            continue
        model.learn_one(features, delay)
        trained += 1
        if trained % 5_000 == 0:
            print(f"  {trained:,} trips learned ...")

    print(f"  Done: {trained:,} learned, {skipped:,} skipped (missing arr)\n")

    print("Evaluating on test set ...")
    mae_m  = metrics.MAE()
    rmse_m = metrics.RMSE()
    r2_m   = metrics.R2()
    within_2 = within_5 = total = 0

    for _, row in test_df.iterrows():
        features = build_features(row.to_dict())
        delay    = compute_delay_min(row.get("scheduled_arr"), row.get("actual_arr"))
        if delay is None:
            continue
        pred = model.predict_one(features)
        mae_m.update(delay, pred)
        rmse_m.update(delay, pred)
        r2_m.update(delay, pred)
        err = abs(pred - delay)
        within_2 += err <= 2
        within_5 += err <= 5
        total    += 1

    if total == 0:
        print("  No test trips with actual_arr — skipping metrics.")
        w2_pct = w5_pct = None
    else:
        w2_pct = within_2 / total * 100
        w5_pct = within_5 / total * 100
        print(f"  Test trips evaluated : {total:,}")
        print(f"  MAE                  : {mae_m.get():.3f} min")
        print(f"  RMSE                 : {rmse_m.get():.3f} min")
        print(f"  R²                   : {r2_m.get():.4f}")
        print(f"  Within ±2 min        : {w2_pct:.1f}%")
        print(f"  Within ±5 min        : {w5_pct:.1f}%")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\nModel saved  → {MODEL_PATH}")

    eval_data = {
        "trained_trips":   trained,
        "test_trips":      total,
        "mae_min":         mae_m.get()  if total else None,
        "rmse_min":        rmse_m.get() if total else None,
        "r2":              r2_m.get()   if total else None,
        "within_2min_pct": w2_pct,
        "within_5min_pct": w5_pct,
        "cutoff_date":     str(cutoff),
        "csv_source":      csv_path,
    }
    with open(EVAL_PATH, "w", encoding="utf-8") as f:
        json.dump(eval_data, f, indent=2)
    print(f"Eval saved   → {EVAL_PATH}")


if __name__ == "__main__":
    main()
