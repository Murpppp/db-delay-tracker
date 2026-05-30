import hashlib
import math
from datetime import timezone, timedelta
from typing import Optional

import pandas as pd

_TWO_PI = 2 * math.pi

SKIP_TYPES = {"IC", "ICE"}
_CET = timezone(timedelta(hours=1))


def stable_hash(val: str, mod: int) -> int:
    return int(hashlib.md5(str(val).encode()).hexdigest()[:8], 16) % mod


def _to_local(dt):
    if dt is None:
        return None
    if hasattr(dt, "tzinfo") and dt.tzinfo is not None:
        return dt.astimezone(_CET)
    return dt  # naive datetimes from parse_time() are already German local time


def _is_null(val) -> bool:
    if val is None:
        return True
    try:
        return bool(pd.isna(val))
    except (TypeError, ValueError):
        return False


def build_features(row: dict) -> dict:
    sched_arr  = row.get("scheduled_arr")
    actual_arr = row.get("actual_arr")
    sched_dep  = row.get("scheduled_dep")
    actual_dep = row.get("actual_dep")

    arr_missing = int(_is_null(sched_arr) or _is_null(actual_arr))

    # Real-time signal: how late is this train already at departure?
    dep_delay = compute_delay_min(sched_dep, actual_dep)
    if dep_delay is not None and -120.0 <= dep_delay <= 120.0:
        dep_delay_min   = dep_delay
        dep_delay_known = 1
    else:
        dep_delay_min   = 0
        dep_delay_known = 0

    ref = _to_local(sched_arr if not _is_null(sched_arr) else sched_dep)
    if ref is not None:
        hour     = ref.hour
        minute   = ref.minute
        weekday  = ref.weekday()
        month    = ref.month
        week     = ref.isocalendar()[1]
        minutes_since_midnight = hour * 60 + minute

        hour_sin    = math.sin(_TWO_PI * hour    / 24)
        hour_cos    = math.cos(_TWO_PI * hour    / 24)
        minute_sin  = math.sin(_TWO_PI * minute  / 60)
        minute_cos  = math.cos(_TWO_PI * minute  / 60)
        weekday_sin = math.sin(_TWO_PI * weekday / 7)
        weekday_cos = math.cos(_TWO_PI * weekday / 7)
        week_sin    = math.sin(_TWO_PI * week    / 52)
        week_cos    = math.cos(_TWO_PI * week    / 52)
        month_sin   = math.sin(_TWO_PI * month   / 12)
        month_cos   = math.cos(_TWO_PI * month   / 12)
    else:
        hour = weekday = minutes_since_midnight = 0
        hour_sin = hour_cos = minute_sin = minute_cos = 0.0
        weekday_sin = weekday_cos = 0.0
        week_sin = week_cos = 0.0
        month_sin = month_cos = 0.0

    line = row.get("line")
    if _is_null(line) or str(line).strip() == "":
        line_missing, line_hash = 1, 0
    else:
        line_missing = 0
        line_hash    = stable_hash(str(line), 1000)

    cancelled = row.get("cancelled")
    cancelled_int = 0 if _is_null(cancelled) else int(bool(cancelled))

    direction = row.get("direction")
    direction_hash = 0 if _is_null(direction) else stable_hash(str(direction), 500)

    train_type = row.get("train_type")
    train_type_hash = 0 if _is_null(train_type) else stable_hash(str(train_type), 100)

    try:
        station_eva = int(row.get("station_eva") or 0)
    except (ValueError, TypeError):
        station_eva = 0

    # Route topology features (from ppth = planned path string "StationA|StationB|...")
    ppth             = row.get("ppth") or ""
    station_name_val = row.get("station_name") or ""
    if ppth:
        stops = ppth.split("|")
        route_length = len(stops)
        route_hash   = stable_hash(ppth, 10000)
        try:
            route_position       = stops.index(station_name_val)
            route_position_known = 1
            route_position_pct   = route_position / max(route_length - 1, 1)
        except ValueError:
            route_position       = 0
            route_position_known = 0
            route_position_pct   = 0.0
    else:
        route_length         = 0
        route_hash           = 0
        route_position       = 0
        route_position_known = 0
        route_position_pct   = 0.0

    # Upstream propagation: last known delay for this train at any prior station
    upstream_delay = row.get("upstream_delay_min")
    if upstream_delay is not None and -120.0 <= upstream_delay <= 120.0:
        upstream_delay_min_val = float(upstream_delay)
        upstream_delay_known   = 1
    else:
        upstream_delay_min_val = 0.0
        upstream_delay_known   = 0

    return {
        "hour":                   hour,
        "weekday":                weekday,
        "station_eva":            station_eva,
        "train_type":             train_type_hash,
        "line":                   line_hash,
        "line_missing":           line_missing,
        "arr_missing":            arr_missing,
        "cancelled":              cancelled_int,
        "direction":              direction_hash,
        "minutes_since_midnight": minutes_since_midnight,
        "dep_delay_min":          dep_delay_min,
        "dep_delay_known":        dep_delay_known,
        # Cyclical time encodings
        "hour_sin":               hour_sin,
        "hour_cos":               hour_cos,
        "minute_sin":             minute_sin,
        "minute_cos":             minute_cos,
        "weekday_sin":            weekday_sin,
        "weekday_cos":            weekday_cos,
        "week_sin":               week_sin,
        "week_cos":               week_cos,
        "month_sin":              month_sin,
        "month_cos":              month_cos,
        # Route topology
        "route_length":           route_length,
        "route_hash":             route_hash,
        "route_position":         route_position,
        "route_position_known":   route_position_known,
        "route_position_pct":     route_position_pct,
        # Upstream delay propagation
        "upstream_delay_min":     upstream_delay_min_val,
        "upstream_delay_known":   upstream_delay_known,
    }


def compute_delay_min(sched_arr, actual_arr) -> Optional[float]:
    if _is_null(sched_arr) or _is_null(actual_arr):
        return None
    try:
        return (actual_arr - sched_arr).total_seconds() / 60
    except (TypeError, AttributeError):
        return None
