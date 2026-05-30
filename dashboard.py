"""
dashboard.py — Live browser dashboard for the DB Delay Tracker pipeline.

Usage:
    streamlit run dashboard.py
"""

import json
import os
from datetime import datetime, timezone
import zoneinfo
BERLIN = zoneinfo.ZoneInfo("Europe/Berlin")

import pandas as pd
import plotly.colors as pc
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv
from streamlit_autorefresh import st_autorefresh

load_dotenv()

# ── Constants ─────────────────────────────────────────────────────────────────

STATIONS = {
    "Vaihingen (Enz)":        "8006053",
    "Sersheim":               "8005540",
    "Sachsenheim":            "8005253",
    "Bietigheim-Bissingen":   "8000038",
    "Asperg":                 "8000630",
    "Stuttgart-Zuffenhausen": "8005778",
    "Stuttgart-Feuerbach":    "8005770",
    "Stuttgart Hbf":          "8000096",
    "Ludwigsburg":            "8000235",
    "Stuttgart Stadtmitte":   "8006700",
    "Stuttgart Feuersee":     "8006699",
    "Stuttgart Universität":  "8006513",
}
EVA_TO_NAME = {v: k for k, v in STATIONS.items()}

DB_CONFIG = dict(
    host=os.getenv("DB_HOST", "localhost"),
    port=int(os.getenv("DB_PORT", 5432)),
    dbname=os.getenv("DB_NAME", "db_delays"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS"),
)

DB_RED     = "#EC0016"
SAVE_EVERY = 500

STATION_COLORS = {
    "Vaihingen (Enz)":        "#636EFA",
    "Sersheim":               "#EF553B",
    "Sachsenheim":            "#00CC96",
    "Bietigheim-Bissingen":   "#AB63FA",
    "Asperg":                 "#FFA15A",
    "Stuttgart-Zuffenhausen": "#19D3F3",
    "Stuttgart-Feuerbach":    "#FF6692",
    "Stuttgart Hbf":          "#B6E880",
    "Ludwigsburg":            "#FF97FF",
    "Stuttgart Stadtmitte":   "#FECB52",
    "Stuttgart Feuersee":     "#72B7B2",
    "Stuttgart Universität":  "#54A24B",
}
DB_DARK   = "#282D37"
CHART_SEQ = [DB_RED, "#F0A500", "#00A878", "#0075BF", "#8B5CF6"]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DB Delay Tracker",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    [data-testid="metric-container"] { background:#f8f9fa; border-radius:8px; padding:12px; }
    .block-container { padding-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

# Auto-refresh every 60 seconds
st_autorefresh(interval=60_000, key="autorefresh")

# ── DB helpers ────────────────────────────────────────────────────────────────

def _qdf(sql: str, params=None) -> pd.DataFrame:
    conn = psycopg2.connect(**DB_CONFIG)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        if cur.description is None:
            return pd.DataFrame()
        cols = [d[0] for d in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)
    finally:
        conn.close()


@st.cache_data(ttl=60)
def fetch_overall_kpis(days: int = None) -> pd.DataFrame:
    date_filter = f"AND predicted_at >= NOW() - INTERVAL '{days} days'" if days else ""
    return _qdf(f"""
        WITH deduped AS (
            SELECT DISTINCT ON (train_id, station_eva, scheduled_arr)
                predicted_delay_min, actual_delay_min
            FROM predictions
            WHERE actual_delay_min IS NOT NULL {date_filter}
            ORDER BY train_id, station_eva, scheduled_arr, predicted_at DESC
        )
        SELECT
            (SELECT COUNT(*) FROM trips)       AS total_trips,
            (SELECT COUNT(*) FROM predictions) AS total_predictions,
            COUNT(*)                           AS evaluated,
            ROUND(AVG(ABS(predicted_delay_min - actual_delay_min))::numeric, 2) AS overall_mae,
            ROUND(COUNT(*) FILTER (WHERE ABS(predicted_delay_min - actual_delay_min) <= 2) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS within_2_pct,
            ROUND(COUNT(*) FILTER (WHERE ABS(predicted_delay_min - actual_delay_min) <= 5) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS within_5_pct
        FROM deduped
    """)


@st.cache_data(ttl=60)
def fetch_station_summary() -> pd.DataFrame:
    df = _qdf("""
        SELECT
            station_eva,
            station_name,
            COUNT(*)                                                                    AS trips,
            ROUND(AVG(EXTRACT(EPOCH FROM (actual_arr - scheduled_arr)) / 60)::numeric, 2) AS avg_delay_min,
            ROUND(COUNT(*) FILTER (WHERE cancelled) * 100.0 / NULLIF(COUNT(*), 0)::numeric, 1) AS cancel_pct,
            COUNT(*) FILTER (WHERE actual_arr IS NOT NULL)                              AS with_actual
        FROM trips
        GROUP BY station_eva, station_name
        ORDER BY avg_delay_min DESC NULLS LAST
    """)
    if not df.empty:
        df["station_name"] = df["station_name"].fillna(df["station_eva"].map(EVA_TO_NAME))
    return df


@st.cache_data(ttl=60)
def fetch_mae_over_time(days: int = None) -> pd.DataFrame:
    date_filter = f"AND predicted_at >= NOW() - INTERVAL '{days} days'" if days else ""
    return _qdf(f"""
        WITH deduped AS (
            SELECT DISTINCT ON (train_id, station_eva, scheduled_arr)
                predicted_delay_min, actual_delay_min,
                TO_TIMESTAMP(FLOOR(EXTRACT(EPOCH FROM predicted_at) / 1800) * 1800) AS bucket
            FROM predictions
            WHERE actual_delay_min IS NOT NULL {date_filter}
            ORDER BY train_id, station_eva, scheduled_arr, predicted_at DESC
        )
        SELECT
            bucket AS hour,
            ROUND(AVG(ABS(predicted_delay_min - actual_delay_min))::numeric, 3) AS mae,
            COUNT(*) AS n
        FROM deduped
        GROUP BY 1
        ORDER BY 1
    """)


@st.cache_data(ttl=60)
def fetch_delay_dist(station_eva: str = None) -> pd.DataFrame:
    if station_eva:
        return _qdf("""
            SELECT ROUND((EXTRACT(EPOCH FROM (actual_arr - scheduled_arr)) / 60)::numeric, 1) AS delay_min
            FROM trips
            WHERE actual_arr IS NOT NULL AND scheduled_arr IS NOT NULL AND station_eva = %s
        """, (station_eva,))
    return _qdf("""
        SELECT ROUND((EXTRACT(EPOCH FROM (actual_arr - scheduled_arr)) / 60)::numeric, 1) AS delay_min
        FROM trips
        WHERE actual_arr IS NOT NULL AND scheduled_arr IS NOT NULL
    """)


@st.cache_data(ttl=60)
def fetch_station_kpis(station_eva: str, days: int = None) -> pd.DataFrame:
    date_filter = f"AND predicted_at >= NOW() - INTERVAL '{days} days'" if days else ""
    return _qdf(f"""
        WITH deduped AS (
            SELECT DISTINCT ON (train_id, station_eva, scheduled_arr)
                predicted_delay_min, actual_delay_min
            FROM predictions
            WHERE station_eva = %s AND actual_delay_min IS NOT NULL {date_filter}
            ORDER BY train_id, station_eva, scheduled_arr, predicted_at DESC
        )
        SELECT
            (SELECT COUNT(*) FROM trips WHERE station_eva = %s) AS total_trips,
            ROUND(AVG(ABS(predicted_delay_min - actual_delay_min))::numeric, 2) AS mae,
            ROUND(COUNT(*) FILTER (WHERE ABS(predicted_delay_min - actual_delay_min) <= 2) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS within_2_pct,
            ROUND(COUNT(*) FILTER (WHERE ABS(predicted_delay_min - actual_delay_min) <= 5) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS within_5_pct,
            (SELECT ROUND(COUNT(*) FILTER (WHERE cancelled) * 100.0 / NULLIF(COUNT(*), 0), 1)
             FROM trips WHERE station_eva = %s) AS cancel_pct
        FROM deduped
    """, (station_eva, station_eva, station_eva))


@st.cache_data(ttl=60)
def fetch_delay_by_hour(station_eva: str) -> pd.DataFrame:
    return _qdf("""
        SELECT
            EXTRACT(HOUR FROM scheduled_arr AT TIME ZONE 'Europe/Berlin')::int AS hour,
            ROUND(AVG(EXTRACT(EPOCH FROM (actual_arr - scheduled_arr)) / 60)::numeric, 2) AS avg_delay,
            COUNT(*) AS trips
        FROM trips
        WHERE station_eva = %s AND actual_arr IS NOT NULL AND scheduled_arr IS NOT NULL
        GROUP BY 1
        ORDER BY 1
    """, (station_eva,))


@st.cache_data(ttl=60)
def fetch_pred_vs_actual(station_eva: str) -> pd.DataFrame:
    return _qdf("""
        SELECT actual, predicted, predicted_at FROM (
            SELECT DISTINCT ON (train_id, scheduled_arr)
                ROUND(actual_delay_min::numeric, 1)    AS actual,
                ROUND(predicted_delay_min::numeric, 1) AS predicted,
                predicted_at
            FROM predictions
            WHERE station_eva = %s AND actual_delay_min IS NOT NULL
            ORDER BY train_id, scheduled_arr, predicted_at DESC
        ) deduped
        ORDER BY predicted_at DESC
        LIMIT 500
    """, (station_eva,))


@st.cache_data(ttl=60)
def fetch_recent_trips(station_eva: str) -> pd.DataFrame:
    return _qdf("""
        SELECT
            train_type,
            line,
            direction,
            TO_CHAR(scheduled_arr AT TIME ZONE 'Europe/Berlin', 'DD.MM HH24:MI') AS sched_arr,
            TO_CHAR(actual_arr    AT TIME ZONE 'Europe/Berlin', 'DD.MM HH24:MI') AS actual_arr,
            ROUND((EXTRACT(EPOCH FROM (actual_arr - scheduled_arr)) / 60)::numeric, 1) AS delay_min,
            cancelled
        FROM trips
        WHERE station_eva = %s
        ORDER BY scheduled_arr DESC
        LIMIT 100
    """, (station_eva,))


@st.cache_data(ttl=60)
def fetch_next_trains(station_eva: str) -> pd.DataFrame:
    return _qdf("""
        SELECT * FROM (
            SELECT DISTINCT ON (t.train_id, t.scheduled_arr)
                t.train_type,
                COALESCE(t.line, '—')      AS line,
                COALESCE(t.direction, '—') AS direction,
                TO_CHAR(t.scheduled_arr AT TIME ZONE 'Europe/Berlin', 'HH24:MI') AS sched_arr,
                t.scheduled_arr            AS sched_arr_raw,
                ROUND(p.predicted_delay_min::numeric, 1)                          AS predicted_delay_min
            FROM trips t
            LEFT JOIN LATERAL (
                SELECT predicted_delay_min
                FROM predictions
                WHERE train_id = t.train_id AND station_eva = t.station_eva
                ORDER BY predicted_at DESC
                LIMIT 1
            ) p ON true
            WHERE t.station_eva = %s
              AND t.scheduled_arr > NOW()
              AND t.cancelled = false
            ORDER BY t.train_id, t.scheduled_arr, t.scraped_at DESC
        ) deduped
        ORDER BY sched_arr_raw ASC
        LIMIT 15
    """, (station_eva,))


@st.cache_data(ttl=60)
def fetch_live_accuracy(days: int = None) -> pd.DataFrame:
    date_filter = f"AND predicted_at >= NOW() - INTERVAL '{days} days'" if days else ""
    return _qdf(f"""
        WITH deduped AS (
            SELECT DISTINCT ON (train_id, station_eva, scheduled_arr)
                predicted_delay_min, actual_delay_min
            FROM predictions
            WHERE actual_delay_min IS NOT NULL {date_filter}
            ORDER BY train_id, station_eva, scheduled_arr, predicted_at DESC
        ),
        mean_val AS (
            SELECT AVG(actual_delay_min) AS mean_actual FROM deduped
        )
        SELECT
            COUNT(*)                                                                        AS evaluated,
            ROUND(AVG(ABS(predicted_delay_min - actual_delay_min))::numeric, 3)            AS mae,
            ROUND(SQRT(AVG(POWER(predicted_delay_min - actual_delay_min, 2)))::numeric, 3) AS rmse,
            ROUND(
                (1 - SUM(POWER(actual_delay_min - predicted_delay_min, 2)) /
                NULLIF(SUM(POWER(actual_delay_min - (SELECT mean_actual FROM mean_val), 2)), 0))::numeric
            , 4)                                                                            AS r2,
            ROUND(COUNT(*) FILTER (WHERE ABS(predicted_delay_min - actual_delay_min) <= 2) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS within_2_pct,
            ROUND(COUNT(*) FILTER (WHERE ABS(predicted_delay_min - actual_delay_min) <= 5) * 100.0
                  / NULLIF(COUNT(*), 0), 1) AS within_5_pct
        FROM deduped
    """)


def load_model_log() -> pd.DataFrame:
    try:
        rows = []
        with open("model_log.jsonl", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        df["ts"] = pd.to_datetime(df["ts"]).dt.tz_localize("UTC").dt.tz_convert("Europe/Berlin")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


def load_model_eval() -> dict:
    try:
        with open("model_eval.json", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@st.cache_data(ttl=60)
def fetch_mae_by_station() -> pd.DataFrame:
    return _qdf("""
        WITH deduped AS (
            SELECT DISTINCT ON (train_id, station_eva, scheduled_arr)
                station_eva, predicted_delay_min, actual_delay_min
            FROM predictions
            WHERE actual_delay_min IS NOT NULL
            ORDER BY train_id, station_eva, scheduled_arr, predicted_at DESC
        )
        SELECT
            station_eva,
            ROUND(AVG(ABS(predicted_delay_min - actual_delay_min))::numeric, 2) AS mae,
            COUNT(*) AS evaluated
        FROM deduped
        GROUP BY station_eva
        ORDER BY mae DESC
    """)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🚂 DB Delay Tracker")
    st.caption(f"Updated: {datetime.now(BERLIN).strftime('%H:%M:%S')} · auto-refresh 60s")
    st.divider()

    days_filter = st.radio("Evaluation window", ["Last 7 days", "All time"], index=0)
    days = 7 if days_filter == "Last 7 days" else None
    st.divider()

    options = ["🗺 All Stations"] + list(STATIONS.keys())
    selected = st.radio("Select view", options, label_visibility="collapsed")

    st.divider()
    if st.button("🔄 Force refresh", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# ── All Stations view ─────────────────────────────────────────────────────────

if selected == "🗺 All Stations":
    st.header("Overall Network Overview")

    kpis = fetch_overall_kpis(days)
    r = kpis.iloc[0] if not kpis.empty else {}

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Trips",       f"{int(r.get('total_trips') or 0):,}")
    c2.metric("Predictions Made",  f"{int(r.get('total_predictions') or 0):,}")
    c3.metric("Evaluated",         f"{int(r.get('evaluated') or 0):,}")
    mae = r.get("overall_mae")
    c4.metric("Overall MAE",       f"{mae} min" if mae else "—")
    w2 = r.get("within_2_pct")
    c5.metric("Within ±2 min",     f"{w2}%" if w2 else "—")
    w5 = r.get("within_5_pct")
    c6.metric("Within ±5 min",     f"{w5}%" if w5 else "—")

    st.divider()

    # Model accuracy comparison
    with st.expander("Model Accuracy — Baseline vs Live", expanded=True):
        ev = load_model_eval()
        live = fetch_live_accuracy(days)
        lr = live.iloc[0] if not live.empty else {}

        col_b, col_l, col_d = st.columns(3)

        with col_b:
            st.markdown("**Baseline** *(train_base.py on CSV)*")
            st.metric("MAE",          f"{ev['mae_min']:.3f} min"        if ev.get("mae_min")         else "—")
            st.metric("RMSE",         f"{ev['rmse_min']:.3f} min"       if ev.get("rmse_min")        else "—")
            st.metric("R²",           f"{ev['r2']:.4f}"                 if ev.get("r2") is not None  else "—")
            st.metric("Within ±2 min",f"{ev['within_2min_pct']:.1f}%"   if ev.get("within_2min_pct") else "—")
            st.metric("Within ±5 min",f"{ev['within_5min_pct']:.1f}%"   if ev.get("within_5min_pct") else "—")
            if ev.get("trained_trips"):
                st.caption(f"Trained on {int(ev['trained_trips']):,} trips · tested on {int(ev.get('test_trips',0)):,}")

        with col_l:
            st.markdown("**Live** *(from pipeline predictions)*")
            live_mae  = lr.get("mae")
            live_rmse = lr.get("rmse")
            live_r2   = lr.get("r2")
            live_w2   = lr.get("within_2_pct")
            live_w5   = lr.get("within_5_pct")
            live_n    = lr.get("evaluated")
            base_mae  = ev.get("mae_min")
            delta_mae = f"{float(live_mae) - float(base_mae):+.3f} min" if live_mae and base_mae else None
            st.metric("MAE",           f"{live_mae} min"  if live_mae  else "—", delta=delta_mae,
                      delta_color="inverse")
            st.metric("RMSE",          f"{live_rmse} min" if live_rmse else "—")
            st.metric("R²",            f"{live_r2}"       if live_r2 is not None else "—")
            st.metric("Within ±2 min", f"{live_w2}%"      if live_w2   else "—")
            st.metric("Within ±5 min", f"{live_w5}%"      if live_w5   else "—")
            if live_n:
                st.caption(f"Evaluated on {int(live_n):,} unique trips with known actual delay")

        with col_d:
            st.markdown("**How to read this**")
            st.markdown("""
- **MAE** — average absolute error in minutes. Lower = better.
- **RMSE** — penalises large errors more. Lower = better.
- **R²** — how much variance the model explains. Closer to 1 = better.
- **Within ±2 / ±5 min** — % of predictions within that error band. Higher = better.
- **Delta** on Live MAE shows improvement vs baseline.
  🟢 negative = model got better with live data.
  🔴 positive = model degraded (could mean new patterns not seen in CSV).
- **Scatter plot** — points on the dashed line = perfect prediction.
  Points above = overpredicting delay. Points below = underpredicting.
""")

    summary = fetch_station_summary()
    mae_by_station = fetch_mae_by_station()

    if summary.empty:
        st.info("No trip data yet — start pipeline.py to collect live data.", icon="ℹ️")
    else:
        col_l, col_r = st.columns(2)

        with col_l:
            st.subheader("Avg Delay per Station")
            fig = px.bar(
                summary.dropna(subset=["avg_delay_min"]).sort_values("avg_delay_min"),
                x="avg_delay_min", y="station_name", orientation="h",
                color="station_name",
                color_discrete_map=STATION_COLORS,
                labels={"avg_delay_min": "Avg Delay (min)", "station_name": ""},
            )
            fig.update_layout(showlegend=False, margin=dict(l=0,r=10,t=0,b=0), height=380)
            st.plotly_chart(fig, use_container_width=True)

        with col_r:
            st.subheader("Model MAE per Station")
            if not mae_by_station.empty:
                mae_by_station["name"] = mae_by_station["station_eva"].map(EVA_TO_NAME)
                fig2 = px.bar(
                    mae_by_station.sort_values("mae"),
                    x="mae", y="name", orientation="h",
                    color="name",
                    color_discrete_map=STATION_COLORS,
                    labels={"mae": "MAE (min)", "name": ""},
                    hover_data=["evaluated"],
                )
                fig2.update_layout(showlegend=False, margin=dict(l=0,r=10,t=0,b=0), height=380)
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("No evaluated predictions yet.")

        # MAE over time
        mae_df = fetch_mae_over_time(days)
        if not mae_df.empty:
            st.subheader("Model MAE over Time")
            fig3 = px.line(
                mae_df, x="hour", y="mae",
                markers=True,
                labels={"hour": "Time", "mae": "MAE (min)", "n": "Predictions"},
                color_discrete_sequence=[DB_RED],
                hover_data=["n"],
            )
            fig3.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=280)
            fig3.add_hline(y=2, line_dash="dot", line_color="gray",
                           annotation_text="±2 min target", annotation_position="top right")
            st.plotly_chart(fig3, use_container_width=True)

        # Model improvement log
        model_log = load_model_log()
        if not model_log.empty:
            st.subheader("Model Improvement over Time")
            fig_log = go.Figure()
            fig_log.add_trace(go.Scatter(
                x=model_log["ts"], y=model_log["mae_100"],
                name="MAE (last 100)", mode="lines+markers",
                line=dict(color="#0075BF"),
            ))
            fig_log.add_trace(go.Scatter(
                x=model_log["ts"], y=model_log["mae_500"],
                name="MAE (last 500)", mode="lines+markers",
                line=dict(color="#7B2FBE", dash="dot"),
            ))
            fig_log.add_hline(y=2, line_dash="dot", line_color="gray",
                              annotation_text="±2 min target")
            fig_log.update_layout(
                xaxis_title="Time", yaxis_title="MAE (min)",
                margin=dict(l=0, r=0, t=0, b=0), height=280,
                legend=dict(orientation="h", y=1.1),
            )
            fig_log.update_yaxes(rangemode="tozero")
            st.plotly_chart(fig_log, use_container_width=True)
            st.caption(f"{len(model_log)} checkpoints logged · every {SAVE_EVERY} trips learned")

        # Delay distribution
        dist = fetch_delay_dist()
        if not dist.empty:
            st.subheader("Network Delay Distribution")
            clipped = dist[(dist["delay_min"] >= -10) & (dist["delay_min"] <= 60)]
            fig4 = px.histogram(
                clipped, x="delay_min", nbins=70,
                color_discrete_sequence=[DB_RED],
                labels={"delay_min": "Delay (min)", "count": "Trips"},
            )
            fig4.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=260)
            fig4.add_vline(x=0, line_dash="dash", line_color="gray")
            st.plotly_chart(fig4, use_container_width=True)

        # Summary table
        st.subheader("Station Summary")
        st.dataframe(
            summary.rename(columns={
                "station_name":  "Station",
                "trips":         "Total Trips",
                "avg_delay_min": "Avg Delay (min)",
                "cancel_pct":    "Cancelled (%)",
                "with_actual":   "With Actual Arr",
            })[["Station", "Total Trips", "Avg Delay (min)", "Cancelled (%)", "With Actual Arr"]],
            use_container_width=True,
            hide_index=True,
        )

# ── Station detail view ───────────────────────────────────────────────────────

else:
    station_name = selected
    eva = STATIONS[station_name]

    st.header(f"📍 {station_name}")

    kpis = fetch_station_kpis(eva, days)
    r = kpis.iloc[0] if not kpis.empty else {}

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Trips",   f"{int(r.get('total_trips') or 0):,}")
    mae = r.get("mae")
    c2.metric("MAE",           f"{mae} min" if mae else "—")
    w2 = r.get("within_2_pct")
    c3.metric("Within ±2 min", f"{w2}%" if w2 else "—")
    w5 = r.get("within_5_pct")
    c4.metric("Within ±5 min", f"{w5}%" if w5 else "—")
    cp = r.get("cancel_pct")
    c5.metric("Cancelled",     f"{cp}%" if cp else "—")

    st.divider()

    # Next trains
    st.subheader("Next Trains")
    next_df = fetch_next_trains(eva)
    if next_df.empty:
        st.info("No upcoming trains found — pipeline may not be running yet.")
    else:
        def _delay_label(val):
            if val is None or (isinstance(val, float) and pd.isna(val)):
                return "❓ no prediction"
            val = float(val)
            if val <= 2:
                return f"🟢 +{val:.1f} min"
            if val <= 5:
                return f"🟡 +{val:.1f} min"
            return f"🔴 +{val:.1f} min"

        def _pred_arrival(row):
            if row["sched_arr_raw"] is None:
                return "—"
            delay = row["predicted_delay_min"]
            if delay is None or (isinstance(delay, float) and pd.isna(delay)):
                return "—"
            from datetime import timedelta
            import pandas as _pd
            arr = _pd.Timestamp(row["sched_arr_raw"])
            pred = arr + timedelta(minutes=float(delay))
            return pred.tz_convert("Europe/Berlin").strftime("%H:%M")

        display = next_df.copy()
        display["Predicted Delay"] = display["predicted_delay_min"].apply(_delay_label)
        display["Pred. Arrival"]   = display.apply(_pred_arrival, axis=1)
        display = display.rename(columns={
            "train_type": "Type",
            "line":       "Line",
            "direction":  "Direction",
            "sched_arr":  "Scheduled",
        })[["Type", "Line", "Direction", "Scheduled", "Pred. Arrival", "Predicted Delay"]]

        st.dataframe(display, use_container_width=True, hide_index=True)

    st.divider()

    col_l, col_r = st.columns(2)

    # Delay by hour
    with col_l:
        st.subheader("Avg Delay by Hour of Day")
        hour_df = fetch_delay_by_hour(eva)
        if hour_df.empty:
            st.info("No arrival data yet.")
        else:
            hour_df["avg_delay"] = hour_df["avg_delay"].astype(float)
            fig5 = px.bar(
                hour_df, x="hour", y="avg_delay",
                color="avg_delay",
                color_continuous_scale=["#00A878", "#F0A500", DB_RED],
                labels={"hour": "Hour of Day", "avg_delay": "Avg Delay (min)", "trips": "Trips"},
                hover_data=["trips"],
            )
            fig5.update_layout(coloraxis_showscale=False, margin=dict(l=0,r=0,t=0,b=0), height=320)
            fig5.add_hline(y=0, line_color="gray", line_dash="dot")
            st.plotly_chart(fig5, use_container_width=True)

    # Predicted vs Actual scatter
    with col_r:
        st.subheader("Predicted vs Actual Delay")
        preds = fetch_pred_vs_actual(eva)
        if preds.empty:
            st.info("No evaluated predictions yet.")
        else:
            lim = float(max(
                preds[["actual", "predicted"]].abs().max().max(), 1
            ))
            lim = min(lim, 60)
            fig6 = px.scatter(
                preds, x="actual", y="predicted",
                opacity=0.45,
                color_discrete_sequence=[DB_RED],
                labels={"actual": "Actual Delay (min)", "predicted": "Predicted Delay (min)"},
                hover_data=["predicted_at"],
            )
            fig6.add_shape(
                type="line", x0=-lim, y0=-lim, x1=lim, y1=lim,
                line=dict(color="gray", dash="dash", width=1),
            )
            fig6.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320)
            st.plotly_chart(fig6, use_container_width=True)

    # Delay distribution for station
    dist = fetch_delay_dist(eva)
    if not dist.empty:
        st.subheader("Delay Distribution")
        clipped = dist[(dist["delay_min"] >= -10) & (dist["delay_min"] <= 60)]
        fig7 = px.histogram(
            clipped, x="delay_min", nbins=50,
            color_discrete_sequence=[DB_RED],
            labels={"delay_min": "Delay (min)"},
        )
        fig7.add_vline(x=0, line_dash="dash", line_color="gray")
        fig7.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=240)
        st.plotly_chart(fig7, use_container_width=True)

    # Recent trips table
    st.subheader("Recent Trips")
    recent = fetch_recent_trips(eva)
    if recent.empty:
        st.info("No trips scraped yet for this station.")
    else:
        st.dataframe(
            recent.rename(columns={
                "train_type": "Type",
                "line":       "Line",
                "direction":  "Direction",
                "sched_arr":  "Scheduled Arr",
                "actual_arr": "Actual Arr",
                "delay_min":  "Delay (min)",
                "cancelled":  "Cancelled",
            }),
            use_container_width=True,
            hide_index=True,
            height=380,
        )
