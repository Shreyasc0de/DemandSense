"""DemandSense: daily demand forecasting and the inventory decision it implies.

The app loads the model and the backtest that scripts/train.py produced. It does
not fit anything on page load, because an app that refits is not the model whose
accuracy is published.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import plotly.graph_objs as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from demandsense.inventory import reorder_by, stockout_window, unmet_demand  # noqa: E402

MODEL_PATH = ROOT / "models" / "forecaster.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
SERIES_PATH = ROOT / "reports" / "series.csv"
BASELINES = ("seasonal_naive_7d", "last_value", "mean")

st.set_page_config(page_title="DemandSense", layout="wide")
st.title("DemandSense")
st.caption(
    "Daily demand forecasting on the UCI Online Retail II transactions, joined to "
    "Open-Meteo weather for London. Every figure comes from `reports/`, written by "
    "`scripts/train.py`."
)


@st.cache_resource
def load_artifact() -> dict:
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_reports() -> tuple[dict, pd.DataFrame]:
    return json.loads(METRICS_PATH.read_text()), pd.read_csv(SERIES_PATH, parse_dates=["ds"])


missing = [p.name for p in (MODEL_PATH, METRICS_PATH, SERIES_PATH) if not p.exists()]
if missing:
    st.error(
        f"Missing {', '.join(missing)}. Run `python scripts/fetch_weather.py` and then "
        "`python scripts/train.py`. Training needs `data/online_retail_II.xlsx`."
    )
    st.stop()

artifact = load_artifact()
metrics, series = load_reports()
engine, regressors = artifact["engine"], list(artifact["regressors"])
facts = metrics["series"]

# ------------------------------------------------------------------ sidebar

st.sidebar.header("Scenario")
horizon = st.sidebar.slider("Forecast horizon (days)", 7, 90, metrics["horizon_days"])
avg_temp = st.sidebar.slider(
    "Average temperature (°C)",
    float(round(facts["min_temp_c"] - 2)), float(round(facts["max_temp_c"] + 2)),
    float(round(facts["mean_temp_c"], 1)), step=0.5,
)
avg_precip = st.sidebar.slider("Average daily rainfall (mm)", 0.0, 20.0, 2.0, step=0.5)

st.sidebar.markdown("---")
st.sidebar.header("Inventory")
current_stock = st.sidebar.number_input(
    "Units on hand", min_value=0,
    value=int(facts["mean_daily_units"] * 20), step=1_000,
)
lead_time = st.sidebar.number_input("Supplier lead time (days)", min_value=1, value=14, step=1)

# ------------------------------------------------------------------ forecast

last_day = series["ds"].max()
future = pd.DataFrame(
    {"ds": pd.date_range(last_day + pd.Timedelta(days=1), periods=horizon, freq="D")}
)
future["temp_c"] = float(avg_temp)
future["precip_mm"] = float(avg_precip)
forecast = engine.predict(future[["ds", *regressors]]).frame()

# ---------------------------------------------------------- how good is this

best = metrics["best_by_mae"]
summary = metrics["engines"][best]
baseline_name = metrics["best_baseline"]
baseline = metrics["engines"][baseline_name]

st.subheader("How accurate is this forecast")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Held-out MAE", f"{summary['mae']:,.0f} units",
          help=f"Engine: {best}, averaged over {summary['n_folds']} rolling-origin folds.")
c2.metric("Fold-to-fold SD", f"{summary['sd_mae_across_folds']:,.0f}",
          help="Spread of MAE across folds. If a model's lead over another is smaller than this, it is noise.")
c3.metric(f"vs {baseline_name.replace('_', ' ')}",
          f"{metrics['best_vs_baseline_mae_pct']:+.1f}%",
          help="MAE improvement over the best naive baseline. Negative means the baseline wins.")
c4.metric("80% interval coverage", f"{summary['coverage_80']:.0%}",
          delta=f"{summary['coverage_80'] - 0.80:+.0%} vs nominal",
          help="Share of held-out days inside the 80% interval. Below 80% means it is too narrow.")

gain_key = f"{best.split('_')[0]}_weather_mae_gain_pct"
if gain_key in metrics:
    gain = metrics[gain_key]
    noisy = metrics.get(f"{best.split('_')[0]}_weather_gain_within_fold_noise", False)
    verdict = "is smaller than the fold-to-fold spread, so it is not a real effect" if noisy \
        else "exceeds the fold-to-fold spread"
    st.caption(
        f"Dropping the weather regressors changes MAE by {gain:+.1f}%. That {verdict}. "
        "The temperature slider moves the forecast, but the backtest is what says whether "
        "the movement means anything."
    )

if summary["coverage_80"] < 0.70:
    st.warning(
        f"The 80% interval contained only {summary['coverage_80']:.0%} of held-out days, so the "
        "range below is narrower than it should be. Treat the earliest stockout date as "
        "optimistic rather than worst-case."
    )

# -------------------------------------------------------------- the decision

st.subheader("Inventory decision")
window = stockout_window(forecast, current_stock)

if not window.possible:
    st.success(
        f"No stockout within {horizon} days. Even the fast-demand path leaves stock on hand "
        f"through {window.horizon_end:%d %B %Y}."
    )
else:
    order_by = reorder_by(window, lead_time)
    if window.certain:
        st.error(
            f"Stockout expected **{window.expected:%d %B %Y}**. Every demand path inside the 80% "
            f"interval runs out between **{window.earliest:%d %B}** and **{window.latest:%d %B %Y}**."
        )
    else:
        tail = (f"the median path runs out on **{window.expected:%d %B %Y}**."
                if window.expected else
                f"the median path holds through {window.horizon_end:%d %B %Y}.")
        st.warning(
            f"A stockout is possible but not certain. The fast-demand path runs out on "
            f"**{window.earliest:%d %B %Y}**, and {tail}"
        )
    st.info(f"Order by **{order_by:%d %B %Y}** to cover the earliest plausible stockout "
            f"(lead time {lead_time} days).")
    st.caption(
        f"Demand beyond available stock over the horizon: **{unmet_demand(forecast, current_stock):,.0f} units**. "
        "This is an upper bound on lost sales, not an estimate. It assumes every customer who "
        "cannot buy is lost, when some wait and some substitute, and nothing here measures how many."
    )

# ------------------------------------------------------------------- charts

st.subheader("History and forecast")
fig = go.Figure()
fig.add_trace(go.Scatter(x=series["ds"], y=series["y"], mode="lines", name="Actual",
                         line=dict(color="#2B2B2B", width=0.9)))
fig.add_trace(go.Scatter(
    x=list(forecast["ds"]) + list(forecast["ds"][::-1]),
    y=list(forecast["upper"]) + list(forecast["lower"][::-1]),
    fill="toself", fillcolor="rgba(11,110,79,0.18)", line=dict(color="rgba(0,0,0,0)"),
    name="80% interval", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast",
                         line=dict(color="#0B6E4F", width=2.2)))
fig.update_layout(height=460, xaxis_title="", yaxis_title="Units sold per day",
                  legend=dict(orientation="h", y=1.08), margin=dict(t=30))
st.plotly_chart(fig, use_container_width=True)

with st.expander("Backtest detail and data notes"):
    st.write(
        f"{facts['n_days']} days, {facts['first_day']} to {facts['last_day']}. "
        f"{facts['n_zero_days']} days with no sales, of which "
        f"{facts['n_saturdays'] - facts['n_saturdays_with_sales']} are Saturdays: this retailer "
        "does not invoice on Saturdays, so those days are kept as zeros rather than dropped. "
        f"Rolling origin: {metrics['initial_days']}-day initial window, "
        f"{metrics['horizon_days']}-day horizon, {metrics['step_days']}-day step."
    )
    st.dataframe(pd.DataFrame(metrics["engines"]).T.sort_values("mae"), use_container_width=True)
    st.image(str(ROOT / "reports" / "figures" / "weekly_shape.png"))
    st.image(str(ROOT / "reports" / "figures" / "backtest.png"))
    st.caption(
        f"Cleaning: kept {metrics['cleaning']['rows_kept']:,} of "
        f"{metrics['cleaning']['rows_raw']:,} transaction rows. "
        f"Sources: {metrics['source']['transactions']}; {metrics['source']['weather']}."
    )

st.download_button("Download this forecast (CSV)",
                   forecast.to_csv(index=False).encode("utf-8"),
                   file_name="demand_forecast.csv", mime="text/csv")
