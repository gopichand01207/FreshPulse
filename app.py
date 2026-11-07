"""FreshPulse Streamlit dashboard for demand, leakage, and margin scenarios."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from freshpulse.data import generate_observations, save_observations
from freshpulse.modeling import train_and_evaluate
from freshpulse.analytics import analyze_leakage
from freshpulse.simulation import recommend_inventory_cover, simulate_policy


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "generated" / "synthetic_observations.csv"
METRICS_PATH = ROOT / "models" / "metrics.json"
PREDICTIONS_PATH = ROOT / "models" / "holdout_predictions.csv"


def ensure_pipeline() -> None:
    if DATA_PATH.exists() and METRICS_PATH.exists() and PREDICTIONS_PATH.exists():
        return
    with st.spinner("Generating synthetic data and fitting the demo models..."):
        observations = generate_observations()
        save_observations(observations, DATA_PATH)
        leakage, leakage_summary = analyze_leakage(
            observations, ROOT / "sql" / "leakage_analysis.sql"
        )
        leakage.to_csv(ROOT / "models" / "leakage_by_sku_store.csv", index=False)
        _, forecast_metrics, _ = train_and_evaluate(observations, ROOT / "models")
        report = {
            "data": {
                "source": "synthetic",
                "observations": len(observations),
                "stores": int(observations["store_id"].nunique()),
                "skus": int(observations["sku_id"].nunique()),
                "interval_minutes": 15,
            },
            "forecast": forecast_metrics,
            "leakage": leakage_summary,
        }
        METRICS_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")


@st.cache_data
def load_outputs() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    return (
        pd.read_csv(DATA_PATH, parse_dates=["timestamp"]),
        pd.read_csv(PREDICTIONS_PATH, parse_dates=["timestamp"]),
        json.loads(METRICS_PATH.read_text(encoding="utf-8")),
    )


def main() -> None:
    st.set_page_config(page_title="FreshPulse | Demand & Margin", layout="wide")
    st.title("FreshPulse")
    st.caption(
        "Quick-commerce demand, perishable leakage, and contribution-margin simulator"
    )
    st.info(
        "Demo uses generated synthetic data only. Results are illustrative and are "
        "not evidence of outcomes at a real company."
    )
    ensure_pipeline()
    observations, predictions, report = load_outputs()
    leakage = report["leakage"]
    forecast_metrics = report["forecast"]

    st.subheader("Observed synthetic sample")
    first, second, third, fourth = st.columns(4)
    first.metric("15-minute observations", f"{len(observations):,}")
    second.metric("Dark stores", f"{observations['store_id'].nunique():,}")
    third.metric("Perishable SKUs", f"{observations['sku_id'].nunique():,}")
    fourth.metric(
        "Sample leakage",
        f"₹{leakage['combined_leakage']:,.0f}",
        help="Synthetic stockout + spoilage cost in the generated sample window.",
    )
    st.caption(
        f"Sample: {leakage['sample_days']:.0f} days | "
        f"Annualized run-rate projection: ₹{leakage['annualized_leakage']:,.0f} "
        "(simple scale-up, not a measured annual result)"
    )

    left, right = st.columns(2)
    with left:
        loss_frame = pd.DataFrame(
            {
                "Loss type": ["Stockout cost", "Spoilage cost"],
                "₹": [leakage["stockout_cost"], leakage["spoilage_cost"]],
            }
        )
        st.plotly_chart(
            px.bar(loss_frame, x="Loss type", y="₹", color="Loss type"),
            use_container_width=True,
        )
    with right:
        hourly = (
            observations.assign(hour=observations["timestamp"].dt.hour)
            .groupby("hour", as_index=False)[["demand_units", "waste_units", "stockout_units"]]
            .sum()
            .melt(id_vars="hour", var_name="Measure", value_name="Units")
        )
        st.plotly_chart(
            px.line(hourly, x="hour", y="Units", color="Measure"),
            use_container_width=True,
        )

    st.subheader("Chronological demand model evaluation")
    col1, col2, col3 = st.columns(3)
    col1.metric("XGBoost holdout WAPE", f"{forecast_metrics['xgboost_wape']:.1%}")
    col2.metric(
        "Moving-average holdout WAPE",
        f"{forecast_metrics['moving_average_wape']:.1%}",
    )
    col3.metric("Fitted price elasticity", f"{forecast_metrics['elasticity']:.2f}")
    st.caption(
        f"Train through {forecast_metrics['train_end'][:10]}; "
        f"holdout {forecast_metrics['test_start'][:10]} to "
        f"{forecast_metrics['test_end'][:10]}. WAPE = total absolute error / total demand."
    )
    prediction_plot = predictions.copy()
    prediction_plot["timestamp"] = prediction_plot["timestamp"].dt.floor("h")
    prediction_plot = prediction_plot.groupby("timestamp", as_index=False)[
        ["demand_units", "forecast_units"]
    ].sum()
    st.plotly_chart(
        px.line(
            prediction_plot,
            x="timestamp",
            y=["demand_units", "forecast_units"],
            labels={"value": "Units", "variable": "Series"},
        ),
        use_container_width=True,
    )

    st.subheader("Dark-store CM2 scenario simulator")
    price_change = st.slider(
        "Price change (%)", min_value=-20, max_value=20, value=0, step=1
    )
    forecasts = (
        predictions.groupby(["store_id", "sku_id", "category"], as_index=False)
        .agg(
            forecast_units=("forecast_units", "mean"),
            unit_price=("unit_price", "mean"),
            base_price=("base_price", "mean"),
            unit_cost=("unit_cost", "mean"),
        )
    )
    elasticity = float(forecast_metrics["elasticity"])
    recommended_cover = recommend_inventory_cover(
        forecasts, elasticity, price_change / 100
    )
    cover = st.slider(
        "Inventory cover multiplier",
        min_value=0.5,
        max_value=2.0,
        value=round(recommended_cover, 2),
        step=0.05,
        help="Stock available relative to model forecast for the holdout window.",
    )
    scenario = simulate_policy(
        forecasts, elasticity, price_change / 100, cover
    )
    baseline_col, scenario_col, waste_col, stockout_col = st.columns(4)
    baseline_col.metric("Baseline CM2", f"₹{scenario['baseline_cm2']:,.0f}")
    scenario_col.metric(
        "Scenario CM2",
        f"₹{scenario['cm2']:,.0f}",
        f"{scenario['cm2_change_pct']:+.1f}%",
    )
    waste_col.metric(
        "Waste reduction vs baseline",
        f"{scenario['waste_reduction_pct']:+.1f}%",
    )
    stockout_col.metric("Unserved demand", f"{scenario['stockout_units']:,.1f} units")
    st.caption(
        f"SciPy CM2-maximizing cover at this price: {recommended_cover:.2f}x. "
        "CM2 is simulated revenue less product cost on stocked units, waste cost, "
        "and ₹5 per sold unit fulfillment cost."
    )

    with st.expander("Top synthetic SKU/store leakage"):
        detail = (
            observations.assign(
                stockout_cost=observations["stockout_units"] * observations["unit_cost"],
                spoilage_cost=observations["waste_units"] * observations["unit_cost"],
            )
            .groupby(["store_id", "sku_id", "category"], as_index=False)[
                ["stockout_cost", "spoilage_cost"]
            ]
            .sum()
        )
        detail["total_leakage"] = detail["stockout_cost"] + detail["spoilage_cost"]
        st.dataframe(
            detail.sort_values("total_leakage", ascending=False).head(20),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
