import numpy as np

from freshpulse.analytics import analyze_leakage
from freshpulse.data import generate_observations
from freshpulse.modeling import add_forecast_features, fit_elasticity, wape
from freshpulse.simulation import recommend_inventory_cover, simulate_policy


def test_generator_is_reproducible_and_has_15_minute_panel():
    first = generate_observations(days=2)
    second = generate_observations(days=2)

    assert len(first) == 2 * 96 * 6 * 12
    assert first.equals(second)
    intervals = first.groupby(["store_id", "sku_id"])["timestamp"].diff().dropna()
    assert (intervals == np.timedelta64(15, "m")).all()
    assert (first["sold_units"] <= first["demand_units"]).all()
    assert (first["sold_units"] <= first["inventory_units"]).all()
    assert (first["stockout_units"] == first["demand_units"] - first["sold_units"]).all()


def test_feature_lags_and_elasticity_are_finite():
    observations = generate_observations(days=3)
    featured = add_forecast_features(observations)

    assert {"lag_1", "lag_4", "lag_96", "rolling_4"}.issubset(featured.columns)
    assert np.isfinite(fit_elasticity(observations))


def test_wape_and_sql_leakage_summary_are_consistent():
    assert wape(np.array([2, 4]), np.array([1, 5])) == 1 / 3
    observations = generate_observations(days=2)
    rows, summary = analyze_leakage(
        observations, "sql/leakage_analysis.sql"
    )

    assert len(rows) == 6 * 12
    assert summary["combined_leakage"] == rows["total_leakage"].sum()
    assert summary["annualized_leakage"] > summary["combined_leakage"]


def test_margin_scenario_and_optimizer_are_bounded():
    observations = generate_observations(days=3)
    forecasts = (
        observations.groupby(["store_id", "sku_id", "category"], as_index=False)
        .agg(
            forecast_units=("demand_units", "mean"),
            unit_price=("unit_price", "mean"),
            base_price=("base_price", "mean"),
            unit_cost=("unit_cost", "mean"),
        )
    )
    elasticity = fit_elasticity(observations)
    cover = recommend_inventory_cover(forecasts, elasticity)
    result = simulate_policy(forecasts, elasticity, 0.05, cover)

    assert 0.5 <= cover <= 2.0
    assert np.isfinite(result["cm2"])
    assert result["demand_units"] > 0
