"""Price and inventory policy simulator using fitted demand elasticity."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar


def _scenario(
    forecast: pd.DataFrame,
    price_change: float,
    inventory_cover: float,
    elasticity: float,
) -> dict[str, float]:
    adjusted_price = forecast["unit_price"] * (1 + price_change)
    demand = forecast["forecast_units"] * (
        adjusted_price / forecast["base_price"]
    ).clip(lower=0.2) ** elasticity
    inventory = forecast["forecast_units"] * inventory_cover
    sold = np.minimum(demand, inventory)
    waste = np.maximum(inventory - demand, 0)
    stockout = np.maximum(demand - inventory, 0)
    revenue = float((sold * adjusted_price).sum())
    contribution = float(
        (sold * (adjusted_price - forecast["unit_cost"])).sum()
        - (waste * forecast["unit_cost"]).sum()
        - sold.sum() * 5.0
    )
    return {
        "revenue": revenue,
        "cm2": contribution,
        "waste_units": float(waste.sum()),
        "stockout_units": float(stockout.sum()),
        "demand_units": float(demand.sum()),
    }


def simulate_policy(
    forecasts: pd.DataFrame,
    elasticity: float,
    price_change: float = 0.0,
    inventory_cover: float = 1.0,
) -> dict[str, float]:
    """Compare policy to neutral price/forecast-cover scenario."""
    baseline = _scenario(forecasts, 0.0, 1.0, elasticity)
    scenario = _scenario(forecasts, price_change, inventory_cover, elasticity)
    scenario["baseline_cm2"] = baseline["cm2"]
    scenario["cm2_change_pct"] = (
        (scenario["cm2"] / baseline["cm2"] - 1) * 100
        if baseline["cm2"] else 0.0
    )
    scenario["waste_reduction_pct"] = (
        (1 - scenario["waste_units"] / baseline["waste_units"]) * 100
        if baseline["waste_units"] else 0.0
    )
    return scenario


def recommend_inventory_cover(
    forecasts: pd.DataFrame,
    elasticity: float,
    price_change: float = 0.0,
) -> float:
    """Find a bounded CM2-maximizing inventory multiplier with SciPy."""
    result = minimize_scalar(
        lambda cover: -_scenario(
            forecasts, price_change, float(cover), elasticity
        )["cm2"],
        bounds=(0.5, 2.0),
        method="bounded",
        options={"xatol": 0.01},
    )
    if not result.success:
        raise RuntimeError(f"Inventory optimization failed: {result.message}")
    return float(result.x)
