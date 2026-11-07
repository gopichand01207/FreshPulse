"""Deterministic generator for synthetic 15-minute store/SKU observations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


STORE_COUNT = 6
SKU_COUNT = 12
SIMULATION_DAYS = 8
INTERVALS_PER_DAY = 96
SEED = 2026


def generate_observations(
    days: int = SIMULATION_DAYS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Create interval-level demand and inventory data, not real customer data."""
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range(
        "2026-01-05", periods=days * INTERVALS_PER_DAY, freq="15min"
    )
    stores = [f"DS-{number:02d}" for number in range(1, STORE_COUNT + 1)]
    skus = [f"SKU-{number:03d}" for number in range(1, SKU_COUNT + 1)]
    categories = ["Dairy", "Produce", "Bakery", "Ready-to-eat"]
    shelf_lives = np.array([24, 36, 18, 12, 48, 30, 20, 42, 16, 28, 14, 40])
    base_prices = np.array([42, 58, 35, 75, 110, 28, 64, 49, 88, 32, 52, 95])
    unit_costs = base_prices * rng.uniform(0.54, 0.72, size=SKU_COUNT)
    elasticity = rng.uniform(-1.7, -0.55, size=SKU_COUNT)
    store_scale = rng.uniform(0.78, 1.35, size=STORE_COUNT)
    sku_scale = rng.uniform(0.45, 1.25, size=SKU_COUNT)

    frames: list[pd.DataFrame] = []
    for store_index, store_id in enumerate(stores):
        for sku_index, sku_id in enumerate(skus):
            hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60
            weekday = timestamps.dayofweek.to_numpy()
            meal_peaks = (
                0.55 * np.exp(-((hour - 9) / 2.2) ** 2)
                + 0.85 * np.exp(-((hour - 13) / 2.8) ** 2)
                + 1.05 * np.exp(-((hour - 19) / 2.5) ** 2)
            )
            weekly_factor = np.where(weekday >= 5, 1.16, 1.0)
            temperature = 24 + 6 * np.sin((hour - 7) * np.pi / 12) + rng.normal(
                0, 1.7, len(timestamps)
            )
            rain = np.maximum(0, rng.gamma(0.7, 1.2, len(timestamps)) - 0.8)
            promo = rng.binomial(1, 0.09, len(timestamps))
            price = base_prices[sku_index] * (
                1 + rng.normal(0.0, 0.055, len(timestamps)) - 0.08 * promo
            )
            weather_factor = np.exp(
                0.012 * (temperature - 24) - 0.025 * rain
            )
            latent_mean = (
                (0.50 + meal_peaks)
                * store_scale[store_index]
                * sku_scale[sku_index]
                * weekly_factor
                * weather_factor
                * (1 + 0.18 * promo)
                * (price / base_prices[sku_index]) ** elasticity[sku_index]
            )
            demand = rng.poisson(np.maximum(latent_mean, 0.02))

            # Inventory availability and shelf-life loss vary by SKU and interval.
            inventory = np.maximum(
                0,
                np.rint(
                    latent_mean * rng.uniform(0.65, 1.9, len(timestamps))
                    + rng.poisson(0.65, len(timestamps))
                ),
            ).astype(int)
            sold = np.minimum(demand, inventory)
            stockout = demand - sold
            daily_spoilage_hazard = np.clip(0.28 / shelf_lives[sku_index], 0.004, 0.03)
            waste = rng.binomial(inventory - sold, daily_spoilage_hazard)

            frames.append(
                pd.DataFrame(
                    {
                        "timestamp": timestamps,
                        "store_id": store_id,
                        "sku_id": sku_id,
                        "category": categories[sku_index % len(categories)],
                        "shelf_life_hours": shelf_lives[sku_index],
                        "unit_price": price.round(2),
                        "base_price": base_prices[sku_index],
                        "unit_cost": round(unit_costs[sku_index], 2),
                        "promo": promo,
                        "temperature_c": temperature.round(1),
                        "rain_mm": rain.round(2),
                        "demand_units": demand,
                        "inventory_units": inventory,
                        "sold_units": sold,
                        "stockout_units": stockout,
                        "waste_units": waste,
                    }
                )
            )

    observations = pd.concat(frames, ignore_index=True)
    observations["day_of_week"] = observations["timestamp"].dt.dayofweek
    observations["hour"] = observations["timestamp"].dt.hour
    observations["date"] = observations["timestamp"].dt.strftime("%Y-%m-%d")
    return observations.sort_values(
        ["store_id", "sku_id", "timestamp"], ignore_index=True
    )


def save_observations(
    observations: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save generated observations as a portable CSV."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    observations.to_csv(output, index=False)
    return output
