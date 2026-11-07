"""Time-ordered XGBoost demand forecast and SciPy elasticity estimate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from xgboost import XGBRegressor


FEATURES = [
    "lag_1",
    "lag_4",
    "lag_96",
    "rolling_4",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "temperature_c",
    "rain_mm",
    "promo",
    "unit_price",
    "shelf_life_hours",
]


def add_forecast_features(observations: pd.DataFrame) -> pd.DataFrame:
    """Create lags within each store/SKU, preventing cross-series leakage."""
    frame = observations.sort_values(
        ["store_id", "sku_id", "timestamp"]
    ).copy()
    series_keys = ["store_id", "sku_id"]
    grouped = frame.groupby(series_keys, sort=False)["demand_units"]
    frame["lag_1"] = grouped.shift(1)
    frame["lag_4"] = grouped.shift(4)
    frame["lag_96"] = grouped.shift(96)
    frame["rolling_4"] = (
        frame.groupby(series_keys, sort=False)["demand_units"]
        .transform(lambda values: values.shift(1).rolling(4, min_periods=1).mean())
    )
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)
    return frame.dropna(subset=FEATURES).reset_index(drop=True)


def wape(actual: np.ndarray | pd.Series, predicted: np.ndarray | pd.Series) -> float:
    """Weighted absolute percentage error as a fraction."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    denominator = np.abs(actual_values).sum()
    if denominator == 0:
        raise ValueError("WAPE is undefined when all actual demand is zero.")
    return float(np.abs(actual_values - predicted_values).sum() / denominator)


def fit_elasticity(observations: pd.DataFrame) -> float:
    """Fit a controlled log-demand response to log price using SciPy."""
    positive = observations.loc[observations["demand_units"] > 0].copy()
    if len(positive) < 20:
        raise ValueError("At least 20 positive-demand observations are required.")

    sku_codes = pd.Categorical(positive["sku_id"]).codes
    sku_indicators = np.eye(sku_codes.max() + 1)[sku_codes]
    hour = positive["hour"].to_numpy(dtype=float)
    price_delta = np.log(
        positive["unit_price"].to_numpy(dtype=float)
        / positive["base_price"].to_numpy(dtype=float)
    )
    design = np.column_stack(
        [
            sku_indicators,
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            positive["temperature_c"].to_numpy(dtype=float) / 10,
            positive["rain_mm"].to_numpy(dtype=float),
            positive["promo"].to_numpy(dtype=float),
            price_delta,
        ]
    )
    target = np.log(positive["demand_units"].to_numpy(dtype=float))

    def residual(parameters: np.ndarray) -> np.ndarray:
        return design @ parameters - target

    initial = np.zeros(design.shape[1])
    initial[: sku_indicators.shape[1]] = np.log(
        positive.groupby("sku_id")["demand_units"].mean().clip(lower=0.1).to_numpy()
    )
    result = least_squares(residual, initial, loss="soft_l1", f_scale=0.4)
    return float(result.x[-1])


def train_and_evaluate(
    observations: pd.DataFrame,
    model_dir: str | Path,
) -> tuple[pd.DataFrame, dict[str, float], XGBRegressor]:
    """Train on earlier days and report holdout WAPE vs a lag baseline."""
    features = add_forecast_features(observations)
    holdout_start = features["timestamp"].max().normalize() - pd.Timedelta(days=1)
    train = features.loc[features["timestamp"] < holdout_start]
    test = features.loc[features["timestamp"] >= holdout_start].copy()
    if train.empty or test.empty:
        raise ValueError("Not enough dates for a chronological train/test split.")

    model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=220,
        max_depth=5,
        learning_rate=0.045,
        subsample=0.85,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        n_jobs=4,
        random_state=2026,
    )
    model.fit(train[FEATURES], train["demand_units"])
    test["forecast_units"] = np.maximum(model.predict(test[FEATURES]), 0)
    metrics = {
        "observations": float(len(observations)),
        "train_rows": float(len(train)),
        "test_rows": float(len(test)),
        "xgboost_wape": wape(test["demand_units"], test["forecast_units"]),
        "moving_average_wape": wape(test["demand_units"], test["lag_4"]),
        "elasticity": fit_elasticity(observations),
        "train_start": str(train["timestamp"].min()),
        "train_end": str(train["timestamp"].max()),
        "test_start": str(test["timestamp"].min()),
        "test_end": str(test["timestamp"].max()),
    }

    output_dir = Path(model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(output_dir / "demand_xgboost.json")
    test[
        [
            "timestamp",
            "store_id",
            "sku_id",
            "category",
            "unit_price",
            "base_price",
            "unit_cost",
            "inventory_units",
            "demand_units",
            "sold_units",
            "stockout_units",
            "waste_units",
            "forecast_units",
        ]
    ].to_csv(output_dir / "holdout_predictions.csv", index=False)
    return test, metrics, model
