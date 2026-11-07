"""SQLite-backed 15-minute stockout/spoilage leakage analysis."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def analyze_leakage(
    observations: pd.DataFrame,
    sql_path: str | Path,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Execute the project's CTE against synthetic interval observations."""
    connection = sqlite3.connect(":memory:")
    try:
        observations.to_sql("demand_observations", connection, index=False)
        query = Path(sql_path).read_text(encoding="utf-8")
        result = pd.read_sql_query(query, connection)
    finally:
        connection.close()

    sample_days = max(
        (observations["timestamp"].max() - observations["timestamp"].min()).days + 1,
        1,
    )
    summary = {
        "sample_days": float(sample_days),
        "stockout_cost": float(result["stockout_cost"].sum()),
        "spoilage_cost": float(result["spoilage_cost"].sum()),
        "combined_leakage": float(result["total_leakage"].sum()),
        "annualized_leakage": float(
            result["total_leakage"].sum() * 365 / sample_days
        ),
    }
    return result, summary
