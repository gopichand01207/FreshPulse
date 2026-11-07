"""Generate data, run leakage SQL, train models, and save reproducible metrics."""

from __future__ import annotations

import json
from pathlib import Path

from freshpulse.analytics import analyze_leakage
from freshpulse.data import generate_observations, save_observations
from freshpulse.modeling import train_and_evaluate


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "generated"
MODEL_DIR = ROOT / "models"


def run() -> dict[str, object]:
    observations = generate_observations()
    dataset_path = save_observations(
        observations, DATA_DIR / "synthetic_observations.csv"
    )
    leakage_rows, leakage_summary = analyze_leakage(
        observations, ROOT / "sql" / "leakage_analysis.sql"
    )
    leakage_rows.to_csv(MODEL_DIR / "leakage_by_sku_store.csv", index=False)
    _, model_metrics, _ = train_and_evaluate(observations, MODEL_DIR)

    report: dict[str, object] = {
        "data": {
            "source": "synthetic",
            "dataset_path": str(dataset_path.relative_to(ROOT)),
            "observations": len(observations),
            "stores": int(observations["store_id"].nunique()),
            "skus": int(observations["sku_id"].nunique()),
            "interval_minutes": 15,
        },
        "forecast": model_metrics,
        "leakage": leakage_summary,
    }
    (MODEL_DIR / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> None:
    report = run()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
