# FreshPulse: Quick-Commerce Demand & Margin Engine

FreshPulse is an end-to-end portfolio project for perishable demand forecasting,
stockout/spoilage analysis, and dark-store contribution-margin scenarios. It ships
with a deterministic **synthetic** dataset generator; it does not contain real
customer, order, or company data.

> **Evidence note:** Outputs are simulated and depend on the generated sample.
> They are not proof of actual operating results. Use the measured metrics printed
> by this project, and describe them as synthetic/demo results on a resume.

## Quick start

Python 3.10+ is recommended.

```bash
cd FreshPulse
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
streamlit run app.py
```

The pipeline generates 55,296 15-minute store/SKU observations by default
(6 stores × 12 SKUs × 8 days × 96 intervals), runs the SQL leakage analysis,
fits the forecast and elasticity models, and writes reproducible output files.
The dashboard can bootstrap these artifacts on its first launch.

Run the test suite from this directory:

```bash
python -m pytest -q
```

## Project layout

```text
FreshPulse/
├── app.py                              # Streamlit demand and CM2 simulator
├── run_pipeline.py                     # One-command data/model/SQL pipeline
├── freshpulse/
│   ├── data.py                         # Deterministic synthetic data generation
│   ├── modeling.py                     # XGBoost forecast + SciPy elasticity
│   ├── analytics.py                    # SQLite CTE execution and cost summary
│   └── simulation.py                   # Price/inventory and CM2 scenarios
├── sql/leakage_analysis.sql            # Stockout vs. spoilage CTE
├── tests/test_freshpulse.py             # Generator, SQL, model, simulator tests
├── data/generated/                     # Generated source observations
└── models/                              # Predictions, model, metrics, leakage
```

## Method

### Data and leakage

Each row is a store × SKU × 15-minute demand/inventory observation. The synthetic
generator includes meal-time and weekday patterns, weather, promotion, randomized
prices, SKU shelf life, available inventory, fulfilled sales, stockout units, and
waste units. It is intentionally not an individual-customer event log.

`sql/leakage_analysis.sql` groups stockout and spoilage units by store and SKU,
then prices each at unit cost. The Python runner executes that CTE with SQLite,
reports the sample-window costs, and separately calculates a simple 365-day
annualized run-rate projection. That projection assumes the short sample is
representative and is not an observed annual P&L.

### Forecast and price response

`freshpulse/modeling.py` builds per-store/SKU demand lags (15 minutes, 1 hour,
and 1 day), rolling demand, time-of-day, weather, promotion, price, and shelf-life
features. XGBoost trains on earlier timestamps and is evaluated on the final
two calendar days; the report compares holdout WAPE with a four-interval moving
average baseline. The SciPy robust least-squares fit controls for SKU, hour,
weather, and promotion when estimating log demand response to log price.

### Margin simulator

The dashboard lets a user change price and inventory cover. The simulator applies
the estimated price elasticity to forecast demand, caps sold units at inventory,
and estimates revenue, product cost, waste cost, fulfillment cost, stockout units,
and CM2. A bounded SciPy scalar optimizer searches for an inventory-cover
multiplier that maximizes the simulated CM2 on the holdout forecast panel.
It is a scenario model, not a causal test or a deployment-ready replenishment
recommendation.

## Generated outputs

| Path | Description |
|---|---|
| `data/generated/synthetic_observations.csv` | Synthetic interval observations |
| `models/demand_xgboost.json` | Trained XGBoost model |
| `models/holdout_predictions.csv` | Chronological holdout and forecasts |
| `models/leakage_by_sku_store.csv` | SQL CTE results |
| `models/metrics.json` | Data counts, WAPE, elasticity, and cost summary |

These are reproducible build artifacts and can be regenerated with `python
run_pipeline.py`. For a verification walkthrough, show the pipeline command,
the metrics JSON, the SQL CTE, model train/test dates, and dashboard controls.

## Resume bullet alignment

- **Hyperlocal demand/inventory analysis:** `data.py`, `leakage_analysis.sql`,
  and `analytics.py`; describe the records as synthetic 15-minute observations,
  not real company order logs.
- **Demand forecast and price elasticity:** `modeling.py`; quote the WAPE and
  moving-average WAPE emitted by the run, not a preselected target metric.
- **CM2 / waste simulator:** `simulation.py` and `app.py`; report scenario deltas
  as model outputs, not realized savings.

Do not present the resume's ₹1.2 Cr, 8.4% WAPE, 31%, or 4.2% figures as verified
by this demo unless independently reproduced on appropriately sourced data.
