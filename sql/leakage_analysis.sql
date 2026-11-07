WITH interval_costs AS (
    SELECT
        store_id,
        sku_id,
        category,
        SUM(stockout_units * unit_cost) AS stockout_cost,
        SUM(waste_units * unit_cost) AS spoilage_cost,
        SUM(stockout_units) AS stockout_units,
        SUM(waste_units) AS waste_units,
        COUNT(*) AS interval_count
    FROM demand_observations
    GROUP BY store_id, sku_id, category
),
ranked_leakage AS (
    SELECT
        *,
        stockout_cost + spoilage_cost AS total_leakage
    FROM interval_costs
)
SELECT
    store_id,
    sku_id,
    category,
    interval_count,
    stockout_units,
    waste_units,
    stockout_cost,
    spoilage_cost,
    total_leakage
FROM ranked_leakage
ORDER BY total_leakage DESC;
