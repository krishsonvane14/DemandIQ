from src.insights import generate_insights

test_metrics = {
    "top_category": "health_beauty",
    "top_state": "SP",
    "revenue_trend_direction": "increasing",
    "forecast_next_30_days": 845000.00,
    "avg_order_value": 137.53,
    "total_revenue": 13315828.19,
}

result = generate_insights(test_metrics)

print(type(result))
print(result)
print(result["insight_paragraph"])
print(result["recommendations"])