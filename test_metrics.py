import pandas as pd
from src.metrics import (
    total_revenue,
    total_orders,
    avg_order_value,
    revenue_by_category,
    revenue_by_state,
    revenue_over_time,
)

df = pd.read_csv("data/cleaned/fact_table.csv", parse_dates=["order_date"])

print("=== Scalars ===")
print("total_revenue:", total_revenue(df))
print("total_orders:", total_orders(df))
print("avg_order_value:", avg_order_value(df))

print("\n=== By category ===")
cat = revenue_by_category(df)
print(cat.head())
print(cat.columns.tolist(), cat.shape)

print("\n=== By state ===")
state = revenue_by_state(df)
print(state.head())
print(state.columns.tolist(), state.shape)

print("\n=== Over time ===")
ts = revenue_over_time(df)
print(ts.head())
print(ts.columns.tolist(), ts.shape)
print(ts.dtypes)