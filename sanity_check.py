import pandas as pd

df = pd.read_csv("data/cleaned/fact_table.csv", parse_dates=["order_date"])

print("shape:", df.shape)
print("columns:", df.columns.tolist())
print("\nnulls:\n", df.isna().sum())
print("\nrevenue >= 0:", (df["revenue"] >= 0).all())
print("price >= 0:", (df["price"] >= 0).all())
print("quantity unique:", df["quantity"].unique()[:10])
print("unique categories:", df["category"].nunique())
print("unique states:", df["customer_state"].nunique())
print("date range:", df["order_date"].min(), "->", df["order_date"].max())

print("\nTop 10 categories by revenue:")
print(df.groupby("category", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False).head(10))

print("\nTop 10 states by revenue:")
print(df.groupby("customer_state", as_index=False)["revenue"].sum().sort_values("revenue", ascending=False).head(10))

print("\nDaily revenue sample:")
print(df.groupby("order_date", as_index=False)["revenue"].sum().head())

print("duplicate rows:", df.duplicated().sum())
print("duplicate item grain (order_id, order_item_id):", df.duplicated(subset=["order_id", "order_item_id"]).sum())