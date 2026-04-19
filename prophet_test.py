import pandas as pd
from src.forecasting import run_forecast_pipeline, plot_forecast

df = pd.read_csv("data/cleaned/fact_table.csv", parse_dates=["order_date"])
history_df, forecast_df = run_forecast_pipeline(df, periods=30)

print(history_df.head())
print(history_df.tail())
print(forecast_df[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail())

fig = plot_forecast(forecast_df, history_df)
print(type(fig))

print(history_df.tail(30))
print(history_df["y"].describe())
print(forecast_df[["ds", "yhat"]].tail(30))