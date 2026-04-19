# src/forecasting.py
#
# Revenue forecasting for DemandIQ using Meta Prophet.
#
# Input:  cleaned fact table (one row per purchased item, grain = order_id +
#         order_item_id) produced by src.data_processing.build_fact_table().
#
# The pipeline:
#   1. prepare_daily_revenue(df)      – aggregate fact table to daily revenue
#                                       and produce a Prophet-compatible ds/y
#                                       time series with no date gaps.
#   2. generate_forecast(df, periods) – fit Prophet on the daily series and
#                                       return the full forecast DataFrame.
#   3. plot_forecast(forecast_df,     – return a Plotly Figure that shows
#                    history_df)         history + forecast + confidence band.
#   4. run_forecast_pipeline(df,      – convenience wrapper: calls (1) + (2)
#                            periods)    and returns (history_df, forecast_df).

import logging
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from prophet import Prophet

# Suppress Stan/cmdstanpy noise that Prophet prints to stdout.
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

# Module-level logger — used for pipeline diagnostics surfaced in the app.
logger = logging.getLogger(__name__)


# ── Step 1: Daily aggregation ──────────────────────────────────────────────────

def prepare_daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate the fact table to a continuous daily revenue time series.

    The fact table grain is one row per purchased item, so multiple rows
    may share the same ``order_date``. This function sums ``revenue`` by
    day and re-indexes the result over a complete date range so that days
    with no sales are represented as 0.

    To avoid training Prophet on incomplete trailing data caused by dataset
    cutoff effects, the final 5 calendar days are removed from the training
    history.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table with at least ``order_date`` (datetime) and
        ``revenue`` (float) columns.

    Returns
    -------
    pd.DataFrame
        Columns: ``ds`` (datetime64), ``y`` (float ≥ 0).
        Sorted ascending by ``ds``; no missing dates.

    Raises
    ------
    ValueError
        If ``df`` is empty or missing required columns.
    """
    if df is None or df.empty:
        raise ValueError(
            "prepare_daily_revenue() received an empty DataFrame. "
            "Load the fact table before calling this function."
        )

    required = {"order_date", "revenue"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"prepare_daily_revenue() is missing required columns: {missing}. "
            "Ensure the fact table was built by src.data_processing.build_fact_table()."
        )

    # Aggregate revenue by calendar day.
    daily = (
        df.assign(order_date=pd.to_datetime(df["order_date"]).dt.normalize())
        .groupby("order_date", sort=True)["revenue"]
        .sum()
        .rename_axis("ds")
        .rename("y")
        .reset_index()
    )

    # Fill missing calendar dates with 0 revenue.
    daily = (
        daily.set_index("ds")
        .asfreq("D")
        .fillna(0.0)
        .reset_index()
    )

    daily["ds"] = pd.to_datetime(daily["ds"])
    daily = daily.sort_values("ds").reset_index(drop=True)

    # Drop final trailing days to avoid partial dataset cutoff effects.
    if len(daily) > 5:
        original_end = daily["ds"].max()
        cutoff_date = original_end - pd.Timedelta(days=5)
        daily = daily[daily["ds"] <= cutoff_date].copy()
        logger.info(
            "Dropped final 5 days from training to avoid partial trailing data "
            "(old end date: %s, new end date: %s).",
            original_end.date(),
            daily["ds"].max().date(),
        )

    logger.info(
        "Daily revenue series: %d observations from %s to %s.",
        len(daily),
        daily["ds"].min().date(),
        daily["ds"].max().date(),
    )

    return daily


# ── Step 2: Prophet fitting and forecasting ───────────────────────────────────

def generate_forecast(df: pd.DataFrame, periods: int = 30) -> pd.DataFrame:
    """
    Fit a Prophet model on the cleaned fact table and forecast future revenue.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table from ``src.data_processing.build_fact_table()``.
    periods : int
        Number of future calendar days to forecast (default 30).

    Returns
    -------
    pd.DataFrame
        Full Prophet forecast DataFrame. Key downstream columns:
        ``ds``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    """
    if periods < 1:
        raise ValueError(f"periods must be >= 1, got {periods}.")

    history_df = prepare_daily_revenue(df)

    logger.info(
        "Fitting Prophet on %d daily observations (%s → %s), then projecting %d periods forward.",
        len(history_df),
        history_df["ds"].min().date(),
        history_df["ds"].max().date(),
        periods,
    )

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.90,
    )
    model.fit(history_df)

    future = model.make_future_dataframe(periods=periods, freq="D")
    forecast_df = model.predict(future)

    logger.info(
        "Forecast complete — %d total rows returned (%d historical + %d future).",
        len(forecast_df),
        len(history_df),
        periods,
    )

    return forecast_df


# ── Step 3: Plotly visualisation ──────────────────────────────────────────────

def plot_forecast(
    forecast_df: pd.DataFrame,
    history_df: pd.DataFrame | None = None,
) -> go.Figure:
    """
    Build a Plotly Figure visualising the Prophet forecast.

    Parameters
    ----------
    forecast_df : pd.DataFrame
        Output of :func:`generate_forecast`.
    history_df : pd.DataFrame | None
        Output of :func:`prepare_daily_revenue`.

    Returns
    -------
    plotly.graph_objects.Figure
        Ready to render with Streamlit or Plotly.
    """
    required = {"ds", "yhat", "yhat_lower", "yhat_upper"}
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(
            f"plot_forecast() missing required forecast columns: {missing}"
        )

    fig = go.Figure()

    # Confidence interval band.
    fig.add_trace(
        go.Scatter(
            x=pd.concat([forecast_df["ds"], forecast_df["ds"].iloc[::-1]]),
            y=pd.concat(
                [
                    forecast_df["yhat_upper"].clip(lower=0),
                    forecast_df["yhat_lower"].clip(lower=0).iloc[::-1],
                ]
            ),
            fill="toself",
            fillcolor="rgba(99, 132, 255, 0.15)",
            line={"color": "rgba(0,0,0,0)"},
            hoverinfo="skip",
            name="90% Confidence Interval",
            showlegend=True,
        )
    )

    # Forecast line.
    fig.add_trace(
        go.Scatter(
            x=forecast_df["ds"],
            y=forecast_df["yhat"].clip(lower=0),
            mode="lines",
            name="Forecast (yhat)",
            line={"color": "#6384FF", "width": 2, "dash": "dash"},
            hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Forecast: R$ %{y:,.0f}<extra></extra>",
        )
    )

    # Historical actuals.
    if history_df is not None and not history_df.empty:
        fig.add_trace(
            go.Scatter(
                x=history_df["ds"],
                y=history_df["y"],
                mode="lines",
                name="Actual Revenue",
                line={"color": "#00C9A7", "width": 1.8},
                hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Actual: R$ %{y:,.0f}<extra></extra>",
            )
        )

        last_hist_date = history_df["ds"].max().strftime("%Y-%m-%d")
        fig.add_shape(
            type="line",
            x0=last_hist_date,
            x1=last_hist_date,
            y0=0,
            y1=1,
            xref="x",
            yref="paper",
            line={"width": 1.5, "dash": "dot", "color": "rgba(200,200,200,0.6)"},
        )
        fig.add_annotation(
            x=last_hist_date,
            y=1,
            xref="x",
            yref="paper",
            text="Forecast start",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font={"size": 11, "color": "rgba(180,180,180,0.9)"},
        )

    fig.update_layout(
        title={
            "text": "Daily Revenue Forecast",
            "font": {"size": 20},
            "x": 0.03,
            "xanchor": "left",
        },
        xaxis={
            "title": "Date",
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "tickformat": "%b %Y",
        },
        yaxis={
            "title": "Revenue (BRL)",
            "showgrid": True,
            "gridcolor": "rgba(255,255,255,0.06)",
            "tickprefix": "R$ ",
            "tickformat": ",.0f",
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        hovermode="x unified",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin={"t": 60, "b": 40, "l": 60, "r": 20},
    )

    return fig


# ── Step 4: Pipeline convenience wrapper ──────────────────────────────────────

def run_forecast_pipeline(
    df: pd.DataFrame,
    periods: int = 30,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    End-to-end convenience wrapper: aggregate → fit → forecast.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        (history_df, forecast_df)
    """
    history_df = prepare_daily_revenue(df)
    forecast_df = generate_forecast(df, periods=periods)
    return history_df, forecast_df


# ── Quick smoke-test (run as script) ──────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    project_root = Path(__file__).resolve().parent.parent
    fact_path = project_root / "data" / "cleaned" / "fact_table.csv"

    if not fact_path.exists():
        print(f"Fact table not found at {fact_path}. Run src/data_processing.py first.")
        sys.exit(1)

    fact_df = pd.read_csv(fact_path, parse_dates=["order_date"])
    print(f"Loaded fact table: {fact_df.shape[0]:,} rows")

    history, forecast = run_forecast_pipeline(fact_df, periods=30)

    print(f"\nHistory shape : {history.shape}")
    print(f"Forecast shape: {forecast.shape}")
    print(
        f"\nForecast tail (last 5 rows):\n"
        f"{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail().to_string(index=False)}"
    )

    fig = plot_forecast(forecast, history)
    out_html = project_root / "forecast_preview.html"
    fig.write_html(str(out_html))
    print(f"\nForecast chart written to: {out_html}")