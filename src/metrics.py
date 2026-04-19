# src/metrics.py
#
# DemandIQ metric functions.
#
# All functions accept the cleaned fact table produced by
# src.data_processing.build_fact_table() and return values that are
# directly usable in Streamlit widgets and Plotly charts.
#
# Fact table grain: one row = one purchased item line, keyed by
# (order_id, order_item_id).  Because a single order can span multiple
# rows, order-level aggregates (e.g. total_orders, avg_order_value) must
# de-duplicate on order_id rather than simply counting rows.
#
# Output schema of the fact table (for reference):
#   order_id        str     – unique per order
#   order_item_id   int     – position within the order (1-based)
#   order_date      datetime64[ns] – purchase date (time stripped)
#   customer_id     str
#   product_id      str
#   category        str     – English product category
#   quantity        int     – always 1 at this grain
#   price           float   – unit price (BRL)
#   revenue         float   – = price (freight excluded)
#   customer_state  str     – 2-letter Brazilian state code
#
# Public API:
#   total_revenue(df)                   -> float
#   total_orders(df)                    -> int
#   avg_order_value(df)                 -> float
#   revenue_by_category(df)             -> pd.DataFrame
#   revenue_by_state(df)                -> pd.DataFrame
#   revenue_over_time(df, freq='W')     -> pd.DataFrame

import pandas as pd


# ── Scalar metrics ─────────────────────────────────────────────────────────────

def total_revenue(df: pd.DataFrame) -> float:
    """
    Sum of the ``revenue`` column across all item rows.

    Because revenue is stored at the item level (one row per item),
    a simple sum correctly aggregates without any de-duplication.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table from ``src.data_processing.build_fact_table()``.

    Returns
    -------
    float
        Total revenue in BRL.
    """
    return float(df["revenue"].sum())


def total_orders(df: pd.DataFrame) -> int:
    """
    Count of unique orders.

    Uses ``order_id.nunique()`` rather than ``len(df)`` because each order
    can contain multiple item rows at the (order_id, order_item_id) grain.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table.

    Returns
    -------
    int
        Number of distinct orders.
    """
    return int(df["order_id"].nunique())


def avg_order_value(df: pd.DataFrame) -> float:
    """
    Average revenue per order (total revenue ÷ unique order count).

    This is the correct AOV calculation for a fact table whose grain is
    per-item:  sum all revenue, then divide by the number of *orders*,
    not the number of rows.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table.

    Returns
    -------
    float
        Average order value in BRL, or 0.0 if the table is empty.
    """
    n_orders = total_orders(df)
    if n_orders == 0:
        return 0.0
    return total_revenue(df) / n_orders


# ── Grouped metrics ────────────────────────────────────────────────────────────

def revenue_by_category(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total revenue and order count grouped by product category.

    The result is sorted by revenue descending so the highest-value
    categories appear first in bar charts and data tables.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table.

    Returns
    -------
    pd.DataFrame
        Columns: ``category``, ``revenue``, ``orders``
        One row per category, sorted by ``revenue`` descending.
    """
    result = (
        df.groupby("category", sort=False)
        .agg(
            revenue=("revenue", "sum"),
            # Count unique orders per category (a single order may span
            # multiple categories if the customer bought from different ones).
            orders=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    return result


def revenue_by_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total revenue and order count grouped by customer state.

    Sorted by revenue descending so the top-performing states appear
    first in choropleth legends and ranking charts.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table.

    Returns
    -------
    pd.DataFrame
        Columns: ``customer_state``, ``revenue``, ``orders``
        One row per state, sorted by ``revenue`` descending.
    """
    result = (
        df.groupby("customer_state", sort=False)
        .agg(
            revenue=("revenue", "sum"),
            # Count unique orders per state.
            orders=("order_id", "nunique"),
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )
    return result


def revenue_over_time(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """
    Revenue and order count resampled at the requested calendar frequency.

    ``order_date`` is used as the time axis.  The column is period-bucketed
    via ``pd.Grouper`` so partial periods at the start/end of the dataset
    are handled consistently regardless of the chosen frequency.

    The returned DataFrame is sorted chronologically (ascending) so it can
    be passed directly to ``px.line`` or ``st.line_chart`` without further
    processing.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned fact table.  ``order_date`` must be ``datetime64[ns]``.
    freq : str
        Any pandas offset alias: ``'D'`` (daily), ``'W'`` (weekly, default),
        ``'ME'`` (month-end), ``'QE'`` (quarter-end), ``'YE'`` (year-end).
        See https://pandas.pydata.org/docs/user_guide/timeseries.html#offset-aliases

    Returns
    -------
    pd.DataFrame
        Columns: ``order_date``, ``revenue``, ``orders``
        Indexed by integer position; ``order_date`` is a ``datetime64``
        column (period start) ready for use as the x-axis in a time-series
        plot.  Periods with zero activity are included as 0 only when the
        resample produces them naturally; no explicit zero-fill is applied
        so sparse datasets remain compact.
    """
    # Ensure order_date is a proper datetime before grouping.
    df = df.copy()
    df["order_date"] = pd.to_datetime(df["order_date"])

    result = (
        df.groupby(pd.Grouper(key="order_date", freq=freq), sort=True)
        .agg(
            revenue=("revenue", "sum"),
            # nunique correctly handles multiple item rows per order.
            orders=("order_id", "nunique"),
        )
        .reset_index()
        # Drop any leading/trailing periods that have no data at all
        # (revenue == 0 AND orders == 0 from the resample fill).
        .query("revenue > 0 or orders > 0")
        .reset_index(drop=True)
    )

    return result
