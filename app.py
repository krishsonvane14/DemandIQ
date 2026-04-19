# app.py
#
# DemandIQ — Streamlit dashboard entry point.
#
# Run with:  streamlit run app.py
#
# Pipeline:
#   1. Load cleaned fact table (uploaded CSV or default data/cleaned/fact_table.csv)
#   2. Apply sidebar date-range filter
#   3. Compute KPI scalars via src.metrics
#   4. Render bar / line charts via Plotly + src.metrics grouped aggregates
#   5. Fit Prophet and render 30-day forecast via src.forecasting
#   6. Generate Gemini narrative insights via src.insights
#
# Business logic lives entirely in src/* — this file only wires the UI.

import warnings
warnings.filterwarnings("ignore")           # suppress Prophet / Stan FutureWarnings

import pandas as pd
import plotly.express as px
import streamlit as st

from src.forecasting import plot_forecast, run_forecast_pipeline
from src.insights import generate_insights
from src.metrics import (
    avg_order_value,
    revenue_by_category,
    revenue_by_state,
    revenue_over_time,
    total_orders,
    total_revenue,
)

# ── Page config (must be the very first Streamlit call) ───────────────────────
st.set_page_config(
    page_title="DemandIQ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inline CSS — minimal, dashboard-like ──────────────────────────────────────
st.markdown(
    """
    <style>
    /* KPI card */
    .kpi-card {
        background: linear-gradient(135deg, #1e1e2e, #2a2a3d);
        border: 1px solid rgba(99, 132, 255, 0.25);
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-label {
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #8888aa;
        margin-bottom: 6px;
    }
    .kpi-value {
        font-size: 1.9rem;
        font-weight: 700;
        color: #ffffff;
        line-height: 1;
    }
    .kpi-sub {
        font-size: 0.75rem;
        color: #6384ff;
        margin-top: 4px;
    }
    /* Section divider */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a0a8ff;
        letter-spacing: 0.04em;
        border-left: 3px solid #6384ff;
        padding-left: 10px;
        margin: 8px 0 16px 0;
    }
    /* Insight card */
    .insight-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid rgba(99, 132, 255, 0.3);
        border-radius: 12px;
        padding: 20px 24px;
        line-height: 1.7;
        color: #d0d4f0;
        font-size: 0.96rem;
    }
    .rec-item {
        background: rgba(99, 132, 255, 0.08);
        border-left: 3px solid #6384ff;
        border-radius: 6px;
        padding: 10px 14px;
        margin-bottom: 8px;
        color: #c8ceee;
        font-size: 0.93rem;
    }
    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FACT_PATH = "data/cleaned/fact_table.csv"


@st.cache_data(show_spinner="Loading data …")
def load_data(file_bytes: bytes | None = None) -> pd.DataFrame:
    """
    Single entry point for all data loading.

    If *file_bytes* is provided (from file uploader), parse it as CSV.
    Otherwise fall back to the default fact table on disk.
    """
    if file_bytes is not None:
        import io
        return pd.read_csv(io.BytesIO(file_bytes), parse_dates=["order_date"])
    return pd.read_csv(DEFAULT_FACT_PATH, parse_dates=["order_date"])


@st.cache_data(show_spinner="Running forecast (this takes ~20 s) …")
def cached_forecast(df_json: str, periods: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fit Prophet and return (history_df, forecast_df).

    The fact table is passed as a JSON string so Streamlit can hash it
    for the cache key without issues around mutable DataFrame objects.
    The string is wrapped in StringIO because pandas ≥ 2.1 treats bare
    strings as file paths, which causes FileNotFoundError.
    """
    import io
    df = pd.read_json(io.StringIO(df_json), orient="split")
    df["order_date"] = pd.to_datetime(df["order_date"])
    return run_forecast_pipeline(df, periods=periods)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚙️ Controls")

    uploaded = st.file_uploader(
        "Upload fact table CSV",
        type=["csv"],
        help="Must match the DemandIQ fact table schema.",
    )

    st.markdown("---")

    freq_label_map = {
        "Daily":     "D",
        "Weekly":    "W",
        "Monthly":   "ME",
        "Quarterly": "QE",
    }
    freq_label = st.selectbox(
        "Revenue time-series frequency",
        options=list(freq_label_map.keys()),
        index=1,        # default: Weekly
    )
    freq = freq_label_map[freq_label]

    st.markdown("---")
    st.caption("Date range filter (applied to all charts)")

    # Placeholder — will be overwritten once the dataframe is loaded.
    date_range_placeholder = st.empty()

# ─────────────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────────────

if uploaded is not None:
    try:
        raw_df = load_data(uploaded.read())
        data_source_label = f"📂 {uploaded.name}"
    except Exception as exc:
        st.error(
            f"Failed to parse uploaded file **{uploaded.name}**: `{exc}`. "
            "Please upload a valid fact-table CSV."
        )
        st.stop()
else:
    try:
        raw_df = load_data()
        data_source_label = "📦 data/cleaned/fact_table.csv"
    except FileNotFoundError:
        st.error(
            "Default fact table not found at `data/cleaned/fact_table.csv`. "
            "Upload a CSV using the sidebar to continue."
        )
        st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Date range filter (rendered in sidebar after data is available)
# ─────────────────────────────────────────────────────────────────────────────

min_date = raw_df["order_date"].min().date()
max_date = raw_df["order_date"].max().date()

with date_range_placeholder:
    date_from, date_to = st.date_input(
        label="Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed",
    )

# Apply filter — guard against the user clearing the picker (returns 1 date).
if isinstance(date_from, type(min_date)) and isinstance(date_to, type(max_date)):
    df = raw_df[
        (raw_df["order_date"].dt.date >= date_from)
        & (raw_df["order_date"].dt.date <= date_to)
    ].copy()
else:
    df = raw_df.copy()

if df.empty:
    st.warning("No data in the selected date range. Adjust the filter and try again.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:4px'>
        <span style='font-size:2.2rem'>📈</span>
        <div>
            <div style='font-size:1.8rem;font-weight:800;color:#ffffff;line-height:1'>DemandIQ</div>
            <div style='font-size:0.85rem;color:#6384ff;margin-top:2px'>Brazilian E-Commerce Demand Intelligence</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption(f"Data source: {data_source_label} · {len(df):,} rows · {date_from} → {date_to}")
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# Section 1: KPI Cards
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

rev   = total_revenue(df)
ords  = total_orders(df)
aov   = avg_order_value(df)

kpi1, kpi2, kpi3 = st.columns(3)

with kpi1:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Total Revenue</div>
            <div class="kpi-value">R$ {rev:,.0f}</div>
            <div class="kpi-sub">BRL across all orders</div>
        </div>""",
        unsafe_allow_html=True,
    )

with kpi2:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Total Orders</div>
            <div class="kpi-value">{ords:,}</div>
            <div class="kpi-sub">unique orders</div>
        </div>""",
        unsafe_allow_html=True,
    )

with kpi3:
    st.markdown(
        f"""<div class="kpi-card">
            <div class="kpi-label">Avg Order Value</div>
            <div class="kpi-value">R$ {aov:,.2f}</div>
            <div class="kpi-sub">revenue per order</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 2: Revenue Breakdown Charts
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">Revenue Breakdown</div>', unsafe_allow_html=True)

cat_df   = revenue_by_category(df)
state_df = revenue_by_state(df)

col_cat, col_state = st.columns(2)

# Shared base layout — axis config is deliberately excluded so each chart
# can set its own xaxis/yaxis without risking duplicate-kwarg TypeErrors.
_GRID = "rgba(255,255,255,0.05)"

_base_chart_layout = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font_color="#c8ceee",
    margin=dict(t=40, b=40, l=40, r=20),
)

with col_cat:
    fig_cat = px.bar(
        cat_df.head(15),            # top 15 categories
        x="revenue",
        y="category",
        orientation="h",
        title="Top 15 Categories by Revenue",
        labels={"revenue": "Revenue (BRL)", "category": "Category"},
        color="revenue",
        color_continuous_scale="Blues",
    )
    fig_cat.update_layout(
        **_base_chart_layout,
        coloraxis_showscale=False,
        title_font_size=14,
    )
    fig_cat.update_xaxes(gridcolor=_GRID)
    fig_cat.update_yaxes(autorange="reversed", gridcolor=_GRID)
    fig_cat.update_traces(
        hovertemplate="<b>%{y}</b><br>Revenue: R$ %{x:,.0f}<extra></extra>"
    )
    st.plotly_chart(fig_cat, use_container_width=True)

with col_state:
    fig_state = px.bar(
        state_df,
        x="customer_state",
        y="revenue",
        title="Revenue by Customer State",
        labels={"revenue": "Revenue (BRL)", "customer_state": "State"},
        color="revenue",
        color_continuous_scale="Purples",
    )
    fig_state.update_layout(
        **_base_chart_layout,
        coloraxis_showscale=False,
        title_font_size=14,
    )
    fig_state.update_xaxes(gridcolor=_GRID)
    fig_state.update_yaxes(gridcolor=_GRID)
    fig_state.update_traces(
        hovertemplate="<b>%{x}</b><br>Revenue: R$ %{y:,.0f}<extra></extra>"
    )
    st.plotly_chart(fig_state, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 3: Revenue Over Time
# ─────────────────────────────────────────────────────────────────────────────

st.markdown(
    f'<div class="section-header">Revenue Over Time — {freq_label}</div>',
    unsafe_allow_html=True,
)

ts_df = revenue_over_time(df, freq=freq)

fig_ts = px.line(
    ts_df,
    x="order_date",
    y="revenue",
    title=f"{freq_label} Revenue",
    labels={"order_date": "Date", "revenue": "Revenue (BRL)"},
    color_discrete_sequence=["#6384ff"],
)
fig_ts.update_traces(
    line_width=2,
    hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Revenue: R$ %{y:,.0f}<extra></extra>",
)
fig_ts.update_layout(
    **_base_chart_layout,
    title_font_size=14,
    hovermode="x unified",
)
fig_ts.update_xaxes(gridcolor=_GRID)
fig_ts.update_yaxes(gridcolor=_GRID, title="Revenue (BRL)")
st.plotly_chart(fig_ts, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 4: 30-Day Forecast
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">30-Day Revenue Forecast (Prophet)</div>', unsafe_allow_html=True)

with st.spinner("Fitting forecast model …"):
    # Pass the unfiltered raw_df to Prophet so it trains on the full history.
    # The date filter is intentionally not applied here — truncating history
    # would degrade forecast quality.
    history_df, forecast_df = cached_forecast(
        raw_df.to_json(orient="split", date_format="iso")
    )

fig_fc = plot_forecast(forecast_df, history_df)
fig_fc.update_layout(**_base_chart_layout)
st.plotly_chart(fig_fc, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# Section 5: Gemini AI Insights
# ─────────────────────────────────────────────────────────────────────────────

st.markdown('<div class="section-header">AI-Generated Insights</div>', unsafe_allow_html=True)


def _infer_trend(ts: pd.DataFrame) -> str:
    """
    Classify the revenue trend from the last 8 periods of the time series.

    Compares the mean of the final 4 periods against the 4 before that.
    Returns 'increasing', 'decreasing', or 'stable'.
    """
    if len(ts) < 8:
        return "stable"
    recent   = ts["revenue"].iloc[-4:].mean()
    previous = ts["revenue"].iloc[-8:-4].mean()
    delta_pct = (recent - previous) / (previous + 1e-9) * 100
    if delta_pct > 5:
        return "increasing"
    if delta_pct < -5:
        return "decreasing"
    return "stable"


# Build metrics dict — derive values from already-computed results.
top_category = cat_df.iloc[0]["category"] if not cat_df.empty else "N/A"
top_state    = state_df.iloc[0]["customer_state"] if not state_df.empty else "N/A"
trend        = _infer_trend(ts_df)

# Sum only future forecast rows (ds > last training date) for the 30-day total.
last_history_date = history_df["ds"].max()
future_rows = forecast_df[forecast_df["ds"] > last_history_date]
forecast_next_30 = float(future_rows["yhat"].clip(lower=0).sum())

metrics_dict = {
    "total_revenue":           rev,
    "avg_order_value":         aov,
    "top_category":            top_category,
    "top_state":               top_state,
    "revenue_trend_direction": trend,
    "forecast_next_30_days":   forecast_next_30,
}

with st.spinner("Generating insights with Gemini …"):
    insights = generate_insights(metrics_dict)

# Insight paragraph
st.markdown(
    f'<div class="insight-box">{insights["insight_paragraph"]}</div>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("**Recommendations**")

for rec in insights.get("recommendations", []):
    st.markdown(
        f'<div class="rec-item">💡 {rec}</div>',
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption("DemandIQ · Olist Brazilian E-Commerce Dataset · Built with Streamlit + Prophet + Gemini")
