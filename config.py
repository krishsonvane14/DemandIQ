# config.py
#
# Central configuration for DemandIQ.
# All shared constants — paths, column mappings, and API credentials —
# live here so every module can import them from one place.
#
# Usage:
#   from config import RAW_DATA_DIR, COLUMN_MAP, GEMINI_API_KEY

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Load .env ──────────────────────────────────────────────────────────────────
# Looks for a .env file in the project root (same directory as this file).
# Call this before accessing any env-var-backed constants.
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent

RAW_DATA_DIR   = BASE_DIR / "data" / "raw"
CLEAN_DATA_DIR = BASE_DIR / "data" / "cleaned"

# ── Column name mappings ───────────────────────────────────────────────────────
# Maps raw source column names → standardised internal names used throughout
# the pipeline. Update the keys to match your actual source data headers.
COLUMN_MAP = {
    # Date / time
    "date":         "ds",          # Prophet expects 'ds'
    "Date":         "ds",
    "order_date":   "ds",

    # Target / demand
    "demand":       "y",           # Prophet expects 'y'
    "Demand":       "y",
    "quantity":     "y",
    "sales":        "y",

    # Optional covariates
    "product_id":   "product_id",
    "region":       "region",
    "price":        "price",
}

# Columns that must be present after mapping (used for schema validation)
REQUIRED_COLUMNS = ["ds", "y"]

# ── API credentials ───────────────────────────────────────────────────────────
# Set GEMINI_API_KEY in your .env file:
#   GEMINI_API_KEY=your_key_here
GEMINI_API_KEY: str | None = os.getenv("GEMINI_API_KEY")

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str | None = os.getenv("DATABASE_URL")   # e.g. sqlite:///demandiq.db

# ── Model defaults ────────────────────────────────────────────────────────────
FORECAST_HORIZON_DAYS = 30       # Default number of days to forecast ahead
SEASONALITY_MODE      = "multiplicative"  # or "additive"

# ── Streamlit ─────────────────────────────────────────────────────────────────
APP_TITLE   = "DemandIQ"
APP_ICON    = "📈"
THEME_COLOR = "#6C63FF"
