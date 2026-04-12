# src/data_processing.py
#
# Builds the DemandIQ cleaned fact table from the raw Olist tables
# returned by src.data_loader.load_all_tables().
#
# Join graph (left-join chain starting from order_items grain):
#
#   order_items
#     ├─ orders                 (on order_id)         → order_status, order_purchase_timestamp, customer_id
#     ├─ products               (on product_id)        → product_category_name
#     ├─ category_translation   (on product_category_name) → product_category_name_english
#     └─ customers              (on customer_id)       → customer_state
#
# Output schema:
#   order_id, order_date, customer_id, product_id,
#   category, quantity, price, revenue, customer_state
#
# Public API:
#   build_fact_table(tables)  -> pd.DataFrame
#   save_fact_table(df)       -> Path

import logging
from pathlib import Path

import pandas as pd

from config import CLEAN_DATA_DIR

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
FACT_TABLE_FILENAME = "fact_table.csv"

# Order statuses to exclude — orders that never generated real demand.
EXCLUDED_STATUSES = {"canceled", "unavailable"}

# Columns that must be non-null in the final fact table.
CRITICAL_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "category",
    "price",
    "customer_state",
]

# Final column selection and order.
OUTPUT_COLUMNS = [
    "order_id",
    "order_date",
    "customer_id",
    "product_id",
    "category",
    "quantity",
    "price",
    "revenue",
    "customer_state",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def _log_shape(df: pd.DataFrame, step: str) -> None:
    """Log row/column count after a processing step."""
    logger.info("  [%s] %s rows remaining", step, f"{len(df):,}")


def _require_tables(tables: dict, names: list[str]) -> None:
    """Raise KeyError if any expected table is absent from the dict."""
    missing = [n for n in names if n not in tables]
    if missing:
        raise KeyError(
            f"build_fact_table() requires these tables which were not loaded: {missing}. "
            "Check data/raw/ and re-run src.data_loader."
        )


# ── Core pipeline ──────────────────────────────────────────────────────────────

def build_fact_table(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join Olist tables into a single cleaned fact table.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Output of ``src.data_loader.load_all_tables()``.
        Must contain: orders, order_items, products,
        category_translation, customers.

    Returns
    -------
    pd.DataFrame
        Cleaned fact table with columns defined in OUTPUT_COLUMNS.
    """
    required = ["orders", "order_items", "products", "category_translation", "customers"]
    _require_tables(tables, required)

    logger.info("Building fact table …")

    # ── Step 1: Start from order_items (one row = one purchased item) ──────────
    df = tables["order_items"].copy()
    logger.info("Starting grain: order_items")
    _log_shape(df, "start")

    # ── Step 2: Join orders → get status, timestamp, customer_id ──────────────
    orders_cols = [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
    ]
    df = df.merge(
        tables["orders"][orders_cols],
        on="order_id",
        how="left",
        validate="many_to_one",   # many items per order
    )
    _log_shape(df, "after join orders")

    # ── Step 3: Filter excluded order statuses ─────────────────────────────────
    before = len(df)
    df = df[~df["order_status"].isin(EXCLUDED_STATUSES)].copy()
    removed = before - len(df)
    logger.info(
        "  [filter statuses] removed %s rows (statuses: %s)",
        f"{removed:,}", EXCLUDED_STATUSES,
    )
    _log_shape(df, "after status filter")

    # ── Step 4: Parse order_purchase_timestamp → order_date (date only) ────────
    df["order_date"] = pd.to_datetime(
        df["order_purchase_timestamp"], errors="coerce"
    ).dt.normalize()           # midnight — strips time component
    df.drop(columns=["order_purchase_timestamp"], inplace=True)
    bad_dates = df["order_date"].isna().sum()
    if bad_dates:
        logger.warning("  [parse dates] %s rows have unparseable timestamps.", f"{bad_dates:,}")

    # ── Step 5: Join products → get product_category_name ─────────────────────
    df = df.merge(
        tables["products"][["product_id", "product_category_name"]],
        on="product_id",
        how="left",
        validate="many_to_one",
    )
    _log_shape(df, "after join products")

    # ── Step 6: Join category_translation → translate category to English ──────
    df = df.merge(
        tables["category_translation"],          # both columns kept
        on="product_category_name",
        how="left",
        validate="many_to_one",
    )
    untranslated = df["product_category_name_english"].isna().sum()
    if untranslated:
        logger.warning(
            "  [category translation] %s rows have no English category name "
            "(raw name kept as fallback).",
            f"{untranslated:,}",
        )
    # Use English name; fall back to raw Portuguese name if translation missing.
    df["category"] = df["product_category_name_english"].fillna(
        df["product_category_name"]
    )
    df.drop(columns=["product_category_name", "product_category_name_english"], inplace=True)
    _log_shape(df, "after category translation")

    # ── Step 7: Join customers → get customer_state ────────────────────────────
    df = df.merge(
        tables["customers"][["customer_id", "customer_state"]],
        on="customer_id",
        how="left",
        validate="many_to_one",
    )
    _log_shape(df, "after join customers")

    # ── Step 8: Derive quantity and revenue ────────────────────────────────────
    # Each row in order_items represents exactly one physical item.
    df["quantity"] = 1
    # Revenue = item price (freight excluded from demand signal).
    df["revenue"] = df["price"]

    # ── Step 9: Drop rows with nulls in critical columns ───────────────────────
    before = len(df)
    df.dropna(subset=CRITICAL_COLUMNS, inplace=True)
    removed = before - len(df)
    logger.info(
        "  [drop nulls] removed %s rows with nulls in critical columns: %s",
        f"{removed:,}", CRITICAL_COLUMNS,
    )
    _log_shape(df, "after drop nulls")

    # ── Step 10: Select and order final columns ────────────────────────────────
    df = df[OUTPUT_COLUMNS].reset_index(drop=True)

    logger.info(
        "Fact table ready — %s rows × %d cols.",
        f"{len(df):,}", len(df.columns),
    )
    return df


# ── Persistence ────────────────────────────────────────────────────────────────

def save_fact_table(
    df: pd.DataFrame,
    clean_dir: Path | str | None = None,
) -> Path:
    """
    Write the fact table to a CSV in *clean_dir*.

    Parameters
    ----------
    df : pd.DataFrame
        The cleaned fact table produced by ``build_fact_table()``.
    clean_dir : Path | str | None
        Destination directory. Defaults to ``config.CLEAN_DATA_DIR``.

    Returns
    -------
    Path
        Absolute path to the written CSV file.
    """
    clean_dir = Path(clean_dir) if clean_dir is not None else CLEAN_DATA_DIR
    clean_dir.mkdir(parents=True, exist_ok=True)

    out_path = clean_dir / FACT_TABLE_FILENAME
    df.to_csv(out_path, index=False)
    logger.info("Saved fact table → %s", out_path.resolve())
    return out_path


# ── Quick smoke-test (run as script) ──────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
        stream=sys.stdout,
    )

    from src.data_loader import load_all_tables

    tables = load_all_tables()
    fact = build_fact_table(tables)
    path = save_fact_table(fact)

    print(f"\nFact table written to: {path}")
    print(f"Shape: {fact.shape}")
    print(f"\nDtypes:\n{fact.dtypes}")
    print(f"\nSample (5 rows):\n{fact.head().to_string()}")
    print(f"\nOrder date range: {fact['order_date'].min()} → {fact['order_date'].max()}")
    print(f"Unique categories: {fact['category'].nunique()}")
    print(f"Unique states:     {fact['customer_state'].nunique()}")
