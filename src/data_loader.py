# src/data_loader.py
#
# Loads the Olist Brazilian E-Commerce dataset from CSV files stored in
# data/raw/ and returns a dict of {table_name: pd.DataFrame}.
#
# Public API:
#   load_all_tables(raw_dir)          -> dict[str, pd.DataFrame]
#   load_table(name, filepath)        -> pd.DataFrame | None
#   validate_table(name, df)          -> bool
#
# File → table name mapping (OLIST_FILES) and required-column registry
# (REQUIRED_COLUMNS) are defined below and can be extended without
# touching any other module.

import logging
from pathlib import Path

import pandas as pd

from config import RAW_DATA_DIR

# ── Logging ────────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)

# ── Olist file manifest ────────────────────────────────────────────────────────
# Maps a short table key → the CSV filename as distributed by Kaggle.
OLIST_FILES: dict[str, str] = {
    "orders":           "olist_orders_dataset.csv",
    "order_items":      "olist_order_items_dataset.csv",
    "products":         "olist_products_dataset.csv",
    "category_translation": "product_category_name_translation.csv",
    "customers":        "olist_customers_dataset.csv",
    "payments":         "olist_order_payments_dataset.csv",
    "geolocation":      "olist_geolocation_dataset.csv",
}

# ── Required-column registry ───────────────────────────────────────────────────
# Each table must contain at least these columns after loading.
# Add/remove columns here as the pipeline evolves.
REQUIRED_COLUMNS: dict[str, list[str]] = {
    "orders": [
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
    ],
    "order_items": [
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
    ],
    "products": [
        "product_id",
        "product_category_name",
    ],
    "category_translation": [
        "product_category_name",
        "product_category_name_english",
    ],
    "customers": [
        "customer_id",
        "customer_zip_code_prefix",
        "customer_city",
        "customer_state",
    ],
    "payments": [
        "order_id",
        "payment_sequential",
        "payment_type",
        "payment_value",
    ],
    "geolocation": [
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    ],
}


# ── Core loaders ───────────────────────────────────────────────────────────────

def load_table(name: str, filepath: Path) -> pd.DataFrame | None:
    """
    Read a single CSV file into a DataFrame.

    Parameters
    ----------
    name : str
        Logical table name (used for logging and validation).
    filepath : Path
        Absolute or relative path to the CSV file.

    Returns
    -------
    pd.DataFrame | None
        The loaded DataFrame, or None if the file could not be read.
    """
    filepath = Path(filepath)

    if not filepath.exists():
        logger.warning("Missing file for table '%s': %s", name, filepath)
        return None

    try:
        df = pd.read_csv(filepath, low_memory=False)
        logger.info(
            "Loaded table '%s' — %d rows × %d cols from %s",
            name, len(df), len(df.columns), filepath.name,
        )
        return df
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read '%s' (%s): %s", name, filepath, exc)
        return None


def validate_table(name: str, df: pd.DataFrame) -> bool:
    """
    Check that a DataFrame contains all required columns for its table.

    Parameters
    ----------
    name : str
        Table key used to look up the required-column list.
    df : pd.DataFrame
        The DataFrame to validate.

    Returns
    -------
    bool
        True if all required columns are present, False otherwise.
        Missing columns are logged as warnings.
    """
    required = REQUIRED_COLUMNS.get(name, [])
    missing = [col for col in required if col not in df.columns]

    if missing:
        logger.warning(
            "Table '%s' is missing required columns: %s", name, missing
        )
        return False

    logger.debug("Table '%s' passed schema validation.", name)
    return True


def load_all_tables(
    raw_dir: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load every Olist CSV file from *raw_dir* and validate each table.

    Parameters
    ----------
    raw_dir : Path | str | None
        Directory containing the raw CSV files.
        Defaults to ``config.RAW_DATA_DIR`` when None.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of table name → DataFrame for every file that was
        successfully loaded *and* passed schema validation.
        Tables that are missing or fail validation are excluded from
        the result (a warning is logged for each).
    """
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_DATA_DIR

    logger.info("Loading Olist tables from: %s", raw_dir.resolve())

    tables: dict[str, pd.DataFrame] = {}
    failed: list[str] = []

    for name, filename in OLIST_FILES.items():
        filepath = raw_dir / filename
        df = load_table(name, filepath)

        if df is None:
            failed.append(name)
            continue

        valid = validate_table(name, df)
        if valid:
            tables[name] = df
        else:
            failed.append(name)

    # ── Summary ────────────────────────────────────────────────────────────────
    loaded = list(tables.keys())
    logger.info(
        "Load complete — %d/%d tables ready: %s",
        len(loaded), len(OLIST_FILES), loaded,
    )
    if failed:
        logger.warning(
            "%d table(s) skipped due to missing files or schema errors: %s",
            len(failed), failed,
        )

    return tables


# ── Quick smoke-test (run as script) ──────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-8s %(name)s — %(message)s",
    )
    data = load_all_tables()
    print("\nAvailable tables:")
    for tname, tdf in data.items():
        print(f"  {tname:<24} {tdf.shape[0]:>7,} rows  {list(tdf.columns)[:4]}…")
