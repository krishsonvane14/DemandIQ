# src/data_loader.py
#
# Responsible for loading raw demand data from disk or external sources.
# Will support multiple file formats (CSV, Parquet, JSON) and provide
# basic validation (schema checks, null detection) before handing
# data off to the preprocessor.
#
# Key functions (to be implemented):
#   load_raw_data(filepath)  -> pd.DataFrame
#   validate_schema(df)      -> bool


def load_raw_data(filepath: str):
    """Load raw demand data from the given file path."""
    # TODO: implement file-format detection and loading
    pass


def validate_schema(df) -> bool:
    """Validate that the DataFrame conforms to the expected schema."""
    # TODO: define required columns and type checks
    pass
