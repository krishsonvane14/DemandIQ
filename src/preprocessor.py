# src/preprocessor.py
#
# Handles all data-cleaning and feature-engineering steps before
# data is fed into the model. Steps will include:
#   - Handling missing values
#   - Parsing and normalising date/time fields
#   - Encoding categorical variables
#   - Scaling / normalising numeric features
#   - Generating lag features and rolling statistics for time-series
#
# Key functions (to be implemented):
#   clean(df)            -> pd.DataFrame
#   engineer_features(df)-> pd.DataFrame
#   split_data(df)       -> tuple[pd.DataFrame, pd.DataFrame]


def clean(df):
    """Remove nulls, fix dtypes, and standardise column names."""
    # TODO: implement cleaning logic
    pass


def engineer_features(df):
    """Add derived features (lags, rolling means, seasonality flags, etc.)."""
    # TODO: implement feature engineering
    pass


def split_data(df, test_size: float = 0.2):
    """Split data into train and test sets (time-aware split)."""
    # TODO: implement chronological split
    pass
