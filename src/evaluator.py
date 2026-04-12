# src/evaluator.py
#
# Computes and reports performance metrics for the trained model.
# Metrics planned:
#   - MAE  (Mean Absolute Error)
#   - RMSE (Root Mean Squared Error)
#   - MAPE (Mean Absolute Percentage Error)
#   - R²   (Coefficient of Determination)
#
# Key functions (to be implemented):
#   evaluate(y_true, y_pred)       -> dict[str, float]
#   print_report(metrics)          -> None
#   plot_predictions(y_true, y_pred, dates) -> None


def evaluate(y_true, y_pred) -> dict:
    """Compute forecast accuracy metrics and return as a dictionary."""
    # TODO: implement metric calculations
    return {}


def print_report(metrics: dict) -> None:
    """Pretty-print the evaluation metrics to stdout."""
    # TODO: implement formatted report output
    pass


def plot_predictions(y_true, y_pred, dates=None) -> None:
    """Plot actual vs. predicted demand values (e.g., with matplotlib)."""
    # TODO: implement visualisation
    pass
