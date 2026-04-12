# src/model.py
#
# Defines the DemandIQ forecasting model.
# Will wrap a chosen algorithm (e.g., XGBoost, LightGBM, Prophet, LSTM)
# behind a consistent interface so the rest of the pipeline remains
# algorithm-agnostic.
#
# Key class / functions (to be implemented):
#   class DemandModel:
#       train(X_train, y_train)         -> self
#       predict(X)                      -> np.ndarray
#       save(path)                      -> None
#       load(path)  [classmethod]       -> DemandModel


class DemandModel:
    """Forecasting model wrapper for DemandIQ."""

    def __init__(self, params: dict = None):
        # TODO: initialise the underlying algorithm with params
        self.params = params or {}
        self.model = None

    def train(self, X_train, y_train):
        """Fit the model on training data."""
        # TODO: implement training loop / fit call
        pass

    def predict(self, X):
        """Generate demand forecasts for the given feature matrix."""
        # TODO: implement prediction
        pass

    def save(self, path: str):
        """Serialise the trained model to disk."""
        # TODO: implement model serialisation (e.g., joblib or pickle)
        pass

    @classmethod
    def load(cls, path: str) -> "DemandModel":
        """Deserialise a previously saved model from disk."""
        # TODO: implement model deserialisation
        pass
