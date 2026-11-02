import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, and R-squared."""
    return {"mae": float(mean_absolute_error(y, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
            "r2": float(r2_score(y, predictions))}

