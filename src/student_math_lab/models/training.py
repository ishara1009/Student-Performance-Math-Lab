import numpy as np
from sklearn.linear_model import LinearRegression

from student_math_lab.data.preprocessing import prepare_train_test
from student_math_lab.models.evaluation import regression_metrics
from student_math_lab.models.linear_regression import NumPyLinearRegression


def train_and_compare(data) -> dict[str, object]:
    """Train scratch, sklearn, and mean-baseline models on one split."""
    prepared = prepare_train_test(data)
    X_train, X_test = prepared["X_train"], prepared["X_test"]
    y_train, y_test = prepared["y_train"], prepared["y_test"]
    model = NumPyLinearRegression().fit(X_train, y_train)
    scratch_predictions = model.predict(X_test)
    sklearn_model = LinearRegression().fit(X_train, y_train)
    sklearn_predictions = sklearn_model.predict(X_test)
    baseline_predictions = np.full_like(y_test, np.mean(y_train), dtype=float)
    return {
        **prepared, "model": model, "sklearn_model": sklearn_model,
        "predictions": scratch_predictions,
        "sklearn_predictions": sklearn_predictions, "baseline_predictions": baseline_predictions,
        "metrics": {
            "numpy_gradient_descent": regression_metrics(y_test, scratch_predictions),
            "sklearn_linear_regression": regression_metrics(y_test, sklearn_predictions),
            "training_mean_baseline": regression_metrics(y_test, baseline_predictions),
        },
        "out_of_range_count": int(((scratch_predictions < 0) | (scratch_predictions > 100)).sum()),
    }

