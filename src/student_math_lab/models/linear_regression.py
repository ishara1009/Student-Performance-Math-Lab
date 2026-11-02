import numpy as np

from student_math_lab.mathematics.calculus import mean_squared_error, mse_gradients


class NumPyLinearRegression:
    """Linear regression trained with batch gradient descent."""

    def __init__(self, learning_rate: float = 0.01, iterations: int = 5000) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be a positive finite number.")
        if not isinstance(iterations, int) or iterations <= 0:
            raise ValueError("iterations must be a positive integer.")
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.weights: np.ndarray | None = None
        self.bias = 0.0
        self.loss_history: list[float] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NumPyLinearRegression":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
        if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be 2D and y must be a matching non-empty 1D array.")
        self.weights = np.zeros(X.shape[1], dtype=float)
        self.bias = float(np.mean(y))
        self.loss_history = []
        for _ in range(self.iterations):
            predictions = X @ self.weights + self.bias
            loss = mean_squared_error(y, predictions)
            if not np.isfinite(loss):
                raise ValueError("Training diverged: loss became non-finite.")
            self.loss_history.append(loss)
            weight_gradient, bias_gradient = mse_gradients(X, y, self.weights, self.bias)
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model must be fitted before predict().")
        return np.asarray(X, dtype=float) @ self.weights + self.bias

