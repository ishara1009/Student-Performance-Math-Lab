import numpy as np


def square_derivative(x: float) -> float:
    """Return the derivative 2x of f(x) = x squared."""
    return 2.0 * x


def mean_squared_error(y: np.ndarray, predictions: np.ndarray) -> float:
    """Return mean squared prediction error."""
    return float(np.mean((np.asarray(predictions) - np.asarray(y)) ** 2))


def mse_gradients(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float
) -> tuple[np.ndarray, float]:
    """Calculate analytical MSE gradients for weights and bias."""
    errors = X @ weights + bias - y
    weight_gradient = (2.0 / len(y)) * X.T @ errors
    bias_gradient = 2.0 * float(np.mean(errors))
    return weight_gradient, bias_gradient


def finite_difference_gradients(
    X: np.ndarray, y: np.ndarray, weights: np.ndarray, bias: float, epsilon: float = 1e-6
) -> tuple[np.ndarray, float]:
    """Approximate gradients using central finite differences."""
    gradients = np.zeros_like(weights, dtype=float)
    for index in range(len(weights)):
        plus, minus = weights.copy(), weights.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        gradients[index] = (
            mean_squared_error(y, X @ plus + bias)
            - mean_squared_error(y, X @ minus + bias)
        ) / (2 * epsilon)
    bias_gradient = (
        mean_squared_error(y, X @ weights + bias + epsilon)
        - mean_squared_error(y, X @ weights + bias - epsilon)
    ) / (2 * epsilon)
    return gradients, bias_gradient

