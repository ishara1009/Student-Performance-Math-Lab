import numpy as np

from student_math_lab.data.preprocessing import Standardizer
from student_math_lab.mathematics.calculus import finite_difference_gradients, mse_gradients
from student_math_lab.models.linear_regression import NumPyLinearRegression


def test_analytical_gradients_match_finite_differences():
    X = np.array([[1.0, 2.0], [3.0, -1.0], [0.5, 4.0]])
    y, weights, bias = np.array([2.0, 1.0, 3.0]), np.array([0.2, -0.1]), 0.4
    analytical = mse_gradients(X, y, weights, bias)
    numerical = finite_difference_gradients(X, y, weights, bias)
    assert np.allclose(analytical[0], numerical[0], atol=1e-5)
    assert np.isclose(analytical[1], numerical[1], atol=1e-5)


def test_loss_reduces_and_predictions_fit_known_line():
    X = np.arange(-2, 3, dtype=float).reshape(-1, 1)
    y = 3 * X[:, 0] + 2
    model = NumPyLinearRegression(learning_rate=0.05, iterations=1000).fit(X, y)
    assert model.loss_history[-1] < model.loss_history[0]
    assert np.allclose(model.predict(X), y, atol=1e-4)


def test_standardizer_uses_fitted_values_and_handles_constant_feature():
    training = np.array([[1.0, 5.0], [3.0, 5.0]])
    test = np.array([[100.0, 5.0]])
    scaler = Standardizer().fit(training)
    transformed = scaler.transform(test)
    assert np.array_equal(scaler.means, np.array([2.0, 5.0]))
    assert np.array_equal(scaler.scales, np.array([1.0, 1.0]))
    assert np.isfinite(transformed).all()
