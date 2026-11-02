import matplotlib.pyplot as plt
import numpy as np


def loss_plot(loss_history: list[float]):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(loss_history)
    ax.set(xlabel="Iteration", ylabel="Training MSE", title="Gradient-descent loss")
    fig.tight_layout()
    return fig


def prediction_plots(y_test: np.ndarray, predictions: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].scatter(y_test, predictions, alpha=0.7)
    lower, upper = min(y_test.min(), predictions.min()), max(y_test.max(), predictions.max())
    axes[0].plot([lower, upper], [lower, upper], "--", color="black")
    axes[0].set(xlabel="Actual score", ylabel="Raw predicted score", title="Actual vs predicted")
    residuals = y_test - predictions
    axes[1].scatter(predictions, residuals, alpha=0.7)
    axes[1].axhline(0, linestyle="--", color="black")
    axes[1].set(xlabel="Raw predicted score", ylabel="Residual (actual - predicted)", title="Residuals")
    fig.tight_layout()
    return fig

