import numpy as np
import streamlit as st

from student_math_lab.mathematics.calculus import (
    finite_difference_gradients,
    mse_gradients,
    square_derivative,
)
from student_math_lab.visualization.model_plots import loss_plot


def render(results):
    demo, training = results["linear_algebra"], results["training"]
    st.header("Mathematics")
    st.write("Student feature vector", demo["student_vector"])
    st.write("Feature matrix", demo["feature_matrix"])
    st.write({key: demo[key] for key in ["X_shape", "y_shape", "weights_shape", "predictions_shape"]})
    st.write("Dot product", demo["dot_product"])
    st.write("Matrix transpose", demo["transpose"])
    st.info("The weights shown above are demonstration values. The trained model learns different weights.")
    st.write("For f(x)=x², f'(3)=", square_derivative(3))
    X, y, weights = np.array([[1.0], [2.0]]), np.array([2.0, 4.0]), np.array([0.5])
    st.write("Analytical gradients", mse_gradients(X, y, weights, 0.0))
    st.write("Finite-difference check", finite_difference_gradients(X, y, weights, 0.0))
    st.pyplot(loss_plot(training["model"].loss_history))

