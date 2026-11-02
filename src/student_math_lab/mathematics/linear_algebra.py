import numpy as np
import pandas as pd

from student_math_lab.data.preprocessing import encode_features


def linear_algebra_demo(data: pd.DataFrame, rows: int = 3) -> dict[str, object]:
    """Return small vector and matrix examples from the dataset."""
    feature_matrix = encode_features(data.iloc[:rows])
    student_vector = feature_matrix[0]
    demo_weights = np.array([0.4, 0.3, 0.2, 2.0])
    demo_bias = 5.0
    targets = data["final_score"].iloc[:rows].to_numpy(dtype=float)
    predictions = feature_matrix @ demo_weights + demo_bias
    return {
        "student_vector": student_vector, "feature_matrix": feature_matrix,
        "X_shape": feature_matrix.shape, "y_shape": targets.shape,
        "weights_shape": demo_weights.shape, "predictions_shape": predictions.shape,
        "vector_addition": student_vector + np.ones(4),
        "scalar_multiplication": 2 * student_vector,
        "dot_product": float(student_vector @ demo_weights),
        "transpose": feature_matrix.T, "matrix_vector_product": feature_matrix @ demo_weights,
        "predictions": predictions, "demo_weights": demo_weights, "demo_bias": demo_bias,
    }

