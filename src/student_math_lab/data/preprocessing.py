from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from student_math_lab.config import FEATURE_COLUMNS, RANDOM_STATE, TEST_SIZE


@dataclass
class Standardizer:
    """Store training-only means and scales."""

    means: np.ndarray | None = None
    scales: np.ndarray | None = None

    def fit(self, values: np.ndarray) -> "Standardizer":
        self.means = np.mean(values, axis=0)
        scales = np.std(values, axis=0, ddof=0)
        self.scales = np.where(scales == 0, 1.0, scales)
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.means is None or self.scales is None:
            raise ValueError("Standardizer must be fitted before transform().")
        return (np.asarray(values, dtype=float) - self.means) / self.scales


def encode_features(data: pd.DataFrame) -> np.ndarray:
    """Encode Yes as 1 and No as 0, returning four numerical features."""
    encoded = data[FEATURE_COLUMNS].copy()
    encoded["extra_classes"] = encoded["extra_classes"].map({"Yes": 1.0, "No": 0.0})
    if encoded["extra_classes"].isna().any():
        raise ValueError("extra_classes must contain only 'Yes' or 'No'.")
    return encoded.to_numpy(dtype=float)


def prepare_train_test(data: pd.DataFrame) -> dict[str, object]:
    """Split first, then fit scaling on training features only."""
    features = encode_features(data)
    targets = data["final_score"].to_numpy(dtype=float)
    indices = np.arange(len(data))
    train_idx, test_idx = train_test_split(
        indices, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    standardizer = Standardizer().fit(features[train_idx])
    return {
        "X_train": standardizer.transform(features[train_idx]),
        "X_test": standardizer.transform(features[test_idx]),
        "y_train": targets[train_idx], "y_test": targets[test_idx],
        "train_indices": train_idx, "test_indices": test_idx,
        "standardizer": standardizer,
    }
