import numpy as np
from scipy import stats


def distribution_summary(values: np.ndarray) -> dict[str, object]:
    """Return fitted-normal parameters, z-scores, and Q-Q coordinates."""
    numbers = np.asarray(values, dtype=float)
    if numbers.size == 0:
        raise ValueError("Distribution analysis requires at least one value.")
    mean, std = float(np.mean(numbers)), float(np.std(numbers, ddof=0))
    z_scores = np.zeros_like(numbers) if std == 0 else (numbers - mean) / std
    (theoretical, ordered), _ = stats.probplot(numbers, dist="norm")
    return {"mean": mean, "std": std, "z_scores": z_scores,
            "qq_theoretical": theoretical, "qq_ordered": ordered}

