from collections import Counter

import numpy as np


def descriptive_statistics(values: np.ndarray) -> dict[str, object]:
    """Calculate core statistics and all modes."""
    numbers = np.asarray(values, dtype=float)
    if numbers.size == 0:
        raise ValueError("Descriptive statistics require at least one value.")
    counts = Counter(numbers.tolist())
    highest = max(counts.values())
    modes = sorted(value for value, count in counts.items() if count == highest)
    mean = float(sum(numbers) / len(numbers))
    population_variance = float(sum((numbers - mean) ** 2) / len(numbers))
    sample_variance = float(np.var(numbers, ddof=1)) if len(numbers) > 1 else None
    return {
        "count": len(numbers), "mean": mean, "median": float(np.median(numbers)),
        "modes": modes, "minimum": float(np.min(numbers)), "maximum": float(np.max(numbers)),
        "population_variance": population_variance,
        "population_std": float(np.sqrt(population_variance)),
        "sample_variance": sample_variance,
        "sample_std": float(np.sqrt(sample_variance)) if sample_variance is not None else None,
    }

