import math

import numpy as np
import pandas as pd
from scipy import stats


def welch_extra_classes_test(data: pd.DataFrame, alpha: float = 0.05) -> dict[str, object]:
    """Compare Yes and No score means with a two-sided Welch t-test."""
    yes = data.loc[data["extra_classes"] == "Yes", "final_score"].to_numpy(dtype=float)
    no = data.loc[data["extra_classes"] == "No", "final_score"].to_numpy(dtype=float)
    if len(yes) < 2 or len(no) < 2:
        raise ValueError("Welch test needs at least two observations in each group.")
    yes_var, no_var = float(np.var(yes, ddof=1)), float(np.var(no, ddof=1))
    standard_error_squared = yes_var / len(yes) + no_var / len(no)
    if standard_error_squared == 0:
        raise ValueError("Welch test is undefined because both groups have zero variance.")
    degrees_freedom = standard_error_squared**2 / (
        (yes_var / len(yes)) ** 2 / (len(yes) - 1)
        + (no_var / len(no)) ** 2 / (len(no) - 1)
    )
    test = stats.ttest_ind(yes, no, equal_var=False)
    difference = float(np.mean(yes) - np.mean(no))
    margin = float(stats.t.ppf(1 - alpha / 2, degrees_freedom) * math.sqrt(standard_error_squared))
    p_value = float(test.pvalue)
    decision = "reject H0" if p_value < alpha else "fail to reject H0"
    return {
        "hypotheses": {"H0": "Population mean scores are equal.",
                       "H1": "Population mean scores differ."},
        "alpha": alpha, "yes_n": len(yes), "no_n": len(no),
        "yes_mean": float(np.mean(yes)), "no_mean": float(np.mean(no)),
        "yes_sample_std": float(np.std(yes, ddof=1)), "no_sample_std": float(np.std(no, ddof=1)),
        "mean_difference_yes_minus_no": difference, "t_statistic": float(test.statistic),
        "degrees_freedom": float(degrees_freedom), "p_value": p_value,
        "confidence_interval_95": [difference - margin, difference + margin],
        "decision": decision,
        "interpretation": f"At alpha={alpha}, we {decision}. This does not establish causation.",
    }
