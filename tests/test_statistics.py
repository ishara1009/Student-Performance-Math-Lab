import numpy as np
import pandas as pd
import pytest
from scipy import stats

from student_math_lab.statistics.descriptive import descriptive_statistics
from student_math_lab.statistics.hypothesis_testing import welch_extra_classes_test
from student_math_lab.statistics.probability import empirical_probabilities


def test_descriptive_known_values_and_multiple_modes():
    result = descriptive_statistics(np.array([1, 1, 2, 2, 4]))
    assert result["mean"] == 2.0
    assert result["median"] == 2.0
    assert result["modes"] == [1.0, 2.0]
    assert result["population_variance"] == 1.2
    assert result["sample_variance"] == 1.5


def test_empirical_and_empty_conditional_probabilities(valid_data):
    result = empirical_probabilities(valid_data)
    assert result["pass"]["result"] == 0.75
    assert result["pass_given_attendance"]["result"] == 1.0
    empty_group = empirical_probabilities(valid_data, attendance_threshold=101)
    assert empty_group["pass_given_attendance"] == {"numerator": 0, "denominator": 0, "result": None}


def test_welch_test_matches_scipy():
    yes, no = [80, 82, 85, 87], [65, 68, 70, 72, 74]
    data = pd.DataFrame({"extra_classes": ["Yes"] * 4 + ["No"] * 5,
                         "final_score": yes + no})
    result = welch_extra_classes_test(data)
    expected = stats.ttest_ind(yes, no, equal_var=False)
    assert result["t_statistic"] == pytest.approx(expected.statistic)
    assert result["p_value"] == pytest.approx(expected.pvalue)
    assert result["confidence_interval_95"][0] < result["mean_difference_yes_minus_no"]
    assert result["confidence_interval_95"][1] > result["mean_difference_yes_minus_no"]

