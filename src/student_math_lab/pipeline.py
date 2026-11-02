from student_math_lab.config import PASS_MARK
from student_math_lab.data.loader import load_student_data
from student_math_lab.data.validation import validate_student_data
from student_math_lab.mathematics.linear_algebra import linear_algebra_demo
from student_math_lab.models.training import train_and_compare
from student_math_lab.statistics.descriptive import descriptive_statistics
from student_math_lab.statistics.distributions import distribution_summary
from student_math_lab.statistics.hypothesis_testing import welch_extra_classes_test
from student_math_lab.statistics.probability import empirical_probabilities
from student_math_lab.statistics.relationships import relationship_matrices


def run_pipeline(path=None) -> dict[str, object]:
    """Run validation, mathematics, statistics, and model training."""
    data = load_student_data() if path is None else load_student_data(path)
    validate_student_data(data)
    return {
        "data": data, "descriptive": descriptive_statistics(data["final_score"].to_numpy()),
        "probabilities": empirical_probabilities(data, PASS_MARK),
        "distribution": distribution_summary(data["final_score"].to_numpy()),
        "relationships": relationship_matrices(data),
        "hypothesis_test": welch_extra_classes_test(data),
        "linear_algebra": linear_algebra_demo(data), "training": train_and_compare(data),
    }
