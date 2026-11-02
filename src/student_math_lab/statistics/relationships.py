import pandas as pd

from student_math_lab.config import NUMERIC_COLUMNS


def relationship_matrices(data: pd.DataFrame) -> dict[str, object]:
    """Return sample covariance and Pearson correlation matrices."""
    numerical = data[NUMERIC_COLUMNS]
    constant_columns = [column for column in numerical if numerical[column].nunique() <= 1]
    return {"covariance": numerical.cov(ddof=1), "correlation": numerical.corr(),
            "constant_columns": constant_columns}

