import numpy as np
import pandas as pd

from student_math_lab.config import NUMERIC_COLUMNS, REQUIRED_COLUMNS


def validate_student_data(data: pd.DataFrame) -> None:
    """Raise ValueError when student data violates the documented schema."""
    missing = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if data.empty:
        raise ValueError("Dataset is empty.")
    if data[REQUIRED_COLUMNS].isna().any().any():
        columns = data[REQUIRED_COLUMNS].columns[data[REQUIRED_COLUMNS].isna().any()].tolist()
        raise ValueError(f"Missing values found in: {', '.join(columns)}")

    student_ids = data["student_id"].astype(str)
    if student_ids.str.strip().eq("").any():
        raise ValueError("student_id contains blank values.")
    if student_ids.duplicated().any():
        raise ValueError("student_id contains duplicate values.")

    for column in NUMERIC_COLUMNS:
        converted = pd.to_numeric(data[column], errors="coerce")
        if converted.isna().any() or not np.isfinite(converted.to_numpy()).all():
            raise ValueError(f"{column} contains invalid or non-finite numeric values.")

    limits = {
        "study_hours_per_week": (0, 168), "attendance_pct": (0, 100),
        "sleep_hours": (0, 24), "final_score": (0, 100),
    }
    for column, (lower, upper) in limits.items():
        if not data[column].between(lower, upper).all():
            raise ValueError(f"{column} must be between {lower} and {upper}.")
    invalid_categories = set(data["extra_classes"]) - {"Yes", "No"}
    if invalid_categories:
        raise ValueError("extra_classes must contain only 'Yes' or 'No'.")

