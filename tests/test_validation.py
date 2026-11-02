import pandas as pd
import pytest

from student_math_lab.data.validation import validate_student_data


def test_valid_data_passes(valid_data):
    validate_student_data(valid_data)


@pytest.mark.parametrize("column,value", [
    ("study_hours_per_week", 169), ("attendance_pct", -1),
    ("sleep_hours", 25), ("final_score", 101), ("extra_classes", "Maybe"),
])
def test_invalid_values_fail(valid_data, column, value):
    valid_data.loc[0, column] = value
    with pytest.raises(ValueError):
        validate_student_data(valid_data)


def test_missing_duplicate_and_blank_ids_fail(valid_data):
    for changed in [valid_data.drop(columns="final_score"), valid_data.assign(student_id=["S1"] * 4),
                    valid_data.assign(student_id=["", "S2", "S3", "S4"]), pd.DataFrame()]:
        with pytest.raises(ValueError):
            validate_student_data(changed)

