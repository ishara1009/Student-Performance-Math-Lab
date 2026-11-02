import pandas as pd
import pytest


@pytest.fixture
def valid_data():
    return pd.DataFrame({
        "student_id": ["S1", "S2", "S3", "S4"],
        "study_hours_per_week": [5, 10, 15, 20],
        "attendance_pct": [70, 80, 90, 100],
        "sleep_hours": [6, 7, 8, 9],
        "extra_classes": ["No", "Yes", "No", "Yes"],
        "final_score": [40, 50, 60, 70],
    })

