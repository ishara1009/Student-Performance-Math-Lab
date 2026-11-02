from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_performance_sample.csv"
FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
RESULTS_DIR = PROJECT_ROOT / "reports" / "results"
PASS_MARK = 50.0
RANDOM_STATE = 42
TEST_SIZE = 0.20
FEATURE_COLUMNS = [
    "study_hours_per_week", "attendance_pct", "sleep_hours", "extra_classes"
]
NUMERIC_COLUMNS = [
    "study_hours_per_week", "attendance_pct", "sleep_hours", "final_score"
]
REQUIRED_COLUMNS = ["student_id", *FEATURE_COLUMNS, "final_score"]

