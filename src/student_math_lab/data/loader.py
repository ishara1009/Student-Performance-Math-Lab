from pathlib import Path

import pandas as pd

from student_math_lab.config import DATA_PATH


def load_student_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the supplied CSV without changing it."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found. Expected CSV at: {path}")
    return pd.read_csv(path)

