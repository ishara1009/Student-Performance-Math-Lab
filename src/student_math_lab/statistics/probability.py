import pandas as pd


def empirical_probabilities(
    data: pd.DataFrame, pass_mark: float = 50.0, attendance_threshold: float = 80.0
) -> dict[str, dict[str, float | int | None]]:
    """Calculate empirical event and conditional probabilities."""
    total = len(data)
    if total == 0:
        raise ValueError("Probability calculations require a non-empty dataset.")
    passed = data["final_score"] >= pass_mark
    attended = data["attendance_pct"] >= attendance_threshold

    def result(numerator: int, denominator: int) -> dict[str, float | int | None]:
        return {"numerator": numerator, "denominator": denominator,
                "result": numerator / denominator if denominator else None}

    joint = int((passed & attended).sum())
    return {
        "pass": result(int(passed.sum()), total),
        "attendance": result(int(attended.sum()), total),
        "pass_and_attendance": result(joint, total),
        "pass_given_attendance": result(joint, int(attended.sum())),
    }

