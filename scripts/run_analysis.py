import json

import pandas as pd

from student_math_lab.config import FIGURES_DIR, RESULTS_DIR
from student_math_lab.pipeline import run_pipeline
from student_math_lab.visualization.model_plots import loss_plot, prediction_plots
from student_math_lab.visualization.statistics_plots import (
    relationship_plots,
    score_distribution_plot,
)


def json_ready(value):
    """Convert nested NumPy/pandas values into strict JSON values."""
    if hasattr(value, "item"):
        return json_ready(value.item())
    if isinstance(value, dict):
        return {key: json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, float) and not (-float("inf") < value < float("inf")):
        return None
    return value


def main() -> None:
    results = run_pipeline()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    training = results["training"]
    relationships = results["relationships"]
    score_distribution_plot(results["data"]["final_score"].to_numpy()).savefig(
        FIGURES_DIR / "score_distribution.png", dpi=150
    )
    relationship_plots(results["data"], relationships["correlation"]).savefig(
        FIGURES_DIR / "relationships.png", dpi=150
    )
    loss_plot(training["model"].loss_history).savefig(FIGURES_DIR / "training_loss.png", dpi=150)
    prediction_plots(training["y_test"], training["predictions"]).savefig(
        FIGURES_DIR / "model_diagnostics.png", dpi=150
    )
    summary = {"descriptive": results["descriptive"], "probabilities": results["probabilities"],
               "hypothesis_test": results["hypothesis_test"]}
    (RESULTS_DIR / "statistics.json").write_text(
        json.dumps(json_ready(summary), indent=2, allow_nan=False), encoding="utf-8"
    )
    (RESULTS_DIR / "model_metrics.json").write_text(
        json.dumps(json_ready(training["metrics"]), indent=2, allow_nan=False), encoding="utf-8"
    )
    pd.DataFrame({"iteration": range(len(training["model"].loss_history)),
                  "training_mse": training["model"].loss_history}).to_csv(
        RESULTS_DIR / "training_loss.csv", index=False
    )
    test_rows = results["data"].iloc[training["test_indices"]]
    pd.DataFrame({"student_id": test_rows["student_id"].to_numpy(),
                  "actual_score": training["y_test"], "raw_predicted_score": training["predictions"],
                  "out_of_range": (training["predictions"] < 0) | (training["predictions"] > 100)}).to_csv(
        RESULTS_DIR / "test_predictions.csv", index=False
    )
    print("Analysis complete (synthetic educational data).")
    print(f"Students: {len(results['data'])}")
    print(f"Mean score: {results['descriptive']['mean']:.2f}")
    print(pd.DataFrame(training["metrics"]).T.to_string())
    print(f"Outputs saved under: {RESULTS_DIR.parent}")


if __name__ == "__main__":
    main()

