# Student Performance Math Lab

A beginner-friendly portfolio project connecting linear algebra, calculus, statistics, hypothesis
testing, and from-scratch NumPy linear regression using **synthetic educational data**. Results
describe only the supplied sample and do not establish real-world educational effects.

## Features and Part 02 topic map

| Topic | Main source file |
|---|---|
| Vectors, matrices, dot products | `mathematics/linear_algebra.py` |
| Derivatives, MSE, gradients | `mathematics/calculus.py` |
| Mean, median, modes, variance, standard deviation | `statistics/descriptive.py` |
| Empirical and conditional probability | `statistics/probability.py` |
| Normal curve, Q-Q values, z-scores | `statistics/distributions.py` |
| Covariance and correlation | `statistics/relationships.py` |
| Welch hypothesis test and confidence interval | `statistics/hypothesis_testing.py` |
| NumPy gradient-descent regression | `models/linear_regression.py` |
| Split, scaling, comparison | `data/preprocessing.py`, `models/training.py` |

## Dataset

The supplied file belongs at `data/raw/student_performance_sample.csv`. Its schema is documented in
`data/README.md`. The ID is excluded from the model; the four behavioural columns are features and
`final_score` is the target. The original CSV is read-only from the application's point of view.

## Setup and commands

Python 3.11 or newer is required.

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
python scripts/run_analysis.py
python -m streamlit run dashboard/app.py
python -m pytest
python -m ruff check .
```

macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/run_analysis.py
python -m streamlit run dashboard/app.py
python -m pytest
python -m ruff check .
```

The analysis saves strict JSON/CSV files to `reports/results/` and PNG figures to
`reports/figures/`. Actual measured dataset results will be recorded here only after the supplied
CSV is available and the analysis has run; no result is fabricated.

## Suggested reading order

Start with `config.py`, then `validation.py`, `descriptive.py`, `calculus.py`,
`linear_regression.py`, `preprocessing.py`, `training.py`, and finally `pipeline.py`. Use the Sinhala
guide alongside those modules.

## Limitations

The data is synthetic, observational sample associations are not causal, linear regression assumes
a linear additive relationship, predictions can fall outside 0–100, and one fixed split gives an
incomplete view of generalisation. Welch's test also assumes independent observations and should
not be interpreted as the probability that its null hypothesis is true.

