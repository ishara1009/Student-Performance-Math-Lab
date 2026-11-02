import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def score_distribution_plot(scores: np.ndarray):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(scores, stat="density", kde=False, ax=axes[0])
    mean, std = np.mean(scores), np.std(scores, ddof=0)
    if std > 0:
        x_values = np.linspace(min(scores), max(scores), 200)
        axes[0].plot(x_values, stats.norm.pdf(x_values, mean, std), label="Fitted normal")
        axes[0].legend()
    axes[0].set_title("Final-score distribution")
    stats.probplot(scores, dist="norm", plot=axes[1])
    axes[1].set_title("Normal Q-Q plot")
    fig.tight_layout()
    return fig


def relationship_plots(data: pd.DataFrame, correlation: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.heatmap(correlation, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=axes[0])
    sns.scatterplot(data=data, x="study_hours_per_week", y="final_score", ax=axes[1])
    axes[1].set_title("Study hours vs final score")
    fig.tight_layout()
    return fig

