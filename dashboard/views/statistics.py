import pandas as pd
import streamlit as st

from student_math_lab.visualization.statistics_plots import (
    relationship_plots,
    score_distribution_plot,
)


def render(results):
    st.header("Statistics")
    st.subheader("Descriptive statistics")
    st.json(results["descriptive"])
    st.caption("Population variance uses ddof=0; sample variance uses ddof=1.")
    st.subheader("Empirical probabilities")
    st.dataframe(pd.DataFrame(results["probabilities"]).T)
    distribution = results["distribution"]
    st.pyplot(score_distribution_plot(results["data"]["final_score"].to_numpy()))
    st.write("First 10 z-scores", distribution["z_scores"][:10])
    relationships = results["relationships"]
    st.write("Sample covariance", relationships["covariance"])
    st.write("Pearson correlation", relationships["correlation"])
    if relationships["constant_columns"]:
        st.warning(f"Constant columns have undefined correlations: {relationships['constant_columns']}")
    st.pyplot(relationship_plots(results["data"], relationships["correlation"]))
    st.caption("Covariance retains combined units; correlation is unitless. Correlation is not causation.")

