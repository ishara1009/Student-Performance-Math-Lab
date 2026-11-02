import streamlit as st


def render(results):
    data, descriptive = results["data"], results["descriptive"]
    st.header("Overview")
    st.warning("This dataset is synthetic and intended only for education.")
    first, second, third = st.columns(3)
    first.metric("Students", len(data))
    second.metric("Mean score", f"{descriptive['mean']:.2f}")
    third.metric("Pass rate", f"{results['probabilities']['pass']['result']:.1%}")
    st.dataframe(data.head())
    st.markdown("**Columns:** ID; weekly study hours; attendance percentage; daily sleep hours; "
                "extra-classes Yes/No; final score (0–100). IDs are never model features.")

