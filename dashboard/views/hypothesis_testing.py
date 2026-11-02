import streamlit as st


def render(results):
    test = results["hypothesis_test"]
    st.header("Hypothesis Testing")
    st.write(test["hypotheses"])
    st.json(test)
    st.info(test["interpretation"])
    st.caption("Welch's test assumes independent observations. A p-value is not P(H0 is true). "
               "The synthetic sample cannot establish that extra classes caused a difference.")

