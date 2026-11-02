import streamlit as st
from views.hypothesis_testing import render as render_hypothesis
from views.mathematics import render as render_mathematics
from views.overview import render as render_overview
from views.prediction import render as render_prediction
from views.statistics import render as render_statistics

from student_math_lab.pipeline import run_pipeline

st.set_page_config(page_title="Student Performance Math Lab", layout="wide")
st.title("Student Performance Math Lab")
st.caption("Synthetic educational data — sample findings do not establish real-world effects.")


@st.cache_resource
def cached_results():
    return run_pipeline()


try:
    results = cached_results()
except (FileNotFoundError, ValueError) as error:
    st.error(str(error))
    st.stop()

pages = {"Overview": render_overview, "Mathematics": render_mathematics,
         "Statistics": render_statistics, "Hypothesis Testing": render_hypothesis,
         "Prediction": render_prediction}
selected = st.sidebar.radio("View", list(pages))
pages[selected](results)

