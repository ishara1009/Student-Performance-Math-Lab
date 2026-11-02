import numpy as np
import pandas as pd
import streamlit as st


def render(results):
    training = results["training"]
    st.header("Prediction")
    study = st.number_input("Study hours per week", 0.0, 168.0, 10.0)
    attendance = st.number_input("Attendance (%)", 0.0, 100.0, 80.0)
    sleep = st.number_input("Sleep hours per day", 0.0, 24.0, 7.0)
    extra = st.selectbox("Extra classes", ["No", "Yes"])
    raw = np.array([[study, attendance, sleep, 1.0 if extra == "Yes" else 0.0]])
    scaled = training["standardizer"].transform(raw)
    prediction = float(training["model"].predict(scaled)[0])
    st.metric("Raw predicted score", f"{prediction:.2f}")
    if prediction < 0 or prediction > 100:
        st.warning("This raw linear-regression prediction is outside 0–100; it was not clipped.")
    st.dataframe(pd.DataFrame(training["metrics"]).T)
    st.caption("Educational demonstration only. This model must not be used for real student decisions.")

