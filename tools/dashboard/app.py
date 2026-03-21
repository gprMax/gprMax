import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="gprMax CI/CD Benchmarks", layout="wide")

st.title("gprMax Historical Regression & Benchmarks Dashboard")

st.markdown("""
This dashboard displays the historical performance of the gprMax simulator across environmental benchmarks,
including soil moisture variations, contamination plumes, and geotechnical inversions.
""")


@st.cache_data
def load_data():
    if os.path.exists("../../benchmark_results.csv"):
        return pd.read_csv("../../benchmark_results.csv")
    elif os.path.exists("benchmark_results.csv"):
        return pd.read_csv("benchmark_results.csv")
    else:
        st.warning("No benchmark data available. Showing mock data.")
        return pd.DataFrame(
            {
                "model": [
                    "soil_moisture.in",
                    "contamination_plume.in",
                    "geotech_inversion.in",
                    "soil_moisture.in",
                ],
                "runtime_s": [12.4, 15.1, 45.2, 11.8],
                "memory_mb": [150, 180, 500, 145],
                "success": [True, True, True, True],
                "grid_size": [1000, 1200, 5000, 1000],
            }
        )


df = load_data()

col1, col2 = st.columns(2)

with col1:
    st.subheader("Runtime by Model")
    fig1 = px.bar(
        df,
        x="model",
        y="runtime_s",
        color="success",
        hover_data=["memory_mb", "grid_size"],
    )
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.subheader("Memory Consumption vs Grid Size")
    fig2 = px.scatter(df, x="grid_size", y="memory_mb", color="model", size="runtime_s")
    st.plotly_chart(fig2, use_container_width=True)

st.subheader("Raw Benchmark Data")
st.dataframe(df)
