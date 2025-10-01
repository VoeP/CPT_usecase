import streamlit as st
import pandas as pd
import altair as alt
import paths_cpt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

df = pd.read_parquet(paths_cpt.PATH_TO_PARQUET)

one_samp = df[df["sondering_id"]==314]


signal = one_samp.icn.values
index = one_samp.index
trend = np.linspace(0, 10, len(index))
signal = trend + np.random.normal(0, 0.5, len(index))
#random_noise = np.random.normal(size = one_samp.shape[0])
random_noise=signal
print(random_noise, len(random_noise), one_samp.shape)
one_samp["noise"] = pd.Series(random_noise)
one_samp["noise_category"] = pd.cut(
    one_samp["noise"],
    bins=5,
    labels=[f"Cat {i}" for i in range(1, 6)])


categories = ["Predicted segments","Segment uncertainty"]
selected_category = st.sidebar.selectbox("Select category", categories)



if selected_category == "Predicted segments":
    #fig = go.Figure()

    fig = px.line(
        one_samp,
        x="index",
        y="icn",
        color="lithostrat_id",
        title="Predicted segments",
    )

    st.plotly_chart(fig, key="predictions")

else:
    #fig = go.Figure()

    fig = px.line(
        one_samp,
        x="index",
        y="icn",
        color="noise_category",
        title="Predicted segments",
    )

    st.plotly_chart(fig, key="uncertainty")
