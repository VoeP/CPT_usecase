import streamlit as st
import pandas as pd
import altair as alt
import paths_cpt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import subprocess
import sys
import os
from pathlib import Path
import joblib
import io
import base64

# Add modeling directory to path to import data_processing
sys.path.append(str(Path(__file__).parent.parent / "modeling"))
try:
    import data_processing as dp
except ImportError:
    st.error("Could not import data_processing from modeling folder.")

from dashboard_internals import *

# --- App.py Configuration ---
MODEL_PATH = Path(__file__).parent.parent / "results" / "models" / "best_rf_model.pkl"
LE_PATH = Path(__file__).parent.parent / "results" / "models" / "label_encoder.pkl"
BIN_W = 0.6
EXTRACT_TREND = True
TREND_TYPE = "multiplicative"

# --- App.py Helper Functions ---
def get_layers_from_bins(df):
    """
    Converts binned predictions into continuous layers for plotting.
    Expects df to have 'depth_bin' (Interval) and 'predicted_label'.
    """
    layers = []
    if df.empty:
        return layers
    
    # Ensure depth_bin is Interval
    if isinstance(df['depth_bin'].iloc[0], str):
        # simplistic parsing of pandas interval string "(a, b]"
        df['top'] = df['depth_bin'].apply(lambda x: float(x.strip('()[]').split(',')[0]))
        df['bottom'] = df['depth_bin'].apply(lambda x: float(x.strip('()[]').split(',')[1]))
    else:
        df['top'] = df['depth_bin'].apply(lambda x: x.left)
        df['bottom'] = df['depth_bin'].apply(lambda x: x.right)

    df = df.sort_values('top')
    
    current_label = None
    start_depth = None
    
    for _, row in df.iterrows():
        label = row['predicted_label']
        if label != current_label:
            if current_label is not None:
                layers.append({
                    'label': current_label,
                    'top': start_depth,
                    'bottom': row['top']
                })
            current_label = label
            start_depth = row['top']
    
    # Add last layer
    if current_label is not None:
        layers.append({
            'label': current_label,
            'top': start_depth,
            'bottom': row['bottom'] # Use bottom of last bin
        })
    
    return pd.DataFrame(layers)

@st.cache_resource
def load_new_model():
    try:
        model = joblib.load(MODEL_PATH)
        le = joblib.load(LE_PATH)
        return model, le
    except FileNotFoundError:
        return None, None

# --- Existing Dashboard Logic ---
segments_oi = [
"Quartair", "Diest", "Bolderberg", "Sint_Huibrechts_Hern", "Ursel",
"Asse", "Wemmel", "Lede", "Brussel", "Merelbeke", "Kwatrecht",
"Mont_Panisel", "Aalbeke", "Mons_en_Pevele",
]

def preprocessing(args):
    steps = "dashboard/dashboard_preparation.py"
    subprocess.run([sys.executable, steps, str(args)], check = True)

def load_data(filename= None, user_input='File'):
    if user_input != 'File':
        filename = 'input_t' +user_input + '.csv'
        input_df = data.DataSet("dashboard/input/"+filename,segments_oi, 0.05, dashboard=True).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
    else:
        input_df = data.DataSet("dashboard/input/input.csv",segments_oi, 0.05, dashboard=True).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
    closest = data.DataSet("dashboard/input/closest.csv",segments_oi, 0.05, dashboard=True).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
    return input_df, closest

# --- Main App ---
st.title("CPT Predictions Dashboard")

mode = st.sidebar.radio("Select Mode", ["Existing Dashboard", "New Model (App.py Integration)"])

if mode == "Existing Dashboard":
    if "preprocessed" not in st.session_state:
        st.session_state.preprocessed = False
    if "input" not in st.session_state:
        st.session_state.input = None
    if "closest" not in st.session_state:
        st.session_state.closest = None

    user_input = st.text_input(
        "Enter either 'File' (csv file from the 'input' folder) or the numeric id of the drilling you wish to use:",
        value="File"
    )

    if not st.session_state.preprocessed:
        st.warning("You must run preprocessing before accessing the dashboard tabs.")

        if st.button("Run preprocessing"):
            with st.spinner("Running preprocessing..."):
                preprocessing(user_input)
            st.success("Preprocessing finished!")

            input_df, closest = load_data(user_input=user_input)
            input_df, closest, copy_samp_gdp, min_samp_gdf, kdes, predictions = internal_logic(input_df, closest)
            st.session_state.preprocessed = True
            st.session_state.input = input_df
            st.session_state.closest = closest
            st.session_state.copy_samp_gdp = copy_samp_gdp
            st.session_state.min_samp_gdf = min_samp_gdf
            st.session_state.kdes = kdes
            st.session_state.predictions = predictions

    if st.session_state.preprocessed:
        input_df = st.session_state.input
        closest = st.session_state.closest
        copy_samp_gdp = st.session_state.copy_samp_gdp
        min_samp_gdf = st.session_state.min_samp_gdf
        kdes = st.session_state.kdes
        predictions = st.session_state.predictions

        options = [
                "Predicted segments",
                "Segment correction",
                "Nearest segment"
            ]

        selected_category = st.sidebar.selectbox("Select category", options)

        if st.button("Switch data"):
            st.session_state.preprocessed = False

        if selected_category == "Predicted segments":
            fig = px.line(
                input_df,
                x="diepte",
                y="icn",
                color="postprocessed",
                title="Predicted segments",
            )
            st.plotly_chart(fig, key="postprocessed")

        elif selected_category == "Segment correction":
            boundaries_indices = input_df["postprocessed"] != input_df["postprocessed"].shift(1)
            boundaries_diepte = input_df.loc[boundaries_indices, "diepte"].tolist()
            segment_labels = input_df.loc[boundaries_indices, "postprocessed"].tolist()

            default = "\n".join([f"{label} : {diepte}" for label, diepte in zip(segment_labels, boundaries_diepte)])

            boundaries_input = st.text_area(
                "Edit segment boundaries (separated by new lines, in the style of \n" \
                "Quartair : 2 \n" \
                "Merelbeke : 4",
                value=default,
            )

            labels = []
            boundaries = []

            for line in boundaries_input.split("\n"):
                if ":" in line:
                    name, value = line.split(":")
                    name = name.strip()
                    value = float(value.strip())
                    labels.append(name)
                    boundaries.append(value)

            if not boundaries:
                boundaries = boundaries_diepte

            input_df["final"] = None
            for i, start in enumerate(boundaries):
                seg_value = labels[i] if i < len(labels) else labels[-1]
                if i < len(boundaries) - 1:
                    end = boundaries[i + 1]
                else:
                    end = input_df["diepte"].max()
                mask = (input_df["diepte"] >= start) & (input_df["diepte"] < end)
                input_df.loc[mask, "final"] = seg_value
            input_df["final"] = input_df["final"].ffill()

            csv_data = input_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download edited data",
                data=csv_data,
                file_name="edited_output.csv",
                mime="text/csv"
            )

            col_main, col_right = st.columns([3, 2])

            with col_main:
                fig_main = px.line(
                    input_df,
                    x="icn",
                    y="diepte",
                    color="final",
                    height = 1200
                )
                st.plotly_chart(fig_main, key="raw", use_container_width=True)

            with col_right:
                fig_postprocessed = px.line(
                    input_df,
                    x="icn",
                    y="diepte",
                    color="postprocessed",
                    title="Predicted",
                    height = 600
                )
                fig_postprocessed.update_layout(showlegend=False)
                cols = st.columns(2)
                with cols[0]:
                    st.plotly_chart(fig_postprocessed, use_container_width=True)

                drilling_ids = closest["sondering_id"].unique()[:3]
                ind = 0
                for i, drilling_id in enumerate(drilling_ids):
                    if ind > 3:
                        break
                    df = closest[closest["sondering_id"] == drilling_id]
                    fig_side = px.line(
                        df,
                        x="icn",
                        y="diepte",
                        color="lithostrat_id",
                        title=f"Drilling {drilling_id}",
                        height = 600,
                        width=600,
                    )
                    fig_side.update_layout(showlegend=False)

                    col_idx = (i + 1) % 2
                    row_idx = (i + 1) // 2
                    if row_idx >= len(cols):
                        cols = st.columns(2)
                    with cols[col_idx]:
                        st.plotly_chart(fig_side, use_container_width=True)

        elif selected_category == "Nearest segment":
            # Reuse logic from Segment correction for boundaries
            boundaries_indices = input_df["postprocessed"] != input_df["postprocessed"].shift(1)
            boundaries_diepte = input_df.loc[boundaries_indices, "diepte"].tolist()
            segment_labels = input_df.loc[boundaries_indices, "postprocessed"].tolist()
            default = "\n".join([f"{label} : {diepte}" for label, diepte in zip(segment_labels, boundaries_diepte)])
            
            boundaries_input = st.text_area("Edit segment boundaries", value=default, key="nearest_text_area")
            
            labels = []
            boundaries = []
            for line in boundaries_input.split("\n"):
                if ":" in line:
                    name, value = line.split(":")
                    name = name.strip()
                    value = float(value.strip())
                    labels.append(name)
                    boundaries.append(value)
            
            if not boundaries: boundaries = boundaries_diepte
            
            input_df["final"] = None
            for i, start in enumerate(boundaries):
                seg_value = labels[i] if i < len(labels) else labels[-1]
                if i < len(boundaries) - 1:
                    end = boundaries[i + 1]
                else:
                    end = input_df["diepte"].max()
                mask = (input_df["diepte"] >= start) & (input_df["diepte"] < end)
                input_df.loc[mask, "final"] = seg_value
            input_df["final"] = input_df["final"].ffill()

            st.subheader("Nearest Segment")
            col_left, col_right = st.columns(2)

            unique_ids = closest["sondering_id"].unique()
            selected_id = st.selectbox("Select nearest drilling", unique_ids, index=0, key="nearest_select")

            filtered_closest = closest[closest["sondering_id"] == selected_id].copy()

            with col_left:
                fig_left = px.line(
                    input_df,
                    x="icn",
                    y="diepte",
                    color="final",
                    title="Selected nearest drilling",
                    height = 800
                )
                st.plotly_chart(fig_left, key="nearest_left_plot")

            with col_right:
                fig_right = px.line(
                    filtered_closest,
                    x="icn",
                    y="diepte",
                    color="lithostrat_id",
                    title=f"Drilling id : {selected_id}",
                    height = 800
                )
                st.plotly_chart(fig_right, key="nearest_right_plot")

            center_lat = (min_samp_gdf.lat.mean() + copy_samp_gdp.lat.mean()) / 2
            center_lon = (min_samp_gdf.lon.mean() + copy_samp_gdp.lon.mean()) / 2

            m = folium.Map(location=[center_lat, center_lon], zoom_start=15)

            folium.CircleMarker(
                location=[min_samp_gdf.lat.mean(), min_samp_gdf.lon.mean()],
                radius=6,
                color="blue",
                fill=True,
                fill_color="blue",
                tooltip=f"sondering_id: {min_samp_gdf['sondering_id'].iloc[0]}",
            ).add_to(m)

            ids_on_map = closest["sondering_id"].unique()

            for sid in ids_on_map:
                df_s = min_samp_gdf[min_samp_gdf["sondering_id"] == sid]
                folium.CircleMarker(
                    location=[df_s.lat.mean(), df_s.lon.mean()],
                    radius=5,
                    color="green" if sid != selected_id else "red",
                    fill=True,
                    fill_color="green" if sid != selected_id else "red",
                    tooltip=f"sondering_id: {sid}",
                ).add_to(m)

            st_folium(m, width=1000, height=1000)

elif mode == "New Model (App.py Integration)":
    st.header("New Model Prediction (from App.py)")
    
    model, le = load_new_model()
    
    if model is None:
        st.error(f"Model not found at {MODEL_PATH}. Please run the modeling notebook.")
    else:
        st.success("Model loaded successfully.")
        
        uploaded_file = st.file_uploader("Upload CPT Data (Parquet or CSV)", type=["parquet", "csv"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.parquet'):
                    df = pd.read_parquet(uploaded_file, engine='fastparquet')
                else:
                    df = pd.read_csv(uploaded_file)
                
                st.write("Data loaded:", df.head())
                
                if st.button("Run Prediction"):
                    with st.spinner("Processing and Predicting..."):
                        # Ensure sondering_id exists
                        if "sondering_id" not in df.columns:
                            df["sondering_id"] = "uploaded_file"
                        
                        # Process data
                        processed_df = dp.process_test_train(
                            df,
                            sondering_ids=df["sondering_id"].unique(),
                            do_extract_trend=EXTRACT_TREND,
                            bin_w=BIN_W,
                            trend_type=TREND_TYPE
                        )
                        
                        if processed_df.empty:
                            st.error("Processing returned empty dataframe.")
                        else:
                            # Predict
                            feature_cols = [c for c in processed_df.columns if c not in ["sondering_id", "depth_bin", "lithostrat_id", "QC_raw"]]
                            
                            try:
                                if hasattr(model, "feature_names_in_"):
                                    X = processed_df[model.feature_names_in_]
                                else:
                                    # Fallback: drop non-features
                                    X = processed_df.drop(columns=["sondering_id", "depth_bin", "lithostrat_id", "QC_raw"], errors="ignore")
                                    # Also drop object columns
                                    X = X.select_dtypes(include=[np.number])
                                
                                preds = model.predict(X)
                                processed_df["predicted_label"] = le.inverse_transform(preds)
                                
                                st.session_state.new_results = processed_df
                                st.success("Prediction complete!")
                                
                            except Exception as e:
                                st.error(f"Prediction failed: {e}")
                                
            except Exception as e:
                st.error(f"Error loading file: {e}")

        if "new_results" in st.session_state:
            results = st.session_state.new_results
            
            # Visualization
            st.subheader("Results Visualization")
            
            # Convert bins to layers for plotting
            layers_df = get_layers_from_bins(results)
            
            # Plotly Figure
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("CPT Data", "Predicted Stratigraphy"))
            
            # Add CPT traces (qc, fs) - simplified
            # We need the original depth for this, but results is binned. 
            # We can plot the binned values or the original if we kept it.
            # Let's plot binned qc/fs
            
            fig.add_trace(go.Scatter(x=results["qc_mean"], y=results["depth_bin"].apply(lambda x: x.mid), name="qc (mean)"), row=1, col=1)
            fig.add_trace(go.Scatter(x=results["fs_mean"], y=results["depth_bin"].apply(lambda x: x.mid), name="fs (mean)"), row=1, col=1)
            
            # Add Stratigraphy
            # We use a bar chart or shapes for layers
            for _, row in layers_df.iterrows():
                fig.add_shape(
                    type="rect",
                    x0=0, x1=1,
                    y0=row["top"], y1=row["bottom"],
                    fillcolor="lightgrey", # You can map colors to labels
                    opacity=0.5,
                    line_width=0,
                    row=1, col=2
                )
                fig.add_annotation(
                    x=0.5, y=(row["top"] + row["bottom"])/2,
                    text=row["label"],
                    showarrow=False,
                    row=1, col=2
                )
            
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig)
            
            # Editable Table
            st.subheader("Edit Predictions")
            edited_df = st.data_editor(layers_df)
            
            if st.button("Save Edits"):
                st.write("Edits saved (in memory).")
                st.dataframe(edited_df)
