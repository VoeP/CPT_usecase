import sys
from pathlib import Path
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.dashboard_internals_jm import *

st.set_page_config(layout="wide")


segments_oi = [
"Quartair",
"Diest",
"Bolderberg",
"Sint_Huibrechts_Hern",
"Ursel",
"Asse",
"Wemmel",
"Lede",
"Brussel",
"Merelbeke",
"Kwatrecht",
"Mont_Panisel",
"Aalbeke",
"Mons_en_Pevele",
]



def add_dual_depth_axes(fig, df_cpt, feature_name):

    dmin = float(df_cpt["diepte"].min())
    dmax = float(df_cpt["diepte"].max())
    if dmax <= dmin:
        return fig

    n_ticks = 6  # TODO: adjust if needed
    tickvals = list(np.linspace(dmin, dmax, num=n_ticks))

    fig.update_layout(
        yaxis=dict(
            title="depth [m]",
            autorange="reversed",
            side="left",
            tickmode="array",
            tickvals=tickvals,
            ticktext=[f"{v:.1f}" for v in tickvals],
            showgrid=True,
            gridcolor="rgba(0,0,0,0.15)",
        )
    )

    depth_mtaw_col = None
    for c in df_cpt.columns:
        cl = c.lower()
        if cl == "diepte_mtaw" or cl.startswith("diepte_mta"):
            depth_mtaw_col = c
            break

    if depth_mtaw_col is None:
        return fig

    depth_mtaw_map = {}
    for tv in tickvals:
        idx = (df_cpt["diepte"] - tv).abs().idxmin()
        depth_mtaw_map[tv] = float(df_cpt.loc[idx, depth_mtaw_col])

    ticktext_mtaw = [f"{depth_mtaw_map[tv]:.2f}" for tv in tickvals]

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=1.02,
        y=1.03,
        text="depth mTAW [m]",
        showarrow=False,
        align="left",
        font=dict(size=10, color="#777777"),
    )

    for tv, txt in zip(tickvals, ticktext_mtaw):
        fig.add_annotation(
            xref="paper",
            yref="y",
            x=1.02,
            y=tv,
            text=txt,
            showarrow=False,
            align="left",
            font=dict(size=9, color="#777777"),
        )

    return fig


def detect_cpt_id_column(df):
    """
    TODO: extend this list if the data schema changes.
    """
    candidates = [
        "sondering_id",
        "sonderingid",
        "sondeernummer",
        "sondernr",
        "cpt_id",
        "cptid",
    ]

    lower_map = {c.lower(): c for c in df.columns}

    for name in candidates:
        if name in lower_map:
            return lower_map[name]

    return None



def preprocessing(args):
    steps = "dashboard/dashboard_preparation_jm.py"
    subprocess.run([sys.executable, steps, str(args)], check = True)



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


st.title("CPT Predictions")

def load_data(filename= None):
    if user_input != 'File':
        filename = 'input_t' +user_input + '.csv'
        input_df = data.DataSet("dashboard/input/"+filename,segments_oi, 0.05, dashboard=True).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
    else:
        input_df = data.DataSet("dashboard/input/input.csv",segments_oi, 0.05, dashboard=True).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
    closest = data.DataSet("dashboard/input/closest.csv",segments_oi, 0.05, dashboard=True).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
    return input_df, closest


if not st.session_state.preprocessed:
    st.warning("You must run preprocessing before accessing the dashboard tabs.")

    if st.button("Run preprocessing"):
        with st.spinner("Running preprocessing..."):
            preprocessing(user_input)
        st.success("Preprocessing finished!")

        input_df, closest = load_data()
        #st.write(type(input_df), input_df is None)
        #st.write(type(closest), closest is None)
        input_df, closest, copy_samp_gdp, min_samp_gdf, kdes, predictions = internal_logic(input_df, closest)
        st.session_state.preprocessed = True
        st.session_state.input = input_df
        st.session_state.closest = closest
        st.session_state.copy_samp_gdp = copy_samp_gdp
        st.session_state.min_samp_gdf = min_samp_gdf
        st.session_state.kdes = kdes
        st.session_state.predictions = predictions
        st.session_state.preprocessed = True


if st.session_state.preprocessed:
    input = st.session_state.input
    closest = st.session_state.closest
    copy_samp_gdp = st.session_state.copy_samp_gdp
    min_samp_gdf = st.session_state.min_samp_gdf
    kdes = st.session_state.kdes
    predictions = st.session_state.predictions

    #selected_category = st.sidebar.selectbox(
    #    "Select category",
    #    ["Predicted segments", "Segment correction", "Segment uncertainty"]
    #)

    #if not st.session_state.preprocessed:
    #    options = [
    #       "Predicted segments (locked)",
    #       "Segment correction (locked)",
    #       "Segment uncertainty (locked)"
    #   ]
    #else:
    options = [
            "Predicted segments",
            "Segment correction",
            #"Segment uncertainty",
            "Nearest segment",
            "Inspection",  # JM: extra tab for detailed plots
        ]

    selected_category = st.sidebar.selectbox("Select category", options)


    # modified for predicted vs truth numeric depth plots
    df_inspection_input = copy_samp_gdp.copy()

    # indices should line up between copy_samp_gdp and input (same rows)
    if len(df_inspection_input) == len(input):
        cols_to_copy = [
            c
            for c in [
                "diepte",
                "diepte_mtaw",
                "qc",
                "fs",
                "qtn",
                "rf",
                "fr",
                "icn",
                "sbt",
                "ksbt",
                "lithostrat_id",
                "predictions",
                "postprocessed",
                "final",
            ]
            if c in input.columns
        ]
        for c in cols_to_copy:
            df_inspection_input[c] = input[c]
    else:
        # TODO: handle mismatch case if this ever happens
        df_inspection_input = copy_samp_gdp


    #st.write(type(input))
    #st.write(input.head())


    if st.session_state.preprocessed:
        if st.button("Switch data"):
            st.session_state.preprocessed = False



    # We use the selected category to show only data we want to see
    if selected_category == "Predicted segments":
        #fig = go.Figure()

        fig = px.line(
            input,
            x="diepte",
            y="icn",
            color="postprocessed",
            title="Predicted segments",
        )

        st.plotly_chart(fig, key="postprocessed")




    #elif selected_category == "Segment uncertainty":
    elif 1==0:

        all_classes = np.unique(predictions)
        class_to_num = {cls: i for i, cls in enumerate(all_classes)}
        #cmap = plt.get_cmap("tab10", len(all_classes))

        classes = np.unique(predictions)
        depth_grid = np.linspace(input.diepte_mtaw.min(), input.diepte_mtaw.max(), len(input))
        density_matrix = np.zeros((len(classes), len(depth_grid)))

        for i, cls in enumerate(classes):
            cls_depths = input.loc[input['predictions'] == cls, 'diepte_mtaw']
            if len(cls_depths) > 1:
                kde = kdes[cls]
                density_matrix[i] = kde(depth_grid)
            else:
                density_matrix[i, np.abs(depth_grid - cls_depths[0]).argmin()] = 1.0

        fig = go.Figure()
        colors = px.colors.qualitative.T10
        class_to_num = {cls: i for i, cls in enumerate(classes)}

        # KDE lines
        print(density_matrix)
        print(depth_grid)
        for i, cls in enumerate(classes):
            fig.add_trace(go.Scatter(
                x=depth_grid,
                y=density_matrix[i],
                mode='lines',
                line=dict(color=colors[i % len(colors)], width=1.2),
                name=f'KDE {cls}'
            ))

        # Original predictions
        fig.add_trace(go.Scatter(
            x=input.diepte_mtaw,
            y=np.full(len(input), -5),
            mode='markers',
            marker=dict(
                color=[class_to_num[val] for val in predictions[::-1]],
                #colorscale='T10',
                symbol='x',
                size=8
            ),
            name='Predictions'
        ))

        # Smoothed labels
        fig.add_trace(go.Scatter(
            x=input.diepte_mtaw,
            y=np.full(len(input), -2.5),
            mode='markers',
            marker=dict(
                color=[class_to_num[val] for val in input["smoothed"][::-1]],
                #colorscale='T10',
                symbol='x',
                size=8
            ),
            name='Smoothed labels'
        ))

        fig.update_layout(
            #title="KDE Max Density (log-scaled, offset smoothed labels)",
            #xaxis_title="KDE Density (log scale)",
            #yaxis_title="Depth (mTAW)",
            #yaxis=dict(autorange="reversed"),
            #yaxis=dict(
            #    type="log",
            #    showgrid=True,
            #    zeroline=False
            #),
            template="plotly_white",
            height=500,
            width=700,
        )

        st.plotly_chart(fig)




    elif selected_category == "Segment correction": 
        

        boundaries_indices = input["postprocessed"] != input["postprocessed"].shift(1)
        boundaries_diepte = input.loc[boundaries_indices, "diepte"].tolist()
        segment_labels = input.loc[boundaries_indices, "postprocessed"].tolist()

        default = "\n".join([f"{label} : {diepte}" for label, diepte in zip(segment_labels, boundaries_diepte)])
        #boundaries = input["diepte"][input["postprocessed"].diff() != 0].tolist()

        boundaries_input = st.text_area(
            "Edit segment boundaries (separated by new lines, in the style of \n" \
            "Quartair : 2 \n" \
            "Merelbeke : 4",
            value=default,
        )

        labels = []
        boundaries = []

        for line in boundaries_input.split("\n"):
            name, value = line.split(":")
            name = name.strip()
            value = float(value.strip())
            labels.append(name)
            boundaries.append(value)


        if not boundaries: #always define boundaries
            boundaries = boundaries_diepte

        #segment_labels = input.loc[boundaries_indices, "postprocessed"].tolist()

        input["final"] = None
        for i, start in enumerate(boundaries):
            seg_value = labels[i]
            if i < len(boundaries) - 1:
                end = boundaries[i + 1]
            else:
                end = input["diepte"].max()
            mask = (input["diepte"] >= start) & (input["diepte"] < end)
            input.loc[mask, "final"] = seg_value
        input["final"] = input["final"].ffill() # this relates to a bugfix for the plots

        csv_data = input.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download edited data",
            data=csv_data,
            file_name="edited_output.csv",
            mime="text/csv"
        )

        col_main, col_right = st.columns([3, 2])


        #input = input.sort_values("diepte").reset_index(drop=True)
        with col_main:
            fig_main = px.line(
                input,
                x="icn",
                y="diepte",
                color="final",
                height = 1200
            )
            st.plotly_chart(fig_main, key="raw", use_container_width=True)



        with col_right:
            fig_postprocessed = px.line(
                input,
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
            plot_idx = 0
            height = 600
            width=600
            ind = 0
            for i, drilling_id in enumerate(drilling_ids):
                if ind > 3: # we add more drillings than 3 so we have to account for that with the plotting indices
                    break
                df = closest[closest["sondering_id"] == drilling_id]
                fig_side = px.line(
                    df,
                    x="icn",
                    y="diepte",
                    color="lithostrat_id",
                    title=f"Drilling {drilling_id}",
                    height = height,
                    width=width,
                )
                fig_side.update_layout(showlegend=False)

                col_idx = (i + 1) % 2
                row_idx = (i + 1) // 2
                if row_idx >= len(cols):
                    cols = st.columns(2)
                with cols[col_idx]:
                    st.plotly_chart(fig_side, use_container_width=True)


    elif selected_category == "Nearest segment":

            boundaries_indices = input["postprocessed"] != input["postprocessed"].shift(1)
            boundaries_diepte = input.loc[boundaries_indices, "diepte"].tolist()
            segment_labels = input.loc[boundaries_indices, "postprocessed"].tolist()

            default = "\n".join([f"{label} : {diepte}" for label, diepte in zip(segment_labels, boundaries_diepte)])
            #boundaries = input["diepte"][input["postprocessed"].diff() != 0].tolist()

            boundaries_input = st.text_area(
                "Edit segment boundaries (separated by new lines, in the style of \n" \
                "Quartair : 2 \n" \
                "Merelbeke : 4",
                value=default,
            )

            labels = []
            boundaries = []

            for line in boundaries_input.split("\n"):
                name, value = line.split(":")
                name = name.strip()
                value = float(value.strip())
                labels.append(name)
                boundaries.append(value)


            if not boundaries: #always define boundaries
                boundaries = boundaries_diepte

            #segment_labels = input.loc[boundaries_indices, "postprocessed"].tolist()

            input["final"] = None

            for i, start in enumerate(boundaries):
                seg_value = labels[i]
                if i < len(boundaries) - 1:
                    end = boundaries[i + 1]
                else:
                    end = input["diepte"].max()
                mask = (input["diepte"] >= start) & (input["diepte"] < end)
                input.loc[mask, "final"] = seg_value
            input["final"] = input["final"].ffill() # this relates to a bugfix for the plots


            st.subheader("Nearest Segment")
            col_left, col_right = st.columns(2)

            csv_data = input.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download edited data",
                data=csv_data,
                file_name="edited_output.csv",
                mime="text/csv"
            )

            unique_ids = closest["sondering_id"].unique()
            selected_id = st.selectbox("Select nearest drilling", unique_ids, index=0, key="nearest_select")

            filtered_closest = closest[closest["sondering_id"] == selected_id].copy()

            with col_left:
                fig_left = px.line(
                    input,
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


    elif selected_category == "Inspection":

        st.subheader("CPT inspection: ground truth vs predictions")

        data_source = st.radio(
            "Data source",
            ["Selected CPT (input)", "Nearest labelled CPTs"],
            horizontal=True,
        )

        if data_source.startswith("Selected"):
            df_source = df_inspection_input.copy()
        else:
            df_source = closest.copy()

        # try to detect a CPT id column (sondering_id / sondeernummer / etc.)

        cpt_col = detect_cpt_id_column(df_source)

        if cpt_col is None:
            st.info("No CPT id column (sondering_id / sondeernummer / ...) found → using entire dataset as one CPT.")
            df_cpt = df_source.copy()
            selected_id = "all_rows"
        else:
            cpt_ids = df_source[cpt_col].unique()
            cpt_ids = np.sort(cpt_ids)

            selected_id = st.selectbox(cpt_col, cpt_ids)
            df_cpt = df_source[df_source[cpt_col] == selected_id].copy()


        if df_cpt.empty:
            st.warning("No rows found for this CPT in the selected source.")
        else:
            candidate_features = ["qc", "fs", "qtn", "rf", "fr", "icn", "sbt", "ksbt"]
            available_features = [f for f in candidate_features if f in df_cpt.columns]

            if not available_features:
                st.warning("No numeric CPT features found (e.g. qc/fs/qtn).")
            else:
                feature = st.selectbox(
                    "Numeric CPT feature on x-axis",
                    available_features,
                )

                has_gt = (
                    "lithostrat_id" in df_cpt.columns
                    and df_cpt["lithostrat_id"].notna().any()
                )

                if "final" in df_cpt.columns and df_cpt["final"].notna().any():
                    pred_col = "final"
                elif (
                    "postprocessed" in df_cpt.columns
                    and df_cpt["postprocessed"].notna().any()
                ):
                    pred_col = "postprocessed"
                elif (
                    "predictions" in df_cpt.columns
                    and df_cpt["predictions"].notna().any()
                ):
                    pred_col = "predictions"
                else:
                    pred_col = None # TODO: maybe fall back to raw model output

                # decide what to plot: GT, prediction, or both
                if has_gt and pred_col is not None:
                    mode = st.radio(
                        "Labels to display",
                        ["Ground truth only", "Prediction only", "Ground truth vs prediction"],
                        index=2,
                    )
                elif has_gt:
                    mode = "Ground truth only"
                elif pred_col is not None:
                    mode = "Prediction only"
                else:
                    st.warning("No ground truth or prediction labels available for this CPT.")
                    mode = None

                # pick an id column if we found one; otherwise skip it in the table
                id_cols = [cpt_col] if (cpt_col is not None and cpt_col in df_cpt.columns) else []

                cols_to_show = [
                    c
                    for c in (
                        id_cols
                        + [
                            "diepte",
                            "diepte_mtaw",
                            feature,
                            "lithostrat_id",
                            pred_col,
                        ]
                    )
                    if c is not None and c in df_cpt.columns
                ]

                if mode is not None:

                    def make_line_fig(data, color_col, title):
                        data_sorted = data.sort_values("diepte")
                        fig = px.line(
                            data_sorted,
                            x=feature,
                            y="diepte",
                            color=color_col,
                            title=title,
                            height=700,
                        )
                        fig = add_dual_depth_axes(fig, data_sorted, feature_name=feature)
                        return fig

                    if mode == "Ground truth vs prediction" and has_gt and pred_col is not None:
                        # two plots side by side + table below
                        col_gt, col_pred = st.columns(2)

                        with col_gt:
                            df_gt = df_cpt[df_cpt["lithostrat_id"].notna()]
                            fig_gt = make_line_fig(
                                df_gt,
                                "lithostrat_id",
                                f"Ground truth vs {feature} (sondering_id={selected_id})",
                            )
                            st.plotly_chart(fig_gt, use_container_width=True)

                        with col_pred:
                            fig_pred = make_line_fig(
                                df_cpt,
                                pred_col,
                                f"Predicted segments ({pred_col}) vs {feature} (sondering_id={selected_id})",
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)

                        st.markdown("### Raw data for this CPT")
                        if cols_to_show:
                            st.dataframe(df_cpt[cols_to_show].head(500))
                        else:
                            st.info("No suitable columns selected for raw data table.")

                    else:
                        # single plot (either GT or prediction) + table on the right
                        plot_col, table_col = st.columns([3, 2])
                        with plot_col:
                            if mode == "Ground truth only" and has_gt:
                                df_gt = df_cpt[df_cpt["lithostrat_id"].notna()]
                                fig = make_line_fig(
                                    df_gt,
                                    "lithostrat_id",
                                    f"Ground truth vs {feature} (sondering_id={selected_id})",
                                )
                            else:  # "Prediction only"
                                fig = make_line_fig(
                                    df_cpt,
                                    pred_col,
                                    f"Predicted segments ({pred_col}) vs {feature} (sondering_id={selected_id})",
                                )
                            st.plotly_chart(fig, use_container_width=True)

                        with table_col:
                            st.markdown("### Raw data")
                            if cols_to_show:
                                st.dataframe(df_cpt[cols_to_show].head(500))
                            else:
                                st.info("No suitable columns selected for raw data table.")

