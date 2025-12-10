from dashboard_internals import *



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



def preprocessing(args):
    steps = "dashboard/dashboard_preparation.py"
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
            "Nearest segment"
        ]

    selected_category = st.sidebar.selectbox("Select category", options)




    #st.write(type(input))
    #st.write(input.head())


    if st.session_state.preprocessed:
        if st.button("Switch data"):
            st.session_state.preprocessed = False



    # We use teh selected category to show only data we want to see
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

        # Original rpedictions
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



