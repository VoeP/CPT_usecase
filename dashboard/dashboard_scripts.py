import streamlit as st
import pandas as pd
import altair as alt
import paths_cpt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
# todo: if you change model type, the model needs to be imported again for pickle.load to work correctly
from sklearn.ensemble import RandomForestClassifier
import data_module_copy as data
import pickle
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium
import geopandas as gpd


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



def smooth_one_samp(pred_labels, depths):

    #ids = validation['sondering_id'].values
    all_classes = np.unique(pred_labels)
    class_to_kde = {cls: i for i, cls in enumerate(all_classes)}
    #cmap = plt.get_cmap("tab10", len(all_classes))

   # mask = ids == id
    d = depths
    l = pred_labels

    sort_idx = np.argsort(d)
    d_sorted = d
    l_sorted = l

    classes = np.unique(l_sorted)
    depth_grid = np.linspace(d_sorted.min(), d_sorted.max(), len(d_sorted))
    density_matrix = np.zeros((len(classes), len(depth_grid)))

    for i, cls in enumerate(classes):
        cls_depths = d_sorted[l_sorted == cls]
        if len(cls_depths) > 1:
            kde = gaussian_kde(cls_depths)
            class_to_kde[cls] = kde
            density_matrix[i] = kde(depth_grid)
        else:
            density_matrix[i, np.abs(depth_grid - cls_depths[0]).argmin()] = 1.0

    max_idx = np.argmax(density_matrix, axis=0)
    smoothed_local = classes[max_idx]

    smoothed_labels = smoothed_local[np.argsort(sort_idx)]

    return smoothed_labels, class_to_kde






theoretical_ordering = {"Quartair":1,
"Diest":2,
"Bolderberg":3,
"Sint_Huibrechts_Hern":4,
"Ursel":5,
"Asse":6,
"Wemmel":7,
"Lede":8,
"Brussel":9,
"Merelbeke":10,
"Kwatrecht":11,
"Mont_Panisel":12,
"Aalbeke":13,
"Mons_en_Pevele":14}

def correct_labels(labels, seq=None, min_length=20):
    clean_run = False
    if len(labels) == 0:
        return [], [], []

    corrected = []
    observed = [labels[0]]
    lengths = []
    current_label = labels[0]
    run_length = 1

    for i in range(1, len(labels)):
            if labels[i] == current_label:
                run_length += 1
            else:
                corrected.append([current_label, run_length])
                lengths.append(run_length)
                current_label = labels[i]
                observed.append(current_label)
                run_length = 1

    corrected.append([current_label, run_length])
    lengths.append(run_length)
    
    while not clean_run:
        problems = [False, False, False, False]


        # merging too short segments
        i = 1
        while i < len(corrected):
            label, length = corrected[i]
            if length < min_length:
                corrected[i-1][1] += length
                corrected.pop(i)
                i -= 1
                problems[0] = True
            i += 1
        i = 1
        while i < len(corrected)-1:
            #if theoretical_ordering[observed[i-1]]< theoretical_ordering[observed[i]]:
                prev, p_len = corrected[i-1]
                curr, c_len = corrected[i]
                next, n_len = corrected[i+1]
                if prev == next:
                    corrected[i-1][1] = p_len + c_len + n_len
                    corrected.pop(i+1)
                    corrected.pop(i)
                    problems[1] = True
                i += 1
        i = 1
        while i < len(corrected):
            #check if adjacent are equal
                prev, p_len = corrected[i-1]
                curr, c_len = corrected[i]
                if prev == curr:
                    corrected[i-1][1] = p_len + c_len
                    corrected.pop(i)
                    problems[2]=True
                i += 1
        i = 1
        while i < len(corrected)-2:
            #if theoretical_ordering[observed[i-1]]< theoretical_ordering[observed[i]]:
                prev, p_len = corrected[i-1]
                curr, c_len = corrected[i]
                next, n_len = corrected[i+1]
                next_2, n_len_2 = corrected[i+2]
                if prev == next_2:
                    corrected[i-1][1] = p_len + c_len + n_len + n_len_2
                    corrected.pop(i+2)
                    corrected.pop(i+1)
                    corrected.pop(i)
                    problems[3] = True
                i += 1


        if not any(problems):
             clean_run = True

    not_clean = True
    while not_clean:
        second_round_prob = [False]
        i=1
        while i < len(corrected):
            
            prev, p_len = corrected[i-1]
            curr, c_len = corrected[i]
            if theoretical_ordering[curr] < theoretical_ordering[prev]:
                #corrected[i-1][0] = curr
                print("order correction occurred for : ",corrected[i])
                print("full seq is : ", corrected)
                corrected[i][0] = prev
                second_round_prob[0] = True
            i += 1

        if not any(second_round_prob):
             not_clean = False

    expanded = []
    for label, length in corrected:
        expanded.extend([label] * length)                     

    return expanded




input = data.DataSet("dashboard/input/input.csv",segments_oi, 0.05).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
closest = data.DataSet("dashboard/input/closest.csv",segments_oi, 0.05).get_preprocessed(split=False, features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])

input.drop(["Unnamed: 0"], axis=1, inplace=True)
closest.drop(["Unnamed: 0"], axis=1, inplace=True)
#train, validation = dset.get_preprocessed(features =  ['qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt'])
#validation.dropna(inplace=True)

#we take a known sample for dashboard drafting
#rand = list(set(validation.sondering_id))[0]
#one_samp = validation[validation["sondering_id"]==rand]

## we load the finished model here (change later, keep used libraries at imports)
with open(paths_cpt.PATH_TO_MODEL, 'rb') as f:
    exported_model = pickle.load(f)

# we only take features that were used to train our model here:
#features = ['diepte',
#       'qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt']


copy_samp = input.copy()

input.drop(["sondering_id",	"index",	"pkey_sondering",	"sondeernummer",
              	"x", "y", "start_sondering_mtaw",	
                "diepte_sondering_tot", "lithostrat_id"], axis=1, inplace=True)

predictions = exported_model.predict(input)

# we create some predictions
input["predictions"] = predictions

input["postprocessed"] = correct_labels(predictions)



signal = input.icn.values
index = input.index
trend = np.linspace(0, 10, len(index))
signal = trend + np.random.normal(0, 0.5, len(index))
#random_noise = np.random.normal(size = one_samp.shape[0])
random_noise=signal
print(random_noise, len(random_noise), input.shape)
#one_samp["noise"] = pd.Series(random_noise)
#one_samp["noise_category"] = pd.cut(
#    one_samp["noise"],
#    bins=5,
#    labels=[f"Cat {i}" for i in range(1, 6)])


categories = ["Predicted segments","Nearest segment","Segment uncertainty",]
selected_category = st.sidebar.selectbox("Select category", categories)

print(copy_samp[["x","y"]])

#train["x_dist"] = train["x"].astype(float) - copy_samp["x"].astype(float).values[0]

#train["y_dist"] = train["y"].astype(float) - copy_samp["y"].astype(float).values[0]

#train["dist"] = np.sqrt(train["x_dist"]**2 + train["y_dist"]**2)
#print(train[["x_dist", "y_dist"]])
#filtered = train.dropna()
#print(filtered)

#min_sample = filtered[filtered["dist"] == filtered["dist"].min()].sondering_id.unique()[0]
#min_sample = filtered[filtered["sondering_id"]==min_sample]


input["smoothed"], kdes = smooth_one_samp(predictions, input.diepte_mtaw)


min_samp_gdf = gpd.GeoDataFrame(
    closest,
    geometry=gpd.points_from_xy(closest["x"], closest["y"]),
    crs="EPSG:31370"
)
min_samp_gdf = min_samp_gdf.to_crs("EPSG:4326")
min_samp_gdf["lon"] = min_samp_gdf.geometry.x
min_samp_gdf["lat"] = min_samp_gdf.geometry.y



copy_samp_gdp = gpd.GeoDataFrame(
    copy_samp,
    geometry=gpd.points_from_xy(copy_samp["x"], copy_samp["y"]),
    crs="EPSG:31370"
)
copy_samp_gdp = copy_samp_gdp.to_crs("EPSG:4326")
copy_samp_gdp["lon"] = copy_samp_gdp.geometry.x
copy_samp_gdp["lat"] = copy_samp_gdp.geometry.y

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




elif selected_category == "Segment uncertainty":

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



elif selected_category == "Nearest segment":
    #fig = go.Figure()

    fig = px.line(
        closest,
        x="diepte",
        y="icn",
        color="lithostrat_id",
        title="Sample from closest drilling",
    )

    st.plotly_chart(fig, key="predictions")

    center_lat = (min_samp_gdf.lat.mean() + copy_samp_gdp.lat.mean()) / 2
    center_lon = (min_samp_gdf.lon.mean() + copy_samp_gdp.lon.mean()) / 2

    m = folium.Map(location=[center_lat, center_lon], zoom_start=14)
    
    folium.CircleMarker(
            location=[min_samp_gdf.lat.mean(), min_samp_gdf.lon.mean()],
            radius=6,
            color="blue",
            fill=True,
            fill_color="blue",
        ).add_to(m)
    
    folium.CircleMarker(
            location=[copy_samp_gdp.lat.mean(), copy_samp_gdp.lon.mean()],
            radius=6,
            color="red",
            fill=True,
            fill_color="red",
        ).add_to(m)

    st_folium(m, width=700, height=500)

