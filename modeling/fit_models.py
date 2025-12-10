import pandas as pd
import folium 
import geopandas as gpd
import numpy as np
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
import paths_cpt 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, RocCurveDisplay, silhouette_score
import os
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from seglearn.transform import FeatureRep, SegmentX
from seglearn.pipe import Pype
from seglearn.datasets import load_watch
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
import random
import pickle as pkl












############# geospatial section
df = pd.read_parquet(paths_cpt.PATH_TO_PARQUET)
print(df.shape)
df.dropna(inplace=True)
df.head()
train_set = df[~((df["lithostrat_id"].isna()) | (df["lithostrat_id"]=="None") | df["lithostrat_id"].str.contains('Onbekend'))]



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


features = ["x", "y", "diepte_mtaw"]

df = df[df["lithostrat_id"].isin(segments_oi)]
y = df["lithostrat_id"]
X = df[features]

ss = StandardScaler()
model0 = KNeighborsClassifier(n_neighbors = 5)
pipe_zero = Pipeline([("scale",ss),
                     ("knn", model0)])
pipe_zero.fit(X, y)

with open('geospatial_interpolation.pkl', 'wb') as f:
    pkl.dump(pipe_zero, f)


#############