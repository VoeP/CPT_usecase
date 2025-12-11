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
X = df[features].copy()

ss = StandardScaler()
model0 = KNeighborsClassifier(n_neighbors = 5)
pipe_zero = Pipeline([("scale",ss),
                     ("knn", model0)])
pipe_zero.fit(X, y)

with open('geospatial_interpolation.pkl', 'wb') as f:
    pkl.dump(pipe_zero, f)


#############

############# fit seglearn model
import seglearn


mapping = {i:j for i,j in zip(segments_oi, range(0, len(segments_oi))) if i in segments_oi}
df["lithostrat_int"] = df.lithostrat_id.map(mapping)

features = ['diepte_mtaw','qc', 'fs', 'qtn', 'rf', 'fr', 'icn', 'sbt', 'ksbt']

# Construct Xt separately for train and test
def Xt_transform(df, cols):
    """Returns time series features for the seglern library"""
    Xt = df[cols].values.astype(np.float32)
    return Xt

X = df.copy()
#X.drop(["x","y","diepte"], inplace=True)
X= X[features]
X = Xt_transform(X, features)

y = df["lithostrat_int"]



pipe = Pype([('segment', SegmentX(width=5, overlap=0.1)),
    ('features', FeatureRep()),
    ('scaler', StandardScaler()),
    ('rf', RandomForestClassifier())])

pipe.fit([X], [y.values])
score = pipe.score([X], [y.values])
print(f"seglearn score : {score}")

with open('seglearn_rf.pkl', 'wb') as f:
    pkl.dump(pipe, f)