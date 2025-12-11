import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # one level up: repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle as pkl
import paths_cpt_jm as paths_cpt

SEGMENTS_OI = [
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

def train_geospatial_model():
    df = pd.read_parquet(paths_cpt.PATH_TO_PARQUET)

    features = ["x", "y", "diepte_mtaw"]
    df = df[df["lithostrat_id"].isin(SEGMENTS_OI)]
    y = df["lithostrat_id"]
    X = df[features]

    ss = StandardScaler()
    model0 = KNeighborsClassifier(n_neighbors=5)
    pipe0 = Pipeline([("scale", ss), ("knn", model0)])
    pipe0.fit(X, y)

    # save pkl
    out_path = paths_cpt.PATH_TO_GEOSPATIAL
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "wb") as f:
        pkl.dump(pipe0, f)

if __name__ == "__main__":
    train_geospatial_model()
