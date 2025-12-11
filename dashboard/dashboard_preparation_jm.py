import sys
from pathlib import Path

# Make repo root importable
ROOT = Path(__file__).resolve().parents[1]  # dashboard/.. = repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import numpy as np
import os
import pickle as pkl
import paths_cpt_jm as paths_cpt


BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "input"


def preprocessing(user_choice: str):
    path_to_parquet = paths_cpt.PATH_TO_PARQUET
    if not Path(path_to_parquet).exists():
        raise FileNotFoundError(f"Parquet file not found at: {path_to_parquet}")

    df = pd.read_parquet(path_to_parquet)

    INPUT_FOLDER.mkdir(exist_ok=True)

    geo_path = paths_cpt.PATH_TO_GEOSPATIAL
    if not Path(geo_path).exists():
        raise FileNotFoundError(
            f"Geospatial model not found at: {geo_path}\n"
            f"Run fit_models_jm.py first to create it."
        )

    with open(geo_path, "rb") as f:
        geospatial = pkl.load(f)
    
    #pick_one = df[df["sondering_id"] == sonderings[0]]
    #pick_one.to_csv("input/input.csv", index=False)

    if user_choice == "File":
        csv_files = [
            f.name
            for f in INPUT_FOLDER.glob("*.csv")
            if f.name != "closest.csv" and "input_t" not in f.name
        ]
        if not csv_files:
            raise FileNotFoundError("Please paste a file in input/ or provide an ID.")
        input_path = INPUT_FOLDER / csv_files[0]
        input_df = pd.read_csv(input_path)
    else:
        sondering_id = int(user_choice)
        input_df = df[df["sondering_id"] == sondering_id].copy()
        if input_df.empty:
            raise ValueError(f"No rows found for sondering_id={sondering_id}")
        name = f"input_t{sondering_id}.csv"
        input_df.to_csv(INPUT_FOLDER / name, index=False)

    interpolated = geospatial.predict(input_df[["x", "y", "diepte_mtaw"]])

    interpolated_path = INPUT_FOLDER / "interpolated.txt"
    with open(interpolated_path, "w") as f:
        for label in interpolated:
            f.write(str(label) + "\\n")

    x = input_df["x"].iloc[0]
    y = input_df["y"].iloc[0]

    df["dist"] = (df["x"] - x) ** 2 + (df["y"] - y) ** 2
    df["lithostrat_id"] = df["lithostrat_id"].astype(str)

    df_nomissing = df[~(
        (df["lithostrat_id"] == "None")
        | (df["lithostrat_id"] == "nan")
        | df["lithostrat_id"].str.contains("Onbekend", case=False)
        | df["lithostrat_id"].str.contains("nan", case=False)
    )]

    closest_ids = (
        df_nomissing.groupby("sondering_id")["dist"]
        .min()
        .nsmallest(10)
        .index
        .tolist()
    )
    closest = df[df["sondering_id"].isin(closest_ids)]

    closest_path = INPUT_FOLDER / "closest.csv"
    closest.to_csv(closest_path, index=False)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        preprocessing(sys.argv[1])
