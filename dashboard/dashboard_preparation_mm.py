import pandas as pd
import os
import sys
from pathlib import Path

# Define paths using pathlib
# BASE_DIR is the directory containing this script (dashboard/)
BASE_DIR = Path(__file__).resolve().parent
INPUT_FOLDER = BASE_DIR / "input"

# Assuming the 'data' folder is in the project root (one level up from dashboard/)
PARQUET_PATH = BASE_DIR.parent / "data" / "vw_cpt_brussels_params_completeset_20250318_remapped.parquet"

def preprocessing(user_choice):
    if not PARQUET_PATH.exists():
        raise FileNotFoundError(f"Parquet file not found at: {PARQUET_PATH}")

    df = pd.read_parquet(PARQUET_PATH, engine="fastparquet")
    sonderings = df["sondering_id"].unique()
    
    # Ensure input folder exists
    INPUT_FOLDER.mkdir(exist_ok=True)

    if user_choice == "File":
        # Use pathlib to list files
        csv_files = [
            f.name for f in INPUT_FOLDER.glob("*.csv")
            if f.name != "closest.csv" and "input_t" not in f.name
        ]
        if not csv_files:
            raise FileNotFoundError(f"Please paste a file in {INPUT_FOLDER} or provide an ID.")
        
        input_path = INPUT_FOLDER / csv_files[0]
        input = pd.read_csv(input_path)
    else:
        input = df[df["sondering_id"] == int(user_choice)]
        name = f"input_t{user_choice}.csv"
        input.to_csv(INPUT_FOLDER / name)
    
    if input.empty:
        print(f"Error: No data found for selection {user_choice}")
        sys.exit(1)

    x = input["x"].iloc[0]
    y = input["y"].iloc[0]

    df["dist"] = (df["x"] - x)**2 + (df["y"] - y)**2

    # Ensure lithostrat_id is string to avoid type errors
    df["lithostrat_id"] = df["lithostrat_id"].astype(str)

    df_nomissing = df[~((df["lithostrat_id"] == "None") | 
                        (df["lithostrat_id"] == "nan") | 
                        df["lithostrat_id"].str.contains('Onbekend', case=False) | 
                        df["lithostrat_id"].str.contains('nan', case=False))]
    
    closest_ids = df_nomissing.groupby("sondering_id")["dist"].min().nsmallest(15).index.tolist()
    closest = df[df["sondering_id"].isin(closest_ids)]
    
    closest_path = INPUT_FOLDER / "closest.csv"
    closest.to_csv(closest_path, index=False)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_choice = sys.argv[1]  # get argument from Streamlit
        preprocessing(user_choice)