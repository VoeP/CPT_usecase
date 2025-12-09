import pandas as pd
import os

path = "C:/Users/volte/Downloads/vw_cpt_brussels_params_completeset_20250318_remapped.parquet" # change this
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_FOLDER = os.path.join(BASE_DIR, "input")

def preprocessing(user_choice):
    path_to_parquet = "C:/Users/volte/Downloads/vw_cpt_brussels_params_completeset_20250318_remapped.parquet"
    df = pd.read_parquet(path_to_parquet)
    sonderings = df["sondering_id"].unique()
    #pick_one = df[df["sondering_id"] == sonderings[0]]
    #pick_one.to_csv("input/input.csv", index=False)
    if user_choice == "File":
        csv_files = [
            f for f in os.listdir(INPUT_FOLDER)
            if f.endswith(".csv") and f != "closest.csv" and "input_t" not in f
        ]
        if not csv_files:
            raise FileNotFoundError("Please paste a file in input/ or provide an ID.")
        input_path = os.path.join(INPUT_FOLDER, csv_files[0])
        input = pd.read_csv(input_path)
    else:
        input = df[df["sondering_id"] == int(user_choice)]
        name = "input_t" + user_choice + ".csv"
        input.to_csv(os.path.join(INPUT_FOLDER, name))
    x = input["x"].iloc[0]
    y = input["y"].iloc[0]

    df["dist"] = (df["x"] - x)**2 + (df["y"] - y)**2

    df_nomissing = df[~((df["lithostrat_id"].isna()) | (df["lithostrat_id"]=="None") \
                        | df["lithostrat_id"].str.contains('Onbekend') | (df["lithostrat_id"]=="Onbekend") | df["lithostrat_id"].str.contains('nan'))]
    closest_ids = df_nomissing.groupby("sondering_id")["dist"].min().nsmallest(15).index.tolist()
    closest = df[df["sondering_id"].isin(closest_ids)]
    closest_path = os.path.join(INPUT_FOLDER, "closest.csv")
    closest.to_csv(closest_path, index=False)




if __name__ == "__main__":
    import sys
    user_choice = sys.argv[1]  # get argument from Streamlit
    preprocessing(user_choice)