import pandas as pd

path_to_parquet = "C:/Users/volte/Downloads/vw_cpt_brussels_params_completeset_20250318_remapped.parquet"
df = pd.read_parquet(path_to_parquet)
sonderings = df["sondering_id"].unique()
#pick_one = df[df["sondering_id"] == sonderings[0]]
#pick_one.to_csv("input/input.csv", index=False)
input = pd.read_csv("input/input.csv")
x = input["x"].iloc[0]
y = input["y"].iloc[0]

df["dist"] = (df["x"] - x)**2 + (df["y"] - y)**2

df_nomissing = df.dropna()
closest_ids = df_nomissing.groupby("sondering_id")["dist"].min().nsmallest(3).index.tolist()
closest = df[df["sondering_id"].isin(closest_ids)]
closest.to_csv("input/closest.csv", index=False)
