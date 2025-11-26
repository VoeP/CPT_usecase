import pandas as pd

path_to_parquet ="C:/Users/volte/Downloads/vw_cpt_brussels_params_completeset_20250318_remapped.parquet"
df = pd.read_parquet(path_to_parquet)
sonderings = list(set(df["sondering_id"]))
pick_one = df[df["sondering_id"] == sonderings[0]]
pick_one.to_csv("input/input.csv")
input = pd.read_csv("input/input.csv")
x=list(set(input["x"]))[0]
y=list(set(input["y"]))[0]

df["dist"] = (df["x"] - x)**2 + (df["y"] - y)**2
minvalue = df["dist"].min()

minvalues= df[df["dist"] == minvalue]
sondids = list(set(minvalues["sondering_id"]))
closest = minvalues[minvalues["sondering_id"] == sondids[0]]
closest.to_csv("input/closest.csv")