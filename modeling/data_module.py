import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer


class DataSet():
    def __init__(self, path_to_parquet):
        self.raw_df = pd.read_parquet(path_to_parquet)