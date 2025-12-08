## This is here because I don't want to fight relative import errors right now (VoeP)
## todo: remove this from here and get relative import to work correctly



import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import random
import numpy as np

seed = 22 #current date

segments_oi = [  # example segments of interest (same as the ones from the company's excel sheet)
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


class DataSet():
    
    def __init__(self, path_to_parquet, segments_of_interest, portion_validation, dashboard = False):
        """Takes path to data, segments of interest to filter on, and
        the fraction that the validation set should be of the whole filtered
        data."""
        try:
            self.raw_df = pd.read_parquet(path_to_parquet)
        except:
            self.raw_df = pd.read_csv(path_to_parquet)
        self.val_size = portion_validation
        #this filters out the segments we don't want and creates the test dataset,
        #since this way the NaNs are also filtered out
        if dashboard == False:
            self.known_data = self.raw_df[self.raw_df["lithostrat_id"].isin(segments_of_interest)] 
        else:
            self.known_data = self.raw_df
        drillings = self.known_data.sondering_id.unique()

        #the below shuffles the drilling ids before we split into test and validation
        random.Random(seed).shuffle(drillings)
        drillings = drillings[0:int(len(drillings)*self.val_size)]

        self.train = self.known_data[self.known_data.sondering_id.isin(drillings)]
        self.validation = self.known_data[~self.known_data.sondering_id.isin(drillings)]


    def preprocess(self, X, points=5, features=[], model_id=1):
        """Preprocessing script, currently incldes:
        - sliding window over selected features
        
        Takes:
        X, either train or test set
        points=int points to average

        Run on train/test/validation sets if desired
        """
        if model_id == 1:
            id_column = "sondering_id"
            processed_subframes = []
            ids_to_process = X[id_column].unique()

            for id_val in ids_to_process:
                subdf = X[X[id_column] == id_val].copy()
                n_rows = len(subdf)
                features_list = []

                for start in range(0, n_rows - points + 1):
                    window = subdf.iloc[start:start + points]
                    feats = window.iloc[-1].to_dict()
                    feats[id_column] = id_val

                    for col in features:
                        col_vals = window[col].values
                        feats[f"{col}_mean"] = col_vals.mean()
                        feats[f"{col}_slope"] = np.polyfit(np.arange(points), col_vals, 1)[0]

                    features_list.append(feats)

                df_feats = pd.DataFrame(features_list)

                if len(df_feats) < n_rows:
                    last_feats = df_feats.iloc[-1].copy()
                    tail = pd.DataFrame([last_feats] * (n_rows - len(df_feats)))
                    df_feats = pd.concat([df_feats, tail], ignore_index=True)

                processed_subframes.append(df_feats)

            return pd.concat(processed_subframes, ignore_index=True)


    
    def get_preprocessed(self, split=True, features = []):
        if split:
            return self.preprocess(self.train, features = features), \
                self.preprocess(self.validation, features = features)
        else:
            #return 1, self.preprocess(self.known_data), self.known_data, self.raw_df
            return self.preprocess(self.raw_df, features=features)