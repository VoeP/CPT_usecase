import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
import random

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
    def __init__(self, path_to_parquet, segments_of_interest, portion_validation):
        """Takes path to data, segments of interest to filter on, and
        the fraction that the validation set should be of the whole filtered
        data."""
        self.raw_df = pd.read_parquet(path_to_parquet)
        self.val_size = portion_validation
        #this filters out the segments we don't want and creates the test dataset,
        #since this way the NaNs are also filtered out
        self.known_data = self.raw_df[self.raw_df["lithostrat_id"].isin(segments_of_interest)] 

 #   def __post_init__(self):
        drillings = self.known_data.sondering_id.unique()

        #the below shuffles the drilling ids before we split into test and validation
        random.Random(seed).shuffle(drillings)
        drillings = drillings[0:int(len(drillings)*self.val_size)]

        self.train = self.known_data[self.known_data.sondering_id.isin(drillings)]
        self.validation = self.known_data[~self.known_data.sondering_id.isin(drillings)]