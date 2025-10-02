## CPT usecase

modeling/ contains notebooks and scripts for model building

dashbord/ contains streamlit dashboard scripts


You need to add your aboslute paths to the paths_cpt_file for it to run. You need to define PATH_TO_PARQUET = "" and PATH_TO_MODEL ="".
The idea there is that we can change this logic later to use environment variables instead to run this in different environments.
For the dashboard to work, you need to also create the pickle of the model by running the EDA.ipynb (or whatever other method).
