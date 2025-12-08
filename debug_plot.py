import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Constants
SEGMENTS_OI = [
    "Quartair", "Diest", "Bolderberg", "Sint_Huibrechts_Hern", "Ursel",
    "Asse", "Wemmel", "Lede", "Brussel", "Merelbeke", "Kwatrecht",
    "Mont_Panisel", "Aalbeke", "Mons_en_Pevele"
]

def get_layers_from_bins(df):
    # Dummy implementation for now as we are testing raw data plotting
    return []

def update_graph_simulation(selected_ids, raw_data_df):
    print("--- Starting Simulation ---")
    
    # Simulate the logic in update_graph
    plot_df = raw_data_df.copy()
    
    # Filter for selected IDs
    plot_df['sondering_id'] = plot_df['sondering_id'].astype(str)
    ids = [str(i) for i in selected_ids]
    plot_df = plot_df[plot_df['sondering_id'].isin(ids)].copy()
    
    if plot_df.empty:
        print("Plot DF is empty after filtering")
        return

    # Ensure numeric Diepte, QC and RF with comma handling
    def clean_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

    if 'diepte' in plot_df.columns:
        plot_df['diepte'] = clean_numeric(plot_df['diepte'])
    
    if 'qc' in plot_df.columns:
        plot_df['qc'] = clean_numeric(plot_df['qc'])
    
    if 'rf' in plot_df.columns:
        plot_df['rf'] = clean_numeric(plot_df['rf'])

    # Loop over IDs
    for i, sondering_id in enumerate(ids):
        print(f"\nProcessing ID: {sondering_id}")
        cpt_data = plot_df[plot_df['sondering_id'] == sondering_id].copy()
        
        if not cpt_data.empty:
            # Drop NaNs
            cpt_plot_data = cpt_data.dropna(subset=['qc', 'rf', 'diepte']).sort_values('diepte')
            
            print(f"  Rows: {len(cpt_plot_data)}")
            if not cpt_plot_data.empty:
                print(f"  Depth range: {cpt_plot_data['diepte'].min()} - {cpt_plot_data['diepte'].max()}")
                print(f"  QC range: {cpt_plot_data['qc'].min()} - {cpt_plot_data['qc'].max()}")
                print(f"  Rf range: {cpt_plot_data['rf'].min()} - {cpt_plot_data['rf'].max()}")
                print(f"  Head:\n{cpt_plot_data[['diepte', 'qc', 'rf']].head()}")
            else:
                print("  cpt_plot_data is empty after dropna")
        else:
            print("  cpt_data is empty")

# Load data
try:
    df = pd.read_csv('data/test_raw_data.csv')
    print(f"Loaded CSV with {len(df)} rows")
    print(f"Columns: {df.columns.tolist()}")
    
    # Pick 3 IDs from the CSV
    ids = df['sondering_id'].unique()[:3]
    print(f"Selected IDs: {ids}")
    
    update_graph_simulation(ids, df)
    
except Exception as e:
    print(f"Error: {e}")
