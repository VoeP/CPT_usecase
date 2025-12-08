
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from app import update_graph, SEGMENTS_OI

# Mock data
raw_data = [
    {'sondering_id': 'S1', 'diepte': 1.0, 'qc': 10, 'rf': 1.5},
    {'sondering_id': 'S1', 'diepte': 2.0, 'qc': 12, 'rf': 1.2},
    {'sondering_id': 'S1', 'diepte': 3.0, 'qc': 15, 'rf': 1.0},
    {'sondering_id': 'S2', 'diepte': 1.0, 'qc': 8, 'rf': 2.0},
    {'sondering_id': 'S2', 'diepte': 2.0, 'qc': 9, 'rf': 1.8},
]

table_data = [
    {'sondering_id': 'S1', 'depth_bin': '(0.0, 1.5]', 'predicted_label': 'Quartair'},
    {'sondering_id': 'S1', 'depth_bin': '(1.5, 3.0]', 'predicted_label': 'Diest'},
    {'sondering_id': 'S2', 'depth_bin': '(0.0, 2.0]', 'predicted_label': 'Quartair'},
]

print("Running update_graph with mock data...")
try:
    fig = update_graph(table_data, raw_data)
    print("Figure created successfully.")
    # print(fig.layout)
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
