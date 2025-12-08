import base64
import io
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table, Input, Output, State, no_update
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import matplotlib.pyplot as plt
import io

# Add modeling directory to path to import data_processing
# Assuming dashboard/ is at the same level as modeling/
sys.path.append(str(Path(__file__).parent.parent / "modeling"))
import data_processing as dp

# Configuration 
MODEL_PATH = Path(__file__).parent.parent / "results" / "models" / "best_xgb_model.pkl"
LE_PATH = Path(__file__).parent.parent / "results" / "models" / "label_encoder.pkl"
BIN_W = 0.6
EXTRACT_TREND = True
TREND_TYPE = "multiplicative"
SEGMENTS_OI = [
    "Quartair", "Diest", "Bolderberg", "Sint_Huibrechts_Hern", "Ursel",
    "Asse", "Wemmel", "Lede", "Brussel", "Merelbeke", "Kwatrecht",
    "Mont_Panisel", "Aalbeke", "Mons_en_Pevele"
]
DEBUG_PLOT_DEFAULT = True  # Default for diagnostics and marker rendering
USE_MPL_DEFAULT = False    # Default for Matplotlib fallback rendering

# Load Model & Encoder 
try:
    model = joblib.load(MODEL_PATH)
    le = joblib.load(LE_PATH)
    print("Model and Label Encoder loaded successfully.")
except FileNotFoundError:
    print("Error: Model or Label Encoder not found. Please run the modeling notebook to save them.")
    model = None
    le = None

# Helper Functions 
def parse_contents(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith('.parquet'):
            df = pd.read_parquet(io.BytesIO(decoded), engine='fastparquet')
        elif filename.lower().endswith('.csv'):
            df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
        else:
            return None
        return df
    except Exception as e:
        print(f"Error parsing {filename}: {e}")
        return None

def get_layers_from_bins(df):
    """
    Converts binned predictions into continuous layers for plotting.
    Expects df to have 'depth_bin' (Interval) and 'predicted_label'.
    """
    layers = []
    if df.empty:
        return layers
    
    # Ensure depth_bin is Interval
    if isinstance(df['depth_bin'].iloc[0], str):
        # simplistic parsing of pandas interval string "(a, b]"
        df['top'] = df['depth_bin'].apply(lambda x: float(x.strip('()[]').split(',')[0]))
        df['bottom'] = df['depth_bin'].apply(lambda x: float(x.strip('()[]').split(',')[1]))
    else:
        df['top'] = df['depth_bin'].apply(lambda x: x.left)
        df['bottom'] = df['depth_bin'].apply(lambda x: x.right)

    df = df.sort_values('top')
    
    current_label = None
    start_depth = None
    
    for _, row in df.iterrows():
        label = row['predicted_label']
        if label != current_label:
            if current_label is not None:
                layers.append({
                    'label': current_label,
                    'top': start_depth,
                    'bottom': row['top']
                })
            current_label = label
            start_depth = row['top']
    
    # Add last layer
    if current_label is not None:
        layers.append({
            'label': current_label,
            'top': start_depth,
            'bottom': df.iloc[-1]['bottom']
        })
        
    return layers

# Matplotlib helpers
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('ascii')

def render_mpl_image(plot_df, ids, use_predictions=False):
    n = len(ids)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 8), sharey=True)
    if n == 1:
        axes = [axes]
    # Colors for background layers
    colors = {label: (np.random.rand(), np.random.rand(), np.random.rand(), 0.25) for label in SEGMENTS_OI}
    for ax, sid in zip(axes, ids):
        cpt = plot_df[plot_df['sondering_id'] == sid].copy().sort_values('diepte')
        # Background lithostrat layers
        if use_predictions and ('predicted_label' in cpt.columns) and ('depth_bin' in cpt.columns):
            layers_df = cpt[['depth_bin', 'predicted_label']].dropna()
            layers = get_layers_from_bins(layers_df)
            for layer in layers:
                color = colors.get(layer['label'], (0.6, 0.6, 0.6, 0.25))
                ax.axhspan(layer['top'], layer['bottom'], facecolor=color, edgecolor=None)
                ax.text(ax.get_xlim()[0], (layer['top'] + layer['bottom'])/2, layer['label'], fontsize=9, va='center')
        # QC line
        if 'qc' in cpt.columns:
            ax.plot(cpt['qc'], cpt['diepte'], color='black', lw=2)
        ax.set_xlabel('QC (MPa)')
        ax.set_title(f'CPT {sid}')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        # Rf twin axis
        ax_top = ax.twiny()
        if 'rf' in cpt.columns:
            ax_top.plot(cpt['rf'], cpt['diepte'], color='purple', lw=1.8)
        ax_top.set_xlabel('Rf (%)', color='purple')
        ax_top.spines['top'].set_color('purple')
        ax_top.tick_params(axis='x', colors='purple')
    fig.tight_layout()
    return fig_to_base64(fig)

# App Layout 
app = dash.Dash(__name__, external_stylesheets=['https://codepen.io/chriddyp/pen/bWLwgP.css'])

app.layout = html.Div([
    html.H1("CPT Lithostratigraphy Prediction Dashboard", style={'textAlign': 'center'}),
    
    # 1. Data Upload
    html.Div([
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Test Data File')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            },
            multiple=False
        ),
        html.Div(id='upload-status')
    ]),
    
    # NEW: Raw Data Preview
    html.Div([
        html.H5("Uploaded Data Preview (First 5 rows)"),
        dash_table.DataTable(
            id='raw-data-table',
            style_table={'overflowX': 'auto'},
            style_cell={'textAlign': 'left'}
        )
    ], style={'margin': '20px', 'padding': '10px', 'border': '1px solid #ddd'}),

    # 2. Controls
    html.Div([
        html.Label("Select Sondering IDs (Max 3):"),
        dcc.Dropdown(id='sondering-dropdown', multi=True),
        html.Br(),
        # Debug toggle
        dcc.Checklist(
            id='debug-toggle',
            options=[{'label': 'Show markers (debug)', 'value': 'debug'}],
            value=['debug'] if DEBUG_PLOT_DEFAULT else [],
            inputStyle={'margin-right': '6px'},
            labelStyle={'display': 'inline-block', 'margin-right': '12px'}
        ),
        html.Br(),
        html.Button('Predict Lithostratigraphy', id='predict-btn', n_clicks=0, className='button-primary'),
    ], style={'padding': '20px', 'backgroundColor': '#f9f9f9'}),
    
    # Visualization 
    html.Div(id='plot-container'),
    
    # Table Filters (Below Graph)
    html.Div([
        html.Label("Table Filters:", style={'fontWeight': 'bold'}),
        html.Div([
            html.Div([
                html.Label("Filter Sondering IDs:"),
                dcc.Dropdown(id='table-id-filter', multi=True, placeholder='Select IDs to show')
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top'}),
            html.Div([
                html.Label("Filter Depth Range (m):"),
                dcc.RangeSlider(id='table-depth-range', min=0, max=50, step=0.1, value=[0, 50], allowCross=False, 
                                marks={0: '0', 10: '10', 20: '20', 30: '30', 40: '40', 50: '50'})
            ], style={'width': '48%', 'display': 'inline-block', 'paddingLeft': '12px', 'verticalAlign': 'top'})
        ]),
    ], style={'margin': '20px', 'padding': '15px', 'border': '1px solid #eee', 'borderRadius': '5px'}),

    # Editable Table
    html.Div([
        html.H4("Prediction Results (Editable)", style={'display': 'inline-block', 'marginRight': '20px'}),
        html.Button("Save to CSV", id="save-btn"),
        dcc.Download(id="download-dataframe-csv"),
    ]),
    html.P("Edit the 'predicted_label' column to adjust the plot above."),
    dash_table.DataTable(
        id='prediction-table',
        columns=[
            {'name': 'Sondering ID', 'id': 'sondering_id', 'editable': False},
            {'name': 'Depth (m)', 'id': 'diepte', 'editable': False},
            {'name': 'Predicted Label', 'id': 'predicted_label', 'editable': True, 'presentation': 'dropdown'},
        ],
        data=[],
        editable=True,
        filter_action="native",
        sort_action="native",
        page_action="native",
        page_current=0,
        page_size=20,
        dropdown={
            'predicted_label': {
                'options': [{'label': i, 'value': i} for i in SEGMENTS_OI]
            }
        }
    ),
    
    # Hidden Stores
    dcc.Store(id='raw-data-store'),  # Stores uploaded raw data
    dcc.Store(id='processed-data-store') # Stores processed data for plotting QC
])

# Callbacks

# Handle Upload & Populate Dropdown & Preview Table
@app.callback(
    [Output('raw-data-store', 'data'),
     Output('sondering-dropdown', 'options'),
     Output('upload-status', 'children'),
     Output('raw-data-table', 'data'),
     Output('raw-data-table', 'columns')],
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def update_output(contents, filename):
    if contents is None:
        return no_update, [], "", [], []
    
    df = parse_contents(contents, filename)
    if df is None:
        return no_update, [], "Error parsing file.", [], []
    
    # Ensure required columns exist
    required_cols = ['sondering_id', 'diepte', 'qc'] # Add others if needed by dp
    if not all(col in df.columns for col in required_cols):
        return no_update, [], f"File missing columns: {required_cols}", [], []
    
    ids = df['sondering_id'].unique().tolist()
    options = [{'label': str(i), 'value': i} for i in ids]
    
    # Preview data (head)
    preview_data = df.head().to_dict('records')
    preview_columns = [{"name": i, "id": i} for i in df.columns]

    # Convert to dict for store (parquet serialization is better but json is default for Dash)
    return df.to_dict('records'), options, f"Loaded {filename} with {len(df)} rows.", preview_data, preview_columns

# 2. Handle Prediction
@app.callback(
    Output('processed-data-store', 'data'),
    Input('predict-btn', 'n_clicks'),
    [State('sondering-dropdown', 'value'),
     State('raw-data-store', 'data')]
)
def run_prediction(n_clicks, selected_ids, raw_data_dict):
    if n_clicks == 0 or not selected_ids or not raw_data_dict:
        return no_update
    
    if len(selected_ids) > 3:
        selected_ids = selected_ids[:3] # Enforce limit
        
    # Reconstruct DataFrame
    cpt_data = pd.DataFrame(raw_data_dict)
    
    # Process Data
    # We pass dummy lithostrat_id if missing because process_test_train might expect it
    if 'lithostrat_id' not in cpt_data.columns:
        cpt_data['lithostrat_id'] = 'Unknown'
        
    try:
        processed_df = dp.process_test_train(
            cpt_df=cpt_data,
            sondering_ids=selected_ids,
            bin_w=BIN_W,
            do_extract_trend=EXTRACT_TREND,
            trend_type=TREND_TYPE
        )
    except Exception as e:
        print(f"Processing Error: {e}")
        return []

    # Prepare Features
    exclude_cols = ["sondering_id", "lithostrat_id", "depth_bin", "diepte", "QC_raw"]
    feature_cols = [c for c in processed_df.columns if c not in exclude_cols]
    
    # Predict
    if model and le:
        X = processed_df[feature_cols]
        # Handle missing cols if any (simple imputation for demo)
        # Ideally use the pipeline from training
        
        # The model pipeline usually handles imputation if it was included
        preds_enc = model.predict(X)
        preds_label = le.inverse_transform(preds_enc)
        
        processed_df['predicted_label'] = preds_label
    else:
        processed_df['predicted_label'] = "Model Not Loaded"

    # MERGE LOGIC START 
    # Replicate binning on raw data to enable merge with predictions
    # Filter raw data to selected IDs
    cpt_subset = cpt_data[cpt_data['sondering_id'].isin(selected_ids)].copy()
    
    # Calculate bins exactly as process_test_train does (global max of subset)
    depth_col = 'diepte'
    if depth_col in cpt_subset.columns:
        max_depth = float(cpt_subset[depth_col].max())
        bins = list(np.arange(0, max_depth + BIN_W, BIN_W))
        cpt_subset['depth_bin'] = pd.cut(cpt_subset[depth_col], bins=bins, include_lowest=True, ordered=True)
        
        # Merge predictions onto raw data
        # processed_df has ['sondering_id', 'depth_bin', 'predicted_label']
        merged_df = pd.merge(
            cpt_subset, 
            processed_df[['sondering_id', 'depth_bin', 'predicted_label']], 
            on=['sondering_id', 'depth_bin'], 
            how='left'
        )
        
        # Prepare store_df (merged data)
        store_df = merged_df.copy()
        store_df['depth_bin'] = store_df['depth_bin'].astype(str) # Serialize
    else:
        store_df = pd.DataFrame()
    # MERGE LOGIC END 

    return store_df.to_dict('records')

# 2b. Filter Prediction Table & Update Options
@app.callback(
    [Output('prediction-table', 'data'),
     Output('table-id-filter', 'options'),
     Output('table-depth-range', 'min'),
     Output('table-depth-range', 'max'),
     Output('table-depth-range', 'marks')],
    [Input('processed-data-store', 'data'),
     Input('table-id-filter', 'value'),
     Input('table-depth-range', 'value')]
)
def update_table_view(store_data, filter_ids, depth_range):
    if not store_data:
        return [], [], 0, 50, {0:'0', 50:'50'}
    
    df = pd.DataFrame(store_data)
    
    # Prepare options from full data
    if 'sondering_id' in df.columns:
        unique_ids = sorted(df['sondering_id'].astype(str).unique())
        options = [{'label': i, 'value': i} for i in unique_ids]
    else:
        options = []
        
    # Prepare depth range from full data
    min_depth = 0
    max_depth = 50
    if 'diepte' in df.columns:
        df['diepte'] = pd.to_numeric(df['diepte'], errors='coerce')
        min_depth = float(df['diepte'].min())
        max_depth = float(df['diepte'].max())
    
    marks = {int(i): str(int(i)) for i in np.linspace(min_depth, max_depth, 6)}

    # Apply Filters
    # 1. ID Filter
    if filter_ids:
        df = df[df['sondering_id'].astype(str).isin(filter_ids)]
    
    # 2. Depth Filter
    if depth_range and len(depth_range) == 2:
        dmin, dmax = depth_range
        df = df[(df['diepte'] >= dmin) & (df['diepte'] <= dmax)]
        
    # Prepare Table Data
    if not df.empty:
        table_df = df[['sondering_id', 'diepte', 'predicted_label']].copy()
        table_df['sondering_id'] = table_df['sondering_id'].astype(str)
        table_df = table_df.dropna(subset=['predicted_label'])
        return table_df.to_dict('records'), options, min_depth, max_depth, marks
    
    return [], options, min_depth, max_depth, marks

# 3. Update Plot (Triggered by Table Edit or New Prediction)
@app.callback(
    Output('plot-container', 'children'),
    [Input('sondering-dropdown', 'value'),
     Input('prediction-table', 'data')],
    [State('raw-data-store', 'data'),
     State('processed-data-store', 'data'),
    State('debug-toggle', 'value')]
)
def update_graph(selected_ids, table_data, raw_data, merged_data, debug_value):
    """Plot QC (bottom axis) and Rf (top axis). Uses merged data if available, else raw data."""
    ctx = dash.callback_context
    if not ctx.triggered:
        return None
    
    if not selected_ids or not raw_data:
        return None

    # Determine which data to use
    # If we have merged_data (predictions) and it matches selected_ids, use it.
    # Otherwise use raw_data.
    
    use_predictions = False
    debug_plot = DEBUG_PLOT_DEFAULT if debug_value is None else ('debug' in debug_value)
    # Force Matplotlib rendering only
    use_mpl = True
    plot_df = pd.DataFrame(raw_data)
    
    # Check if we should use predictions
    if merged_data:
        merged_df = pd.DataFrame(merged_data)
        # Check if merged_df contains the selected IDs
        # We convert to string to be safe
        merged_ids = set(merged_df['sondering_id'].astype(str).unique())
        sel_ids = set(map(str, selected_ids))
        
        if sel_ids.issubset(merged_ids):
            plot_df = merged_df
            use_predictions = True

    # If table edits exist, ensure we render predicted layers even without merged_data
    if table_data and len(table_data) > 0:
        use_predictions = True
    
    # Filter for selected IDs
    plot_df['sondering_id'] = plot_df['sondering_id'].astype(str)
    ids = [str(i) for i in selected_ids]
    plot_df = plot_df[plot_df['sondering_id'].isin(ids)].copy()

    # Apply table edits (override predicted_label) if available
    if merged_data and table_data:
        edits_df = pd.DataFrame(table_data)
        if not edits_df.empty and 'predicted_label' in edits_df.columns:
            edits_df['sondering_id'] = edits_df['sondering_id'].astype(str)
            # Prefer merging by diepte if available, else fallback to depth_bin
            if 'diepte' in edits_df.columns and 'diepte' in plot_df.columns:
                # Ensure diepte numeric alignment
                def clean_numeric_local(series):
                    return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')
                edits_df['diepte'] = clean_numeric_local(edits_df['diepte'])
                plot_df['diepte'] = clean_numeric_local(plot_df['diepte'])
                join_cols = ['sondering_id','diepte']
                edits_join = edits_df[join_cols + ['predicted_label']].rename(columns={'predicted_label':'edited_label'})
                plot_df = plot_df.merge(edits_join, on=join_cols, how='left')
            elif 'depth_bin' in edits_df.columns and 'depth_bin' in plot_df.columns:
                edits_df['depth_bin'] = edits_df['depth_bin'].astype(str)
                plot_df['depth_bin'] = plot_df['depth_bin'].astype(str)
                join_cols = ['sondering_id','depth_bin']
                edits_join = edits_df[join_cols + ['predicted_label']].rename(columns={'predicted_label':'edited_label'})
                plot_df = plot_df.merge(edits_join, on=join_cols, how='left')
            # Prefer edited label where provided, else keep existing
            if 'edited_label' in plot_df.columns:
                if 'predicted_label' in plot_df.columns:
                    plot_df['predicted_label'] = plot_df['edited_label'].combine_first(plot_df['predicted_label'])
                else:
                    plot_df['predicted_label'] = plot_df['edited_label']
                plot_df = plot_df.drop(columns=['edited_label'])
    
    if plot_df.empty:
        return None

    # Ensure numeric Diepte, QC and RF with comma handling
    def clean_numeric(series):
        return pd.to_numeric(series.astype(str).str.replace(',', '.'), errors='coerce')

    if 'diepte' in plot_df.columns:
        plot_df['diepte'] = clean_numeric(plot_df['diepte'])
    
    if 'qc' in plot_df.columns:
        plot_df['qc'] = clean_numeric(plot_df['qc'])
    else:
        plot_df['qc'] = np.nan
        
    if 'rf' in plot_df.columns:
        plot_df['rf'] = clean_numeric(plot_df['rf'])
    else:
        if 'fs' in plot_df.columns:
            plot_df['fs'] = clean_numeric(plot_df['fs'])
            # Calculate Rf = (fs / qc) * 100
            plot_df['rf'] = (plot_df['fs'] / plot_df['qc'].replace(0, np.nan)) * 100
        else:
            plot_df['rf'] = np.nan

    # 
    # For better layout with many IDs, we will stack plots horizontally with shared Y axis (Depth)
    num_ids = len(ids)
    
    # Spacing and width calculations
    spacing = 0.02 # Space between plots
    plot_width = (1.0 - spacing * (num_ids - 1)) / num_ids # Width of each plot

    # Calculate global depth range for consistent plotting
    min_depth = None
    max_depth = None
    has_depth = (not plot_df.empty) and ('diepte' in plot_df.columns)
    if has_depth:
        # Ensure diepte is numeric before min/max (it should be already)
        min_depth = plot_df['diepte'].min()
        max_depth = plot_df['diepte'].max()
        # Add some padding
        pad = (max_depth - min_depth) * 0.02 if max_depth != min_depth else 1.0
        # Reversed range: [bottom, top]
        # Note: In Plotly, for reversed axis, range is [max, min] if we want 0 at top? 
        # No, autorange='reversed' flips it. If we provide range, we should provide [max, min] to flip it?
        # Actually, standard is [min, max] and autorange='reversed' flips display.
        # But if we provide explicit range, we must provide [max, min] to achieve reversed effect.
        y_range = [max_depth + pad, min_depth - pad]
    else:
        y_range = [30, 0]

    if debug_plot:
        print(f"Selected IDs: {ids}")
        # Only print min/max if defined in the branch above
        if has_depth:
            print(f"Global depth range: {min_depth} - {max_depth}")
        print(f"Using y_range: {y_range}")

    # Matplotlib rendering path
    if use_mpl:
        img_b64 = render_mpl_image(plot_df, ids, use_predictions=use_predictions)
        img = html.Img(src='data:image/png;base64,' + img_b64, style={'width': '100%'})
        return img

    # Fallback (unused): Create empty Plotly figure
    fig = go.Figure()

    # Colors for background layers
    colors = {label: f"rgba({np.random.randint(0,255)},{np.random.randint(0,255)},{np.random.randint(0,255)},0.25)" 
              for label in SEGMENTS_OI}
    
    # Dictionary to store axis layouts
    layout_axes = {}
    
    for i, sondering_id in enumerate(ids):
        # Axis refs for subplot
        axis_idx = i + 1
        base_axis_name = f'xaxis{axis_idx}' if axis_idx > 1 else 'xaxis'
        y_axis_name = f'yaxis{axis_idx}' if axis_idx > 1 else 'yaxis'
        
        # Get data for this CPT
        cpt_data = plot_df[plot_df['sondering_id'] == sondering_id].copy()

        # Add background rectangles ONLY if we have predictions
        if use_predictions and 'predicted_label' in cpt_data.columns:
            # We need to reconstruct the table_df structure for get_layers_from_bins
            # Or just use cpt_data if it has depth_bin and predicted_label
            if 'depth_bin' in cpt_data.columns:
                 # Process layers for background
                layers = get_layers_from_bins(cpt_data[['depth_bin', 'predicted_label']].dropna())
                for layer in layers:
                    layer['color'] = colors.get(layer['label'], 'rgba(150,150,150,0.25)')

                for layer in layers:
                    fig.add_shape(
                        type="rect",
                        x0=0, x1=1, xref=f'x{axis_idx} domain' if axis_idx > 1 else 'x domain',
                        y0=layer['top'], y1=layer['bottom'], yref=f'y{axis_idx}' if axis_idx > 1 else 'y',
                        fillcolor=layer['color'], opacity=0.25, layer="below", line_width=0
                    )
                    
                    # Add text annotation
                    fig.add_annotation(
                        x=0.5, xref=f'x{axis_idx} domain' if axis_idx > 1 else 'x domain',
                        y=(layer['top'] + layer['bottom'])/2, yref=f'y{axis_idx}' if axis_idx > 1 else 'y',
                        text=layer['label'], showarrow=False, font=dict(size=10, color='black')
                    )

        # QC and Rf traces
        # Fixed fallbacks to guarantee visibility pre-prediction
        qc_range = [0, 45]
        rf_range = [0, 10]

        if not cpt_data.empty:
            # Prepare per-series datasets to avoid over-dropping rows
            qc_plot_data = cpt_data.dropna(subset=['qc', 'diepte']).sort_values('diepte')
            rf_plot_data = cpt_data.dropna(subset=['rf', 'diepte']).sort_values('diepte')

            # Calculate dynamic ranges with padding from available series
            if not qc_plot_data.empty:
                qc_min, qc_max = qc_plot_data['qc'].min(), qc_plot_data['qc'].max()
                if np.isfinite(qc_min) and np.isfinite(qc_max) and qc_max > qc_min:
                    qc_pad = (qc_max - qc_min) * 0.05
                    qc_range = [qc_min - qc_pad, qc_max + qc_pad]
            if not rf_plot_data.empty:
                rf_min, rf_max = rf_plot_data['rf'].min(), rf_plot_data['rf'].max()
                if np.isfinite(rf_min) and np.isfinite(rf_max) and rf_max > rf_min:
                    rf_pad = (rf_max - rf_min) * 0.05
                    rf_range = [rf_min - rf_pad, rf_max + rf_pad]

            if debug_plot:
                print(f"ID {sondering_id}: qc_rows={len(qc_plot_data)} rf_rows={len(rf_plot_data)}")
                if not qc_plot_data.empty:
                    print(f"  qc diepte: {qc_plot_data['diepte'].min()} - {qc_plot_data['diepte'].max()}")
                    print(f"  qc: {qc_plot_data['qc'].min()} - {qc_plot_data['qc'].max()} -> range {qc_range}")
                if not rf_plot_data.empty:
                    print(f"  rf diepte: {rf_plot_data['diepte'].min()} - {rf_plot_data['diepte'].max()}")
                    print(f"  rf: {rf_plot_data['rf'].min()} - {rf_plot_data['rf'].max()} -> range {rf_range}")

            # QC trace
            if not qc_plot_data.empty:
                fig.add_trace(go.Scatter(
                    x=qc_plot_data['qc'], y=qc_plot_data['diepte'], mode='markers+lines' if debug_plot else 'lines',
                    name='QC', line=dict(color='black', width=2.2, simplify=False),
                    marker=dict(size=4, color='black', opacity=0.9) if debug_plot else None,
                    legendgroup='QC', showlegend=(i == 0)
                ), row=1, col=axis_idx)

            # Rf trace (on secondary x-axis)
            if not rf_plot_data.empty:
                fig.add_trace(go.Scatter(
                    x=rf_plot_data['rf'], y=rf_plot_data['diepte'], mode='markers+lines' if debug_plot else 'lines',
                    name='Rf', line=dict(color='purple', width=2.0, simplify=False),
                    marker=dict(size=4, color='purple', opacity=0.9) if debug_plot else None,
                    legendgroup='Rf', showlegend=(i == 0)
                ), row=1, col=axis_idx)

        # Configure axes via update_xaxes/update_yaxes for each subplot
        fig.update_xaxes(title_text='QC (MPa)' if i == len(ids)//2 else None, range=qc_range, row=1, col=axis_idx)
        fig.update_yaxes(title_text='Depth (m)' if i == 0 else None, autorange='reversed', row=1, col=axis_idx, showticklabels=(i == 0))

    # Update layout
    fig.update_layout(
        title='Predicted Lithostratigraphy with QC and Rf Profiles',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right")
    )

    return fig, None

# 4. Save Predictions to CSV
@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("save-btn", "n_clicks"),
    State("prediction-table", "data"),
    prevent_initial_call=True,
)
def download_predictions(n_clicks, table_data):
    if not table_data:
        return no_update
    df = pd.DataFrame(table_data)
    return dcc.send_data_frame(df.to_csv, "predictions.csv", index=False)

if __name__ == '__main__':
    app.run_server(debug=True)