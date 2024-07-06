import dash
from dash import dcc, html
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import hyrox_results_analysis as _hra
import data_loading_helpers as _dlh
import constants as _constants

# Constants and styles
DATA_PATH = "/Users/vladmatei/VS_code_files/hyrox_analysis/assets/hyroxData"
all_race_names = _dlh.list_files_in_directory(DATA_PATH)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "250px",
    "padding": "20px",
    "background-color": "#f8f9fa",
}

# Sidebar layout
sidebar = html.Div(
    [
        html.P('Select a race:'),
        dcc.Dropdown(
            id="Race",
            options=[{'label': name, 'value': name} for name in all_race_names],
            value=all_race_names[0],
            style={'width': '100%'}
        ),
        html.P('Select Division:'),
        dcc.Dropdown(
            id="Division",
            options=[],
            value='all',
            style={'width': '100%'}
        ),
        html.P('Select Gender:'),
        dcc.Dropdown(
            id='Gender',
            options=[],
            value='all',
            style={'width': '100%'}
        ),
    ],
    style=SIDEBAR_STYLE
)

# Application layout
app = dash.Dash(external_stylesheets=[dbc.themes.LUX])
app.layout = html.Div([
    dbc.Row([
        dbc.Col(
            html.H2("Hyrox Results Analysis", style={
                'textAlign': 'center',
                'color': '#333',
                'marginBottom': '40px'
            })
        )
    ]),
    dbc.Row([
        dbc.Col(sidebar, width=3),
        dbc.Col(html.Div(children=[
            dcc.Loading(
                id="loading-race-info",
                type="circle",
                children=html.Div(id='race-info', style={
                    'textAlign': 'center',
                    'marginTop': '20px'
                })
            ),
            dcc.Loading(
                id="loading-race-graph",
                type="circle",
                children=dcc.Graph(figure={}, id="race_graph")
            )
        ]), width=9)
    ]),
    dcc.Store(id='race_df')
])

# Callbacks
@app.callback(Output('race_df', 'data'), Input('Race', 'value'))
def load_data(selected_race):
    """Load race data based on the selected race."""
    try:
        if selected_race.lower() == _constants.ALL_RACES.lower():
            df = _hra.load_all_races()
        else:
            df = _hra.load_one_file(f"{DATA_PATH}/{selected_race}.csv")
        return df.to_json()
    except Exception as e:
        return pd.DataFrame().to_json()  # Return an empty DataFrame in case of an error

@app.callback(Output('Division', 'options'), Input('race_df', 'data'))
def update_division_options(race_df):
    """Update division dropdown options based on the loaded race data."""
    try:
        df = pd.read_json(race_df)
        divisions = df['division'].unique().tolist()
        divisions.insert(0, 'all')
        return [{'label': division, 'value': division} for division in divisions]
    except Exception as e:
        return []

@app.callback(Output('Gender', 'options'), Input('race_df', 'data'))
def update_gender_options(race_df):
    """Update gender dropdown options based on the loaded race data."""
    try:
        df = pd.read_json(race_df)
        genders = df['gender'].unique().tolist()
        genders.insert(0, 'all')
        return [{'label': gender, 'value': gender} for gender in genders]
    except Exception as e:
        return []

@app.callback(Output('race-info', 'children'), [Input('race_df', 'data'), Input('Race', 'value')])
def update_race_info(race_df, selected_race):
    """Update race information based on the loaded race data."""
    try:
        df = pd.read_json(race_df)
        info_message = f"Race: {selected_race}, Number of entries: {len(df)}"
        return info_message
    except Exception as e:
        return "Race: N/A, Number of entries: N/A"

@app.callback(Output('race_graph', 'figure'), Input('race_df', 'data'))
def update_graph(race_df):
    """Update race graph based on the loaded race data."""
    try:
        df = pd.read_json(race_df)
        mean_value_runs, mean_value_stations = _hra.extract_mean_values_runs_stations(df)
        x_vals = np.arange(len(mean_value_runs))
        fig = go.Figure(
            data=[
                go.Scatter(x=x_vals, y=mean_value_runs, name='Runs', text=_constants.RUN_LABELS, mode="markers+text",
                           textposition='top center'),
                go.Scatter(x=x_vals, y=mean_value_stations, name='Stations', text=_constants.STATIONS, mode="markers+text",
                           textposition='top center')
            ],
            layout={"xaxis": {"title": "Runs/Stations Numbers"}, "yaxis": {"title": "Time (Minutes)"},
                    "title": "Average Times Analysis"}
        )
        return fig
    except Exception as e:
        return go.Figure()  # Return an empty figure in case of an error

if __name__ == '__main__':
    app.run_server(debug=True)
