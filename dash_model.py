import dash
from dash import dcc
from dash import html
import dash_bootstrap_components as dbc
from dash.dependencies import Output
from dash.dependencies import Input

import plotly.express as px
import hyrox_results_analysis as _hra
import data_loading_helpers as _dlh
import pandas as _pd
import constants as _constants
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

all_race_names = _dlh.list_files_in_directory("/Users/vladmatei/VS_code_files/hyrox_analysis/assets/hyroxData")

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "100px",
    "background-color": "#f8f9fa",
}

sidebar = html.Div(
    [
        dcc.Dropdown(
            id="Race",
            options=[{
                'label': i,
                'value': i
            } for i in all_race_names],
            value=all_race_names[0],
            style={
                'width': '90%',
                'textAlign': 'center'
            }),
    ],
    style={'width': '50%',
           'display': 'flex',
           'justifyContent': 'center'}),

###  Layout of the application
app = dash.Dash(external_stylesheets=[dbc.themes.LUX])

app.layout = html.Div([
    dbc.Row([
        dbc.Col(),
        dbc.Col(
            html.H2("Hyrox Results Analysis", style={
                'textAlign': 'center',
                'color': '#333',
                'marginBottom': '40px'
            })
        )
    ]),
    dbc.Row([
        dbc.Col(sidebar),
        dbc.Col(html.Div(children=[
            html.Div(id='race-info', style={
                'textAlign': 'center',
                'maringTop': '20px'
            }),
            dcc.Graph(figure={}, id="race_graph")]))
    ])
    ,
    dcc.Store(id='race_df')
])


#####    CALLBACK   ######
# first callback, that makes use of the 'Store' component to be then used by subsequent callback in our function
@app.callback(Output('race_df', 'data'), Input('Race', 'value'))
def load_data(selected_race):
    if selected_race.lower() == _constants.ALL_RACES.lower():
        df = _hra.load_all_races()
    else:
        df = _hra.load_one_file(f"/Users/vladmatei/VS_code_files/hyrox_analysis/assets/hyroxData/{selected_race}.csv")
    # data needs to be serialized into a JSON string before being placed into storage
    return df.to_json()


@app.callback(Output('race-info', 'children'), Input('race_df', 'data'), Input('Race', 'value'))
def update_race_info(race_df, selected_race):
    _pd.read_json(race_df)
    info_message = f"Race:{selected_race}, Number of entries: {len(race_df)}"
    return info_message


# callback function for updating the race plot
@app.callback(Output('race_graph', 'figure'), Input('race_df', 'data'))
def update_graph(race_df):
    df = _pd.read_json(race_df)
    mean_value_runs, mean_value_stations = _hra.extract_mean_values_runs_stations(df)
    x_vals = np.array((range(len(mean_value_runs))))
    fig = go.Figure(
        data=[
            go.Scatter(x=x_vals, y=mean_value_runs, name='Runs'),
            go.Scatter(x=x_vals, y=mean_value_stations, name='Stations')
        ],
        layout={"xaxis": {"title": "Runs/Stations Numbers"}, "yaxis": {"title": "Average Values"},
                "title": "Average Times Analysis"}
    )
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
