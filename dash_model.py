import dash
from dash import dcc
from dash import html

from dash.dependencies import Output
from dash.dependencies import Input

import plotly.express as px
import hyrox_results_analysis as _hra
import data_loading_helpers as _dlh
import pandas as pd


all_race_names = _dlh.list_files_in_directory("/Users/vladmatei/VS_code_files/hyrox_analysis/assets/hyroxData")
print(all_race_names)
app = dash.Dash()

app.layout = html.Div([
    html.H2("Hyrox Results Analysis", style={
        'textAlign': 'center',
        'color': '#333',
        'marginBottom': '40px'
    }),
    html.Div(
        [
            dcc.Dropdown(
                id="Race",
                options=[{
                    'label': i,
                    'value': i
                } for i in all_race_names],
                value=all_race_names[0],
                style={
                'width': '75%',
                'textAlign': 'center'
                }),
        ],
        style={'width': '50%',
               'display': 'flex',
               'justifyContent': 'center'})
])

if __name__ == '__main__':
    app.run_server(debug=True)
