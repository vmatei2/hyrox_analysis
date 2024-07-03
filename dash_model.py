import dash
from dash import dcc
from dash import html

from dash.dependencies import Output
from dash.dependencies import Input

import plotly.express as px
import hyrox_results_analysis as _hra
import data_loading_helpers as _dlh
import pandas as _pd
import constants as _constants


all_race_names = _dlh.list_files_in_directory("/Users/vladmatei/VS_code_files/hyrox_analysis/assets/hyroxData")
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
               'justifyContent': 'center'}),
    html.Div(id='race-info', style={
        'textAlign':'center',
        'maringTop': '20px'
    }),
    dcc.Store(id='race_df')
])
#####    CALLBACK   ######
# first callback, that makes use of the 'Store' component to be then used by subsequent callback in our function
@app.callback(Output('race_df','data'), Input('Race', 'value'))
def load_data(selected_race):
    if selected_race.lower() == _constants.ALL_RACES:
        df = _hra.load_all_races()
    else:
        df = _pd.read_csv(f"/Users/vladmatei/VS_code_files/hyrox_analysis/assets/hyroxData/{selected_race}.csv")
    # data needs to be serialized into a JSON string before being placed into storage
    return df.to_json(date_format='iso', orient='split')
@app.callback(Output('race-info', 'children'), Input('race_df', 'data'), Input('Race', 'value'))
def update_race_info(race_df, selected_race):
    info_message = f"Race:{selected_race}, Number of entries: {len(race_df)}"
    return info_message
# callback function for updating the race plot


if __name__ == '__main__':
    app.run_server(debug=True)
