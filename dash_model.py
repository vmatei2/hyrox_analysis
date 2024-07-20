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
    "width": "350px",
    "padding": "10px",
    "background-color": "#f8f9fa",
    "overflow-y": "auto"
}

CONTENT_STYLE = {
    "margin-left": "350px",
    "padding": "20px"
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
        html.P('Top percentile of athletes: '),

        dcc.Slider(
            10, 100, 10,
            value=100,
            id='top_percentile_slider'
        ),

        html.P("Please Input Your Own Times For Plotting and Analysis:"),

        dbc.Input(placeholder='Run 1', size="sm", className='mt-2'),
        dbc.Input(placeholder='Ski Erg', size="sm", className='mt-2'),
        dbc.Input(placeholder='Run 2', size="sm", className='mt-2'),
        dbc.Input(placeholder='Sled Push', size="sm", className='mt-2'),
        dbc.Input(placeholder='Run 3', size="sm", className='mt-2'),
        dbc.Input(placeholder='Sled Pull', size="sm", className='mt-2'),
        dbc.Input(placeholder='Run 4', size="sm", className='mt-2'),
        dbc.Input(placeholder='Burpee Broad Jump', size="sm", className='mt-2'),
        dbc.Input(placeholder='Run 5', size="sm", className='mt-2'),
        dbc.Input(placeholder='Row Erg', size="sm", className='mt-2'),
        dbc.Input(placeholder='Run 6', size="sm", className='mt-2'),
        dbc.Input(placeholder='Farmers Carry', size="sm", className='mt-2'),
        dbc.Input(placeholder='Run 7', size="sm", className='mt-2'),
        dbc.Input(placeholder='Sandbag Lunges', size="sm", className='mt-2'),
        dbc.Input(placeholder='Run 8', size="sm", className='mt-2'),
        dbc.Input(placeholder='Wall Balls', size="sm", className='mt-2'),



    ],
    style=SIDEBAR_STYLE
)

# Application layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX])

app.layout = html.Div([
    html.Div(sidebar),
    html.Div(
        [
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
                dbc.Col(
                    dbc.Row([
                        dbc.Col(
                            dcc.Loading(
                                id="loading-race-info",
                                type="circle",
                                children=dbc.Card(
                                    [
                                        dbc.CardHeader("Race Information"),
                                        dbc.CardBody([
                                            html.P(className="card-title", id="race_name"),
                                            html.P(className="card-text", id="participants")
                                        ])
                                    ]
                                )
                            ), width=4
                        ),
                        dbc.Col(
                            dcc.Loading(
                                id="loading-fastest-time",
                                type="circle",
                                children=dbc.Card(
                                    [
                                        dbc.CardHeader("Fastest Race Time"),
                                        dbc.CardBody([
                                            html.P(className='card-text', id="fastest")
                                        ])
                                    ]
                                )
                            ), width=4
                        ),
                        dbc.Col(
                            dcc.Loading(
                                id="loading-average-time",
                                type="circle",
                                children=dbc.Card(
                                    [
                                        dbc.CardHeader("Average Race Time"),
                                        dbc.CardBody([
                                            html.P(className="card-text", id="average")
                                        ])
                                    ]
                                )
                            ), width=4
                        ),
                    ]), width=12
                )
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div(children=[
                        dcc.Loading(
                            id="loading-race-graph",
                            type="circle",
                            children=dcc.Graph(figure={}, id="race_graph")
                        )
                    ]), width=12
                )
            ])
        ],
        style=CONTENT_STYLE
    ),
    dcc.Loading(
        id="loading-race-df",
        type="circle",
        children=dcc.Store(id='race_df')
    ),
    dcc.Loading(
        id="loading-filtered-df",
        type="circle",
        children=dcc.Store(id='filtered_df')
    )
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

@app.callback(Output('filtered_df', 'data'), [Input('race_df', 'data'), Input('Division', 'value'), Input('Gender', 'value'), Input('top_percentile_slider', 'value')])
def filter_df(race_df, division, gender, top_percentile):
    """
    Filter the race_df based on user selections
    :return:
    """
    try:
        race_df = pd.read_json(race_df)
        # let's create the filter based on what the user has requested - first we create a filter that is always true - in case user has requested all values
        # create a copy - more for debugging reasons as would like to keep the reference to the original one - can be optimised if causing issues in the future - small dataset so unlikely
        filtered_df = race_df.copy()
        if division != _constants.REQUEST_ALL_VALUES:
            filtered_df = filtered_df[filtered_df['division'] == division]

        if gender != _constants.REQUEST_ALL_VALUES:
            filtered_df = filtered_df[filtered_df['gender'] == gender]
        filtered_df = filtered_df[(filtered_df['CDF'] * 100) <= top_percentile]
        return filtered_df.to_json()
    except Exception as e:
        return pd.DataFrame().to_json()  # return an empty DataFrame in case of an error

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

@app.callback(Output('race_name', 'children'), Output('participants', 'children'), [Input('filtered_df', 'data'), Input('Race', 'value')])
def update_race_info(filtered_df, selected_race):
    """Update race information based on the loaded race data."""
    try:
        filtered_df = pd.read_json(filtered_df)
        race_name = selected_race
        participants = f'Total Participants: {len(filtered_df)}'
        return race_name, participants
    except Exception as e:
        return "Race: N/A, Number of entries: N/A"

@app.callback(Output('race_graph', 'figure'), Input('filtered_df', 'data'))
def update_graph(filtered_df):
    """Update race graph based on the loaded race data."""
    try:
        df = pd.read_json(filtered_df)
        mean_value_runs, mean_value_stations = _hra.extract_mean_values_runs_stations(df)
        x_vals = np.arange(len(mean_value_runs))
        fig = go.Figure(
            data=[
                go.Scatter(x=x_vals, y=mean_value_runs, name='Runs', text=_constants.RUN_LABELS, mode="markers+text",
                           textposition='top center'),
                go.Scatter(x=x_vals, y=mean_value_stations, name='Stations', text=_constants.STATIONS,
                           mode="markers+text",
                           textposition='top center')
            ],
            layout={"xaxis": {"title": "Runs/Stations Numbers"}, "yaxis": {"title": "Time (Minutes)"},
                    "title": "Average Times Analysis"}
        )
        return fig
    except Exception as e:
        return go.Figure()  # Return an empty figure in case of an error

@app.callback(Output('fastest', 'children'), Output('average', 'children'), Input('filtered_df', 'data'))
def update_card_displays(filtered_df):
    """
    Update the card display extracting the overall average time and the fastest time from the filtered_df
    :param filtered_df:
    :return:
    """
    try:
        filtered_df = pd.read_json(filtered_df)
        # get the first entry in the df given they have been sorted from fastest to slowest
        fastest = round(filtered_df.head(1)['total_time'], 2).values[0]
        fastest_text = f'{str(fastest)} minutes'
        average = round(filtered_df['total_time'].mean(), 2)
        average_text = f'{str(average)} minutes'
        return fastest_text, average_text
    except Exception as e:
        return f"Exception caught when extracting fastest and average times from filtered df: {e}"

if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
