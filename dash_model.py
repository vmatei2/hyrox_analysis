import dash
from dash import dcc, html, ctx
import dash_bootstrap_components as dbc
from dash.dependencies import Output, Input, State
import re
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import hyrox_results_analysis as _hra
import data_loading_helpers as _dlh
import constants as _constants
import helpers as _helpers
import pickle

# Constants and styles
DATA_PATH = "assets/hyroxData"
all_race_names = _dlh.list_files_in_directory(DATA_PATH)

SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "350px",
    "padding": "10px",
    "background-color": "#f8f9fa",
    "overflow-y": "auto",
    "width": "15%",
    "display": "inline-block",
    "vertical-align": "top"
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
        html.P("Please Input Your Own Times For Plotting and Analysis: (Minutes:Seconds Format e.g. 4:05)"),
        dbc.Input(placeholder='Run 1', size="sm", className='mt-2', id=_constants.USER_RUN_1),
        dbc.Input(placeholder='Ski Erg', size="sm", className='mt-2', id=_constants.USER_SKI_ERG),
        dbc.Input(placeholder='Run 2', size="sm", className='mt-2', id=_constants.USER_RUN_2),
        dbc.Input(placeholder='Sled Push', size="sm", className='mt-2', id=_constants.USER_SLED_PUSH),
        dbc.Input(placeholder='Run 3', size="sm", className='mt-2', id=_constants.USER_RUN_3),
        dbc.Input(placeholder='Sled Pull', size="sm", className='mt-2', id=_constants.USER_SLED_PULL),
        dbc.Input(placeholder='Run 4', size="sm", className='mt-2', id=_constants.USER_RUN_4),
        dbc.Input(placeholder='Burpee Broad Jump', size="sm", className='mt-2', id=_constants.USER_BURPEE_BROAD_JUMP),
        dbc.Input(placeholder='Run 5', size="sm", className='mt-2', id=_constants.USER_RUN_5),
        dbc.Input(placeholder='Row Erg', size="sm", className='mt-2', id=_constants.USER_ROW_ERG),
        dbc.Input(placeholder='Run 6', size="sm", className='mt-2', id=_constants.USER_RUN_6),
        dbc.Input(placeholder='Farmers Carry', size="sm", className='mt-2', id=_constants.USER_FARMERS_CARRY),
        dbc.Input(placeholder='Run 7', size="sm", className='mt-2', id=_constants.USER_RUN_7),
        dbc.Input(placeholder='Sandbag Lunges', size="sm", className='mt-2', id=_constants.USER_SANDBAG_LUNGES),
        dbc.Input(placeholder='Run 8', size="sm", className='mt-2', id=_constants.USER_RUN_8),
        dbc.Input(placeholder='Wall Balls', size="sm", className='mt-2', id=_constants.USER_WALL_BALLS),

        dbc.Button(
            "Analyse my times", outline=True, id="analyse_button", color="info", className="mt-2", disabled=True
        ),
        html.P("", id="fill_all_inputs")
    ],
    style=SIDEBAR_STYLE,
    id='sidebar'  # assigning id for CSS targeting
)

# Tabs Layout
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.LUX, "/assets/styles.css"])
server = app.server

app.layout = html.Div([
    html.Div(sidebar),
    html.Div(
        [
            dbc.Modal(
                [
                    dbc.ModalHeader(dbc.ModalTitle("Wrong Input")),
                    dbc.ModalBody(
                        "Please ensure all times are inputted and follow the expected format (e.g. 4:50)"),
                    dbc.ModalFooter(
                        dbc.Button("Close", id="close", className="ms-auto", n_clicks=0)
                    )
                ],
                id="modal",
                is_open=False
            ),
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
            ]),
            dbc.Row([
                dbc.Col(
                    html.Div(children=[
                        dcc.Loading(
                            id="loading-run-distribution-graph",
                            type="circle",
                            children=dcc.Graph(figure={}, id="run_distribution_graph")
                        )
                    ]), width=6
                ),
                dbc.Col(
                    html.Div(children=[
                        dcc.Loading(
                            id='loading-station-distribution-graph',
                            type="circle",
                            children=dcc.Graph(figure={}, id="station_distribution_graph")
                        )
                    ])
                )
            ]),
            dbc.Row([
                dbc.Col([
                    html.H4("Your predicted percentile finish ", style={'textAlign': 'center'}),
                    dcc.Loading(
                        id='loading-gauge',
                        type="circle",
                        children=dcc.Graph(
                            id='gauge_graph',
                            figure=go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=0,  # Sample value, you can set this dynamically based on your data
                                title={'text': "Percentile Finish"},
                                gauge={
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': "#f5a623"},
                                    'steps': [
                                        {'range': [0, 50], 'color': "#e6e6e6"},
                                        {'range': [50, 100], 'color': "#c8d6e5"}
                                    ],
                                }
                            )),
                        )
                    )
                ], width=12)
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

@app.callback(
    [Output("fill_all_inputs", "children"), Output("fill_all_inputs", "style"), Output("analyse_button", 'disabled')],
    [Input(i, 'value') for i in _constants.ALL_USER_INPUTS],

)
def toggle_button(*values):
    """
    Function to toggle analyse my times button on if all values to be passed by user are filled in
    :param values:
    :param is_open:
    :return:
    """
    if all(values):
        return "", {"display": "none"}, False
    else:
        return "Pleaes fill in all inputs", {"display": "block", "color": "red"}, True


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


@app.callback(Output('filtered_df', 'data'),
              [Input('race_df', 'data'), Input('Division', 'value'), Input('Gender', 'value'),
               Input('top_percentile_slider', 'value')])
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


@app.callback(Output('race_name', 'children'), Output('participants', 'children'),
              [Input('filtered_df', 'data'), Input('Race', 'value')])
def update_race_info(filtered_df, selected_race):
    """Update race information based on the loaded race data."""
    try:
        filtered_df = pd.read_json(filtered_df)
        race_name = selected_race
        participants = f'Total Participants: {len(filtered_df)}'
        return race_name, participants
    except Exception as e:
        return "Race: N/A, Number of entries: N/A"


@app.callback(Output('race_graph', 'figure'),
              Output("modal", "is_open"),
              Input('filtered_df', 'data'),
              Input('analyse_button', 'n_clicks'),
              [State(i, 'value') for i in _constants.ALL_USER_INPUTS],
              State("modal", "is_open"))
def update_graph(filtered_df, analyse_button, *values):
    """Update race graph based on the loaded race data."""
    try:
        df = pd.read_json(filtered_df)
        mean_value_runs, mean_value_stations = _hra.extract_mean_values_runs_stations(df)
        # start at 1 to avoid 0-indexing
        x_vals = np.arange(start=1, stop=len(mean_value_runs) + 1)
        fig = go.Figure(
            data=[
                go.Scatter(x=x_vals, y=mean_value_runs, name='Runs', text=_constants.RUN_LABELS, mode="lines+text",
                           textposition='top center'),
                go.Scatter(x=x_vals, y=mean_value_stations, name='Stations', text=_constants.STATIONS,
                           mode="lines+text",
                           textposition='top center')
            ],
            layout={"xaxis": {"title": "Runs/Stations Numbers"}, "yaxis": {"title": "Time (Minutes)"},
                    "title": "Average Times Analysis"}
        )

        ctx_clicked = ctx.triggered_id
        modal_open = False
        if ctx_clicked == "analyse_button":
            values = [entry for entry in values if type(entry) == str]
            if _validate_time_format(values):
                # here please validate user input and ensure everything filled in and correct
                # let's extract the user's runs
                user_runs, user_stations = _extract_runs_stations(values)
                # we have one entry that is a run, one that is a station - i.e values[0] = run_1, values[2] = run_2, values[1] = ski_erg, values[3] = sled_push etc..
                user_runs = _hra.convert_string_times_to_model_inputs(user_runs)
                user_stations = _hra.convert_string_times_to_model_inputs(user_stations)
                fig.add_trace(
                    go.Scatter(x=x_vals, y=user_runs, name='Your run times', mode="lines+text",
                               textposition='top center')
                )
                fig.add_trace(
                    go.Scatter(x=x_vals, y=user_stations, name='Your station times', mode="lines+text",
                               textposition='top center'))
            else:
                modal_open = True

        return fig, modal_open
    except Exception as e:
        return go.Figure(), modal_open  # Return an empty figure in case of an error



@app.callback(Output('run_distribution_graph', 'figure'),
              Output('station_distribution_graph', 'figure'),
              Output('gauge_graph', 'figure'),
              Input('filtered_df', 'data'),
              Input('analyse_button', 'n_clicks'),
              [State(i, 'value') for i in _constants.ALL_USER_INPUTS])
def update_distribution_graphs(filtered_df, n_clicks, *values):
    try:
        df = pd.read_json(filtered_df)
        run_data_points = []
        station_data_points = []

        for i, column_name in enumerate(_constants.RUN_LABELS):
            run_data_points.append(df[column_name])
            station_data_points.append(df[_constants.WORK_LABELS[i]])
        run_data_points = [sorted(entry) for entry in run_data_points]
        station_data_points = [sorted(entry) for entry in station_data_points]

        # Create distribution figures using the helper function
        run_distribution_fig = _helpers.create_distribution_figure(
            run_data_points, _constants.RUN_LABELS, 'Laps', 'Time (minutes)', 'Lap Times Scatter Plot'
        )
        station_distribution_fig = _helpers.create_distribution_figure(
            station_data_points, _constants.STATIONS, 'Stations', 'Time (minutes)', 'Station Times Scatter Plot'
        )
        ctx_clicked = ctx.triggered_id
        if ctx_clicked == "analyse_button":
            if _validate_time_format(time_list=values):
                user_runs, user_stations = _extract_runs_stations(values)
                user_runs = _hra.convert_string_times_to_model_inputs(user_runs)
                user_stations = _hra.convert_string_times_to_model_inputs(user_stations)

                x_vals = [i for i in range(8)]
                run_distribution_fig.add_trace(
                    (
                        go.Scatter(x=x_vals, y=user_runs, name='Your run times', mode="lines+text",
                                   textposition='top center')
                    )
                )
                station_distribution_fig.add_trace(
                    (
                        go.Scatter(x=x_vals, y=user_stations, name='Your station times', mode="lines+text",
                                   textposition='top center')
                    )
                )
                # if we are in the case where the user has clicked the analyse button, then we are also interested in updating the 'gauge' graph
                gauge_graph = _predict_percentile_finish_figure(user_runs, user_stations)
                return run_distribution_fig, station_distribution_fig, gauge_graph
        else:
            return run_distribution_fig, station_distribution_fig, go.Figure()

    except Exception as e:
        return go.Figure(), go.Figure(), go.Figure()  # Return an empty figure in case of an error


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


####  PRIVATE API ####
def _predict_percentile_finish_figure(user_runs, user_stations):
    #  load in the random forest classifier
    rf_classifier_path = "rf_classifier.sav"
    with open(rf_classifier_path, 'rb') as file:
        model = pickle.load(file)
    runs_stations = user_runs + user_stations
    runs_stations = np.array(runs_stations)
    percentile_finish = model.predict(runs_stations.reshape(1, -1))
    updated_figure = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percentile_finish[0],
        title={'text': "Percentile Finish"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#f5a623"},
            'steps': [
                {'range': [0, 50], 'color': "#e6e6e6"},
                {'range': [50, 100], 'color': "#c8d6e5"}
            ],
        }
    ))
    return updated_figure



def _extract_runs_stations(values):
    user_runs = []
    user_stations = []
    for i, entry in enumerate(values):
        if i % 2 == 0:
            user_runs.append(entry)
        else:
            user_stations.append(entry)
    return user_runs, user_stations


def _validate_time_format(time_list):
    # Define a regex pattern for "minutes:seconds"
    pattern = re.compile(r'^\d+:[0-5]\d$')

    # Iterate through each time string in the list
    for time_str in time_list:
        # Check if the string matches the pattern
        if not pattern.match(time_str):
            return False  # Return False if any string does not match
    return True  # Return True if all strings are valid


if __name__ == '__main__':
    app.run_server(debug=True, port=8000)
