import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

#  Global variables to be used for run and work/station labels throughout this analysis file!

WORK_LABELS = []
RUN_LABELS = []
STATIONS = ['SkiErg', 'SledPush', 'Sled Pull', 'Burpee Broad Jump', 'Rowing', 'Farmers Carry',
                      'Sandbag Lunges', 'Wall Balls']
for i in range(1, 9):
    WORK_LABELS.append('work_' + str(i))
    RUN_LABELS.append('run_' + str(i))

def load():
    path2csv = Path("hyroxData")
    csvlist = path2csv.glob("*.csv")
    csvs = [pd.read_csv(g) for g in csvlist]
    return csvs

def load_one_file(path):
    return pd.read_csv(path)

def get_division_entry(df, gender, division):
    subset_df = df[(df['gender'] == gender) & (df['division'] == division)]
    return subset_df

def get_filtered_df(df, column, value):
    df = df.sort_values(by='total_time', ascending=True)
    subset_df = df.loc[df[column] < value]
    subset_df = subset_df.sort_values(by='total_time', ascending=True)
    return subset_df


def analyse_race_under_seventy(df):
    df = get_filtered_df(df, column='total_time', value='1:10:00')

    df['total_time'] = pd.to_timedelta(df['total_time'])


    # let's plot all the values of each run
    extract_averages(df)
    plot_data_points(df, RUN_LABELS, 'Run analysis for Male Open, under 1:10', 'Runs')
    plot_data_points(df, WORK_LABELS, 'Station analysis for Male Open, under 1:10', 'Stations',
                     ['SkiErg', 'SledPush', 'Sled Pull', 'Burpee Broad Jump', 'Rowing', 'Farmers Carry',
                      'Sandbag Lunges', 'Wall Balls'])

def extract_averages(df):
    """
    Function to extract and plot the average time for each of the hyrox stations for the given df
    :param df:
    :return:
    """

    for col in RUN_LABELS + WORK_LABELS:
        df[col] = pd.to_timedelta(df[col])
        df[col] = df[col].dt.total_seconds()

    mean_values_run = [df[col].mean() for col in RUN_LABELS]
    mean_values_station = [df[col].mean() for col in WORK_LABELS]

    plt.figure(figsize=(10, 6))

    plt.plot(range(1, 9), mean_values_run, 'ro-', label="Mean value for runs")
    plt.plot(range(1, 9), mean_values_station, "gx-", label="Mean value for stations")

    plt.xlabel('Row/Station')

    xticks = [f'{run}/{station}' for run, station in zip(RUN_LABELS, STATIONS)]

    plt.ylabel('Time (s)')
    plt.title('Mean Values for Rows and Stations')

    plt.xticks(range(1, 9), xticks, rotation=45)
    plt.grid(True)
    plt.legend()

    plt.show()

def extract_general_statistics(df):
    df['total_time'] = pd.to_timedelta(df['total_time'])

    summary_stats = df['total_time'].describe()
    print("Summary Statistics Overall: ")
    print(summary_stats)

    top_80_percentile = df['total_time'].quantile(0.8)
    print('Top 80% percentile: ', top_80_percentile)


def plot_data_points(df, columns_to_extract, title, xlabel, x_labels=None):
    data_points = []
    for column_name in columns_to_extract:
        data_points.append(df[column_name])
    data_points = [sorted(entry) for entry in data_points]
    # Plotting
    plt.figure(figsize=(12, 6))
    if x_labels == None:
        x_labels = columns_to_extract
    for i, data in enumerate(data_points):
        plt.scatter([i] * len(data), data, label=f'Lap {i + 1}')
    # Set x-axis labels
    plt.xticks(range(len(data_points)), x_labels)
    num_ticks = len(plt.gca().get_yticks())

    # Set labels and title
    plt.xlabel(xlabel)
    plt.ylabel('Time (s)')
    plt.yticks(fontsize=6)
    plt.title(title)

    # Add legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df




sns.set_style('darkgrid')
barcelona2023 = load_one_file("hyroxData/S5 2023 London.csv")
types = barcelona2023.dtypes
barca_entry_male_open = get_division_entry(barcelona2023, 'male', 'open')

analyse_race_under_seventy(barca_entry_male_open)
