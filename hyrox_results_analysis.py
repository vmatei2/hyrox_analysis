import pandas as pd
import glob
from os import listdir
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns


def load():
    path2csv = Path("hyroxData")
    csvlist = path2csv.glob("*.csv")
    csvs = [pd.read_csv(g) for g in csvlist]
    return csvs

def load_one_file(path):
    return pd.read_csv(path)

def get_division_entry(df, gender, division):
    subset_df =  df[(df['gender'] == gender) & (df['division'] == division)]
    return subset_df

def get_filtered_df(df, column, value):
    subset_df = df.loc[df[column] < value]
    subset_df = subset_df.sort_values(by='total_time', ascending=True)
    return subset_df
def analyse_race(df):
    df = df.sort_values(by='total_time', ascending=True)
    df = get_filtered_df(df, column='total_time', value='1:10:00')
    # let's plot all the values of each run

    work_labels = []
    run_labels = []
    for i in range(1, 9):
        work_labels.append('work_' + str(i))
        run_labels.append('run_' + str(i))
    plot_data_points(df, run_labels, 'Run analysis for Male Open, under 1:10', 'Runs')
    plot_data_points(df, work_labels, 'Station analysis for Male Open, under 1:10', 'Stations')

def plot_data_points(df, columns_to_extract, title, xlabel, x_labels=None):
    data_points = []
    for column_name in columns_to_extract:
        data_points.append(df[column_name])
    data_points = [sorted(entry) for entry in data_points]
    # Plotting
    plt.figure(figsize=(14, 16))
    if x_labels == None:
        x_labels = columns_to_extract
    for i, data in enumerate(data_points):
        plt.scatter([i] * len(data), data, label=f'Lap {i+1}')
    # Set x-axis labels
    plt.xticks(range(len(data_points)), x_labels)

    # Set labels and title
    plt.xlabel(xlabel)
    plt.ylabel('Time')
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
barcelona2023 = load_one_file("hyroxData/S5 2023 Barcelona.csv")
types = barcelona2023.dtypes
barca_entry = get_division_entry(barcelona2023, 'male', 'open')
analyse_race(barca_entry)

stopHere = 0


