import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

#  Global variables to be used for run and work/station labels throughout this analysis file!

WORK_LABELS = []
RUN_LABELS = []
ROXZONE_LABELS = []
STATIONS = ['SkiErg', 'SledPush', 'Sled Pull', 'Burpee Broad Jump', 'Rowing', 'Farmers Carry',
                      'Sandbag Lunges', 'Wall Balls']
for i in range(1, 9):
    WORK_LABELS.append('work_' + str(i))
    RUN_LABELS.append('run_' + str(i))
    ROXZONE_LABELS.append('roxzone_' + str(i))

def load():
    """
    Function to load in all data from the folder and return a list of csvs
    :return:
    """
    path2csv = Path("hyroxData")
    csvlist = path2csv.glob("*.csv")
    csvs = [load_one_file(g) for g in csvlist]
    return csvs

def load_one_file(path):
    df = pd.read_csv(path)
    # sort by total time
    df = df.sort_values(by='total_time', ascending=True)
    # reset the index after sorting and add DROP=True to remove the existing index
    df = df.reset_index(drop=True)
    df['Position'] = range(1, len(df) + 1)
    # pre-processing part
    for col in RUN_LABELS + WORK_LABELS + ROXZONE_LABELS + ["total_time"]:
        df[col] = pd.to_timedelta(df[col])
        df[col] = df[col].dt.total_seconds() / 60.0
    # create top x% column
    total_athletes = len(df)
    df['CDF'] = (df.index + 1) / total_athletes
    df['Top Percentage'] = df['CDF'] * 100
    # let's round it
    df['Top Percentage'] = df['Top Percentage'].map(lambda x: np.round(x/5) * 5)
    return df


def round_to_nearest_5(x):
    return  np.round(x / 5)*5

def get_division_entry(df, gender, division):
    subset_df = df[(df['gender'] == gender) & (df['division'] == division)]
    return subset_df

def get_filtered_df(df, column, value, lower_then=True):
    df = df.loc[df[column] < value] if lower_then else df.loc[df[column] > value]
    subset_df = df.sort_values(by='total_time', ascending=True)
    return subset_df

def analyse_race(df):
    # let's plot all the values of each run
    line_plot_runs(df)
    extract_averages(df)
    plot_distributions(df)
    plot_data_points(df, RUN_LABELS, 'Run analysis for Male Open', 'Runs')
    plot_data_points(df, WORK_LABELS, 'Station analysis for Male Open', 'Stations',
                     ['SkiErg', 'SledPush', 'Sled Pull', 'Burpee Broad Jump', 'Rowing', 'Farmers Carry',
                      'Sandbag Lunges', 'Wall Balls'])

    linear_regression_model(df)

def extract_averages(df):
    """
    Function to extract and plot the average time for each of the hyrox stations for the given df
    :param df:
    :return:
    """
    mean_values_run = [df[col].mean() for col in RUN_LABELS]
    mean_values_station = [df[col].mean() for col in WORK_LABELS]
    mean_values_roxzone = [df[col].mean() for col in ROXZONE_LABELS]
    plt.figure(figsize=(10, 6))

    plt.plot(range(1, 9), mean_values_run, 'ro-', label="Mean value for runs")
    plt.plot(range(1, 9), mean_values_station, "gx-", label="Mean value for stations")
    plt.plot(range(1, 9), mean_values_roxzone, 'y+-', label="Mean value for roxzones")

    plt.xlabel('Row/Station')

    xticks = [f'{run}/{station}' for run, station in zip(RUN_LABELS, STATIONS)]

    plt.ylabel('Time (minutes)')
    plt.title('Avg Times for Runs and Stations')

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
    plt.ylabel('Time (minutes)')
    plt.yticks(fontsize=6)
    plt.title(title)

    # Add legend
    plt.legend()

    # Show plot
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return df

def plot_distributions(df):
    fig, axs = plt.subplots(2, 4, figsize=(12, 8))

    for i, ax in enumerate(axs.flat):
        ax.hist(df[f'run_{i+1}'], bins=10,color='blue', alpha=0.5, label=f'Run{i+1}')


        ax.hist(df[f'work_{i+1}'], bins=10,color='red', alpha=0.5, label=STATIONS[i])

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Number of athletes')
        ax.legend()

    plt.title('Time distributions of each run and station')
    plt.tight_layout()
    plt.show()

def line_plot_runs(df):

    run_cols = df.filter(regex=r'^run_\d+$')
    # Transpose the filetered dataframe
    df_transposed = run_cols.transpose()
    plt.figure(figsize=(10, 10))
    i = 0
    for index, row in (run_cols.head(10)).iterrows():
        plt.plot(row, label=f'Position {i + 1}')
        i += 1
    plt.xlabel('Runs')
    plt.ylabel('Run Times (minutes - i.e 4.5 = 4min30s)')
    plt.title('Line Plot of Each Row (Run Columns Only)')
    plt.legend()
    plt.show()

def linear_regression_model(df):
    # CREATE THE X (predictors) and y (target)
    X = df[RUN_LABELS + WORK_LABELS]
    y = df['Top Percentage']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=99)
    model = LogisticRegression()
    model.fit(X_train, y_train)

def svm_model(df):
    clf = svm.SVC(kernel='linear')
    X = df[RUN_LABELS + WORK_LABELS]
    y = df['Top Percentage']
    X_train, X_Test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_Test)

    results_df = pd.DataFrame({
        "y_pred": y_pred,
        "y_test": y_test,
    })

    results_df['Difference'] = abs(y_pred - y_test)
    print("Accuracy: ", metrics.accuracy_score(y_test, y_pred))

def f_importances(coef, names, top=-1):
    imp = coef
    imp, names = zip(*sorted(list(zip(imp, names))))

    # Show all features
    if top == -1:
        top = len(names)
    plt.figure(figsize=(8, 8))
    plt.barh(range(top), imp[::-1][0:top], align='center')
    plt.yticks(range(top), names[::-1][0:top])
    plt.show()

def get_correlation_matrix(df):
    X = df[RUN_LABELS + WORK_LABELS + ['Top Percentage']]
    # get the correlations of each fearures in dataset
    corrmat = X.corr()
    plt.figure(figsize=(10, 10))
    mask = np.triu(np.ones_like(X.corr(), dtype=np.bool_))
    g = sns.heatmap(corrmat, mask=mask, vmin=0, vmax=1, annot=True, cmap='RdYlGn')
    plt.show()

def analyse_all_races():
    # get the data
    all_races = load()
    all_races = pd.concat(all_races, ignore_index=True)
    return all_races

sns.set_style('darkgrid')

all_races = analyse_all_races()
all_male_open = get_division_entry(all_races, 'male', 'open')
# analyse_race(all_male_open)

linear_regression_model(all_male_open)


# london2023 = load_one_file("hyroxData/S5 2023 London.csv")
# london_male_open = get_division_entry(london2023, 'male', 'open')
# male_open_sub_70 = get_filtered_df(london_male_open, column='total_time', value='1:10:00', lower_then=True)
# analyse_race(male_open_sub_70)
#
# male_open_over_90 = get_filtered_df(london_male_open, column='total_time', value='1:30:00', lower_then=False)
# line_plot_runs(male_open_over_90)
#
#
# analyse_race(london_male_open)

