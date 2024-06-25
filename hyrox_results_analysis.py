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
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import plot_tree
import pickle

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
    """
    Function to simply load one Hyrox race, given a specified file path to a .csv file
    :param path:
    :return:
    """
    # quick short-circuit to exit the function while retrieving 2024 data, but still working on 2023 analysis
    print(path)
    df = pd.read_csv(path)
    # sort by total time
    df = df.sort_values(by='total_time', ascending=True)
    # reset the index after sorting and add DROP=True to remove the existing index
    df = df.reset_index(drop=True)
    df['Position'] = range(1, len(df) + 1)
    # pre-processing part
    for col in RUN_LABELS + WORK_LABELS + ROXZONE_LABELS + ["total_time"]:
        df[col] = pd.to_timedelta(df[col])
        # convert to minutes
        df[col] = df[col].dt.total_seconds() / 60.0
    # create top x% column
    total_athletes = len(df)
    df['CDF'] = (df.index + 1) / total_athletes
    df['Top Percentage'] = df['CDF'] * 100
    # let's round it
    df['Top Percentage'] = df['Top Percentage'].map(lambda x: np.round(x / 10.0) * 10)
    return df


def round_to_nearest_5(x):
    return np.round(x / 5) * 5


def get_division_entry(df, gender, division):
    subset_df = df[(df['gender'] == gender) & (df['division'] == division)]
    return subset_df


def get_filtered_df(df, column, value, lower_then=True):
    df = df.loc[df[column] < value] if lower_then else df.loc[df[column] > value]
    subset_df = df.sort_values(by='total_time', ascending=True)
    return subset_df


def analyse_race(df):
    """
    Function to analyse various aspects of the race data, including run times,
    distributions and averages across each station and run
    :param df:
    :return:
    """
    line_plot_runs(df)
    scatter_plot_averages(df)
    plot_distributions(df)
    plot_data_points(df, RUN_LABELS, 'Run analysis for Male Open', 'Runs')
    plot_data_points(df, WORK_LABELS, 'Station analysis for Male Open', 'Stations',
                     STATIONS)


def scatter_plot_averages(df):
    """
    Function to extract and plot the average time for each of the hyrox stations for the given df
    :param df:
    :return:
    """
    mean_values_run = []
    mean_values_station = []
    mean_values_roxzone = []
    run_count = 0
    station_count = 0
    for i in range(16):
        if i%2 == 0:
            mean_values_run.append(df[RUN_LABELS[run_count]].mean())
            mean_values_roxzone.append(df[ROXZONE_LABELS[run_count]].mean())
            mean_values_station.append(0)
            run_count += 1
        else:
            mean_values_run.append(0)
            mean_values_roxzone.append(0)
            mean_values_station.append(df[WORK_LABELS[station_count]].mean())
            station_count += 1
    # mean_values_run = [df[col].mean() for col in RUN_LABELS]
    # mean_values_station = [df[col].mean() for col in WORK_LABELS]
    # mean_values_roxzone = [df[col].mean() for col in ROXZONE_LABELS]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot mean values for runs and roxzones on the first x-axis
    ax1.plot(range(1, 17), mean_values_run, 'ro', label='Mean Value for Runs')
    ax1.plot(range(1, 17), mean_values_roxzone, 'y+', label='Mean Value for Roxzones')

    # Set labels for the first x-axis and y-axis
    ax1.set_xlabel('Runs')
    ax1.set_ylabel('Time (minutes)')
    ax1.set_title('Average Times for Runs, Stations, and Roxzones')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    # Create the second x-axis for stations
    ax2 = ax1.twiny()
    offset = 5
    ax2.plot(np.array(range(1, 17)), mean_values_station, "gx", label="Mean Value for Stations")
    ax2.set_xlabel('Stations')
    ax2.tick_params(axis='x')

    # Customize x-ticks for both x-axes
    xticks_run_roxzone = [f'Run {i+1}/RZ{i+1}' for i in range(8)]
    xticks_stations = [f'Station {i+1}' for i in range(8)]
    ax1.set_xticks(range(1, 16, 2))
    ax1.set_xticklabels(xticks_run_roxzone, rotation=45)
    ax2.set_xticks(np.array(range(2, 17, 2)))
    ax2.set_xticklabels(xticks_stations, rotation=45)
    ax1.set_ylim([0.2, 9])
    # Add legend for the second x-axis
    ax2.legend(loc='upper right')

    # Adjust the layout to prevent overlap
    fig.tight_layout()

    plt.show()


def line_plot_averages(df):
    """
    Function to line plot averages of top 20 athletes against the averages of all athletes and those of the bottom 20
    The main aim of this function is to search for common patterns that can help separate the best performing athletes in a race from the bottom ones
    :param df:
    :return:
    """
    top_20 = df.head(20)
    bottom_20 = df.tail(20)
    top_20_runs, top_20_stations = extract_mean_values_runs_stations(top_20)
    bottom_20_runs, bottom_20_stations = extract_mean_values_runs_stations(bottom_20)
    average_runs, average_stations = extract_mean_values_runs_stations(df)

    # Plot the different values for runs
    plot_top_bottom_average(top_20_runs, bottom_20_runs, average_runs, 'run')

    # Plot the different values for stations
    plot_top_bottom_average(top_20_stations, bottom_20_stations, average_stations, 'station')

def plot_top_bottom_average(top_20, bottom_20, average, run_or_stations):
    """
    Function to plot the avearge values for top 20, bottom 20 and all athletes on runs or stations
    :param top_20:
    :param bottom_20:
    :param average:
    run_or_station: str -- value to be used in label
    :return:
    """
    plt.figure(figsize=(6, 6 ))
    average_of_top_20 = sum(top_20) / len(top_20)
    average_of_bottom_20 = sum(bottom_20) / len(bottom_20)
    average_of_all = sum(average) / len(average)
    average_of_top_20_list = [average_of_top_20 for x in top_20]
    average_of_bottom_20_list = [average_of_bottom_20 for x in bottom_20]
    average_of_all_list = [average_of_all for x in average]
    plt.plot(range(1, 9), top_20, 'ro-', label=f'Average {run_or_stations} times of top 20 athletes')
    plt.plot(range(1, 9), average_of_top_20_list, 'r--', label="Average time of the top 20")
    plt.plot(range(1, 9), bottom_20, 'go-', label=f'Average {run_or_stations} times of bottom 20 athletes')
    plt.plot(range(1, 9), average_of_bottom_20_list, 'g--', label="Average time of bottom 20")
    plt.plot(range(1, 9), average, 'yo-', label=f'Average {run_or_stations} times of all athletes')
    plt.plot(range(1, 9), average_of_all_list,'y--', label="Average time all athletes ")
    plt.grid(True)
    plt.legend()
    plt.xlabel("")
    if run_or_stations == "run":
        xticks = [f'{run}' for run in RUN_LABELS]
    elif run_or_stations == "station":
        xticks = [f'{station}' for station in STATIONS]
    plt.xticks(range(1, 9), xticks, rotation=45)
    plt.ylabel("Time (Minutes)", fontsize=12)
    plt.show()



def extract_mean_values_runs_stations(df):
    """
    Function to extract the mean values on both runs and stations
    This is to be used as we aim to analyse differences between top 20 / bottom 20 / and all people in a Hyrox race - avoid having repeated code
    :param df:
    :return:
    """
    mean_value_runs = [df[col].mean() for col in RUN_LABELS]
    mean_value_stations = [df[col].mean() for col in WORK_LABELS]
    return mean_value_runs, mean_value_stations

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
        ax.hist(df[f'run_{i + 1}'], bins=10, color='blue', alpha=0.5, label=f'Run{i + 1}')

        ax.hist(df[f'work_{i + 1}'], bins=10, color='red', alpha=0.5, label=STATIONS[i])

        ax.set_xlabel('Time (minutes)')
        ax.set_ylabel('Number of athletes')
        ax.legend()

    plt.title('Time distributions of each run and station')
    plt.tight_layout()
    plt.show()


def line_plot_runs(df):
    """
    Function to line plot all runs of top 10 athletes
    Main point behind this function is to explore whether we can see a pattern emerging in how the top athletes in Hyrox run across a race?
    :param df:
    :return:
    """

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


def random_forest_classifier(df, save_as_name):
    X = df[RUN_LABELS + WORK_LABELS]
    y = df['Top Percentage']
    random_state = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    rf = RandomForestClassifier(random_state=random_state)
    params = {
        'max_depth': [2, 5, 12],
        'min_samples_leaf': [5, 20, 100],
        'n_estimators': [10, 25, 50]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=params, cv=3, verbose=1, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    print(f"Best score found through grid search was {grid_search.best_score_}")
    rf_classifier = grid_search.best_estimator_
    filename = save_as_name + ".sav"
    try:
        pickle.dump(rf_classifier, open(filename, 'wb'))
        print('have succesfully saved model')
    except Exception as e:
        print('Unfortunately caught exception: ', e)

    plt.figure(figsize=(20, 12))
    plot_tree(rf_classifier.estimators_[5], feature_names=X.columns, class_names=list(str(val) for val in y.unique()),
              fontsize=12)
    plt.show()


def analyse_rf_classifier(df, rf_classifier):
    X = df[RUN_LABELS + WORK_LABELS]
    y = df['Top Percentage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=20)
    result = rf_classifier.score(X_test, y_test)
    y_pred = rf_classifier.predict(X_test)
    print(result)
    feature_names = RUN_LABELS + STATIONS
    importances = pd.Series(rf_classifier.feature_importances_, index=feature_names)
    importances_sorted = importances.sort_values(ascending=False)
    plt.figure(figsize=(6, 6))
    sns.barplot(x=importances_sorted.values, y=importances_sorted.index, palette='viridis')
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance")

    plt.show()


def predict_my_race(rf_model, run_station_times):
    """
    Function to predict an athlete's percentile finish within a Hyrox race
    :param rf_model: the trained machine learning model (assumed to be RF in the context of this project - technically function could take any type of model as input)
    :param run_station_times: array of float values representing an athlete's times on the stations and the runs
    :return:
    """
    run_station_times = convert_string_times_to_model_inputs(run_station_times)
    run_station_times = np.array(run_station_times)
    prediction = rf_model.predict(run_station_times.reshape(1, -1))
    print(f"Your predicted percentile result given your run and station times is: {prediction}")
    return prediction


def convert_string_times_to_model_inputs(times):
    """
    Function to convert the times as a list of strings to a list of values the model expects (i.e 4mins30s = 4.5)
    :param times:
    :return:
    """
    float_list = []
    for time_str in times:
        minutes, seconds = map(float, time_str.split(":"))
        total_minutes = minutes + seconds / 60
        float_list.append(total_minutes)
    return float_list


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


def load_all_races():
    """
    Wrapper on top of the load function to concat and return one dataframe containing all race data
    :return:
    """
    # get the data
    all_races = load()
    all_races = pd.concat(all_races, ignore_index=True)
    return all_races

def load_classifier(classifier_path:str):
    """
    Function to load in the classifier model that's been trained and saved down
    :param clasifier_path: path to the classifier
    :return: loaded model
    """
    rf_classifier = pickle.load(open(classifier_path, 'rb'))
    return rf_classifier


sns.set_style('darkgrid')
# all_races = load_all_races()
# all_male_open = get_division_entry(all_races, 'male', 'open')
# analyse_race(all_male_open)
# s5_london = load_one_file("hyroxData/S5 2023 London.csv")
# s5_london_male_open = get_division_entry(s5_london, 'male', 'open')

all_races = load_all_races()
classifier = load_classifier("rf_classifier.sav")
analyse_rf_classifier(all_races, classifier)
