from hyrox_results_analysis import load_one_file, extract_mean_values_runs_stations, get_division_entry, \
    plot_data_points, line_plot_runs
import constants as _ct
import matplotlib.pyplot as plt
import seaborn as sns


def select_user(df, name):
    """
    Extract the subset of rows mapping to a given username
    :param df:
    :param name:
    :return:
    """
    user_df = df[df["name"].str.contains(name, case=False, na=False)]
    return user_df


def line_plot_user(df, average_runs, top20_runs):
    user_runs = df[_ct.RUN_LABELS]
    values = user_runs.iloc[0]
    values.plot(kind='line', marker='o', figsize=(10, 6), title="Run Segments", label=df["name"].item())
    plt.plot(_ct.RUN_LABELS, average_runs, marker='o', label="Average runs of all male-open")
    plt.plot(_ct.RUN_LABELS, top20_runs, marker='o', label="Average runs of the top 20 times in the race")
    plt.xlabel("Run Segments")
    plt.ylabel("Values")
    plt.legend()
    plt.show()


s7_birmingham = load_one_file("assets/hyroxData/S7 2024 Birmingham.csv")
bmham_filtered = get_division_entry(s7_birmingham, "male", "open")
top_20 = bmham_filtered.head(20)
sns.set_style('darkgrid')
user_df = select_user(bmham_filtered, "ingham, james ")
average_runs, average_stations = extract_mean_values_runs_stations(bmham_filtered)
top20_runs, top20_stations = extract_mean_values_runs_stations(top_20)

line_plot_user(user_df, average_runs, top20_runs)
