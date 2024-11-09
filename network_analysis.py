from hyrox_results_analysis import load_one_file, extract_mean_values_runs_stations, get_division_entry, \
    plot_data_points, line_plot_runs
import constants as _ct
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import euclidean, pdist, squareform


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


def calculate_performance_ratios(df):
    """
    With this function, we are looking to add several new features to our dataframe to help
    With the calculations for specific ratios
    1. Work (Station) Time to Run Time
    2. Roxzone Time - to Total Time
    3. First Half of the race to second half of the race (all runs up to and including run4 and work 4 vs all run and work stations after)
    :param df:
    :return: df with the 3 new columns mentioned above
    """
    #  1. Work/Station Time to Run Time
    df['work_to_run_ratio'] = df['work_time'] / df['run_time']
    #  2. Roxzone Time to Total Time ratio
    df['roxzone_to_total_ratio'] = df['roxzone_time'] / df['total_time']

    #  3. First half to second half ratio (run and work times)
    first_half_time = df[['run_1', 'run_2', 'run_3', 'run_4', 'work_1', 'work_2', 'work_3', 'work_4']].sum(axis=1)
    second_half_time = df[['run_5', 'run_6', 'run_7', 'run_8', 'work_5', 'work_6', 'work_7', 'work_8']].sum(axis=1)
    df['first_half_to_second_half_ratio'] = first_half_time / second_half_time
    return df


def build_weighted_athlete_network_with_features(df, threshold=10):
    """
    Builds a network of athletes based on similarity in performance metrics, including engineered features.

    :param df: DataFrame containing athlete performance data.
    :param threshold: Minimum similarity score for creating an edge between two athletes.
    :return: A NetworkX graph with athletes as nodes and similarity-based edges.
    """

    # Define weights for each feature
    weights = np.array([
        2,  # weight for work_time
        2,  # weight for roxzone_time
        2,  # weight for run_time
        1, 1, 1, 1, 1, 1, 1, 1,  # weights for individual run segments (run_1 to run_8)
        3,  # weight for work_to_run_ratio
        3,  # weight for roxzone_to_total_ratio
        2  # weight for first_half_to_second_half_ratio
    ])

    weights = np.array([2, 0.3, 2, 1, 0.5, 0.3, 1.5, 3.1, 1, 1, 1, 0.3, 0.1, 0.1, 4])
    metrics = df[['work_time', 'roxzone_time', 'run_time', 'run_1', 'run_2', 'run_3', 'run_4',
                  'run_5', 'run_6', 'run_7', 'run_8', 'work_to_run_ratio', 'roxzone_to_total_ratio',
                  'first_half_to_second_half_ratio', 'total_time']].values * weights
    # scaler = MinMaxScaler()
    # metrics = scaler.fit_transform(metrics)
    # Initialize the graph
    G = nx.Graph()

    # Add nodes with athlete names
    for name in df['name']:
        G.add_node(name)

    for i, name_i in enumerate(df['name']):
        athlete_i_times = metrics[i]
        print(f'done with athlete: {i+1}')
        print((f'{len(df) - i + 1} athletes left'))
        for j in range(i + 1, len(df)):
            athlete_j_times = metrics[j]
            name_j = df['name'].iloc[j]
            euclidead_distance = euclidean(metrics[i], metrics[j])
            if euclidead_distance < threshold:
                G.add_edge(name_i, name_j,
                           weight=1 / euclidead_distance + 1e-5)  # use inverse distance for 'similarity'

    return G




s7_birmingham = load_one_file("assets/hyroxData/S7 2024 Birmingham.csv")
bmham_filtered = get_division_entry(s7_birmingham, "male", "open")
bmham_filtered = calculate_performance_ratios(bmham_filtered)
network = build_weighted_athlete_network_with_features(bmham_filtered, 0.95)


# Position nodes using the spring layout for a more spread-out look
pos = nx.spring_layout(network, seed=42)

# Draw nodes and edges
plt.figure(figsize=(20, 20))
nx.draw_networkx_nodes(network, pos, node_size=50, node_color="skyblue")
nx.draw_networkx_edges(network, pos, width=0.7, alpha=1)
nx.draw_networkx_labels(network, pos, font_size=10)

# Display the visualization
plt.title("Athlete Network Based on Performance Similarity")
plt.show()
