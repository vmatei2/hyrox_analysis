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
import network_helpers as net_help
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

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
    #  3. Run 1 to Run 8 -- how much can people hold their pace?
    df['run1_to_run8'] = df['run_1'] / df['run_8']
    df['sled_oush_to_sled_pull'] = df['work_2'] / df['work_3']


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

    # Prepare data
    metrics = df[['roxzone_time', 'total_time', 'work_2', 'work_3', 'work_4', 'work_to_run_ratio', 'roxzone_to_total_ratio',
                  'first_half_to_second_half_ratio']].values

    # Calculate all pairwise Euclidean distances
    distances = pdist(metrics, metric='euclidean')
    distance_matrix = squareform(distances)  # Convert to a square matrix format

    # Initialize the graph and add nodes
    G = nx.Graph()
    for name in df['name']:
        G.add_node(name)

    #  Helper function to add edges based on the threshold distance
    def add_edges_for_row(i):
        edges = []
        for j in range(i + 1, len(df)):
            euclidean_distance = distance_matrix[i, j]
            if euclidean_distance < threshold:
                name_i = df['name'].iloc[i]
                name_j = df['name'].iloc[j]
                weight = 1 / (euclidean_distance + 1e-5)  # Use inverse distance for weight
                edges.append((name_i, name_j, weight))
        return edges

    # Parallel processing using ThreadPoolExecutor
    all_edges = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(add_edges_for_row, i): i for i in range(len(df))}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing athletes"):
            all_edges.extend(future.result())

    G.add_weighted_edges_from(all_edges)
    return G

s7_birmingham = load_one_file("assets/hyroxData/S7 2024 Birmingham.csv")
bmham_filtered = get_division_entry(s7_birmingham, "male", "open")
bmham_filtered = calculate_performance_ratios(bmham_filtered)
network = build_weighted_athlete_network_with_features(bmham_filtered, 5)

# Position nodes using the spring layout for a more spread-out look
d = nx.degree(network)
pos = nx.spring_layout(network, seed=42)
degree_values = [v for k, v in d]
print('calculating communities')
communities = list(nx.community.louvain_communities(network))
print('finished calculating communities')
breakhere = 0

# Draw nodes and edges
# plt.figure(figsize=(16, 16))
# nx.draw_networkx(network, pos=nx.spring_layout(network, k=0.99), nodelist=network.nodes(), node_size=[v * 10 for v in degree_values],
#                      with_labels=True,
#                      node_color='lightgreen', alpha=0.6)
# # Display the visualization
# plt.title("Athlete Network Based on Performance Similarity")
# plt.show()
