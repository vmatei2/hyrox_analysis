import pandas as pd

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
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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
    df[_ct.RUN_2_TOTAL] = df['run_time'] / df['total_time']
    #  2. Roxzone Time to Total Time ratio
    df['roxzone_to_total_ratio'] = df['roxzone_time'] / df['total_time']
    #  3. Run 1 to Run 8 -- how much can people hold their pace?
    df['run1_to_run8_ratio'] = df['run_1'] / df['run_8']
    df['sled_push_to_sled_pull_ratio'] = df['work_2'] / df['work_3']
    #  4. First half to second half ratio (run and work times)
    first_half_time = df[['run_1', 'run_2', 'run_3', 'run_4', 'work_1', 'work_2', 'work_3', 'work_4']].sum(axis=1)
    second_half_time = df[['run_5', 'run_6', 'run_7', 'run_8', 'work_5', 'work_6', 'work_7', 'work_8']].sum(axis=1)
    df['first_half_to_second_half_ratio'] = first_half_time / second_half_time

    #  calculate the average percentage change between runs for an athlete
    run_columns = [f'run_{i}' for i in range(1, 9)]
    run_changes = df[run_columns].pct_change(axis=1).abs()
    df['avg_run_pacing_change'] = run_changes.mean(axis=1)

    #  we will look at an athlete's performance on the stations in strength vs endurance - i.e. sleds / farmers carry  vs endurance lunges / burpees / wallballs / skierg / rowerg
    work_columns = [f'work_{i}' for i in range(1, 9)]
    for col in work_columns:
        df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
    strength_stations = ['work_2', 'work_3', 'work_6']
    endurance_stations = ['work_1', 'work_4', 'work_5', 'work_6', 'work_7', 'work_8']
    df['strength_score'] = df[[f'{station}_zscore' for station in strength_stations]].sum(axis=1)
    df['endurance_score'] = df[[f'{station}_zscore' for station in endurance_stations]].sum(axis=1)


    # Calculate Strength-to-Endurance Balance
    df['strength_to_endurance_balance'] = df['strength_score'] / (df['endurance_score'] + 1e-9)  # Avoid division by zero

    # infs may come from data issues (i.e. run stored as 0)
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    # Replace inf/-inf with NaN only in numeric columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Drop rows with NaN in any numeric column
    df.dropna(subset=numeric_cols, inplace=True)
    return df


def build_athlete_network(df, similarity_threshold):
    """
    Builds a network of athletes based on similarity in performance metrics, including engineered features.

    :param df: DataFrame containing athlete performance data.
    :param threshold: Minimum similarity score for creating an edge between two athletes.
    :return: A NetworkX graph with athletes as nodes and similarity-based edges.
    """

    # Prepare data
    metrics = df[[_ct.WORK_2_RUN, _ct.ROXZONE_2_TOTAL, _ct.RUN_1_TO_8, _ct.SLED_PUSH_2_PULL, _ct.FIRST_HALF_TO_SECOND_HALF_RATIO, _ct.AVG_RUN_PACING_CHANGE,
                  _ct.STRENGTH_TO_ENDURANCE_BALANCE, _ct.RUN_2_TOTAL]].values

    # Calculate all pairwise Euclidean distances
    distances = pdist(metrics, metric='euclidean')
    #  Step 1. Normalised the distances between 0 and 1 - Step 2. Subtract 1, so a higher value will mean a higher similarity score (i.e. 0.1 distance - 1-0.1 = 0.9 --> high similarity)
    adjacency_matrix = squareform(distances)
    #  similarity_weights = 1 - (distances - distances.min()) / (distances.max() - distances.min())  #  normalise the values
    # adjacency_matrix = squareform(similarity_weights)  # Convert to a square matrix format

    # Initialize the graph and add nodes
    G = nx.Graph()
    for i, name_i in enumerate(df['name']):
        for j, name_j in enumerate(df['name']):
            if i < j:  # Avoid duplicates and self-loops
                weight = adjacency_matrix[i, j]  # extrac the similarity value between the two athletes
                if weight < similarity_threshold:
                    G.add_edge(name_i, name_j, weight=adjacency_matrix[i, j])
    return G

def extract_community_dataframes(df, communities):
    """
    Retrusn a list of dataframes, each corresponding to a community
    :param df: the datframe of athletes
    :param commuities: communities generated using network analysis
    :return:
    """
    return [df[df['name'].isin(community)].copy() for community in communities]

def profile_communities(community_dfs):
    """
    Function for profiling the communities returned by the algorithms
    :param community_dfs:
    :return:
    """

    profiling_data = []

    for i, community_df in enumerate(community_dfs):
        community_name = f"Community {i + 1}"
        num_athletes = len(community_df)
        fastest_time = community_df['total_time'].min()
        slowest_time = community_df['total_time'].max()
        for column in [_ct.WORK_2_RUN, _ct.ROXZONE_2_TOTAL, _ct.RUN_1_TO_8, _ct.SLED_PUSH_2_PULL, _ct.FIRST_HALF_TO_SECOND_HALF_RATIO, _ct.AVG_RUN_PACING_CHANGE,
                       _ct.STRENGTH_TO_ENDURANCE_BALANCE]:
            mean_value = community_df[column].mean()
            std_dev_value = community_df[column].std()
            max_value = community_df[column].max()
            min_value = community_df[column].min()
            print(f"{column}: Mean = {mean_value:.2f}, Std dev = {std_dev_value:.2f} Max = {max_value:.2f} Min = {min_value:.2f}")
            profiling_data.append(({
                "Community": community_name,
                "Number of Athletes": num_athletes,
                "Fastest Time": fastest_time,
                "Slowest Time": slowest_time,
                "Metric": column,
                "Mean": mean_value,
                "Max": max_value,
                "Min": min_value
            }))

    profiling_df = pd.DataFrame(profiling_data)
    profiling_df.to_csv("assets/reports/report.csv")

def plot_communities(communities):
    num_communities = 0

    # collect data for plotting
    lap_avg_run_times = []
    avg_stations_times = []
    avg_total_times = []

    for i, community_df in enumerate(communities):
        if len(community_df) > 10:
            num_communities += 1
            lap_avg_run_time = community_df[[f'run_{i+1}' for i in range(8)]].mean()
            station_avg_times = community_df[[f'work_{i+1}' for i in range(8)]].mean()
            avg_total_time = community_df['total_time'].mean()

            lap_avg_run_times.append(lap_avg_run_time)
            avg_stations_times.append(station_avg_times)
            avg_total_times.append(avg_total_time)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Plot average lap run times
    for idx, avg_times in enumerate(lap_avg_run_times):
        axes[0].plot(range(1, 9), avg_times, marker='o', label=f'Community {idx+1}')
    axes[0].set_title('Average Run Time Per Lap')
    axes[0].set_xlabel('Lap Number')
    axes[0].set_ylabel('Average Time (s)')
    axes[0].legend()
    axes[0].grid(True)

    # Plot average station times
    for idx, avg_times in enumerate(avg_stations_times):
        axes[1].plot(range(1, 9), avg_times, marker='s', label=f'Community {idx+1}')
    axes[1].set_title('Average Station Time Per Station')
    axes[1].set_xlabel('Station Number')
    axes[1].set_ylabel('Average Time (s)')
    axes[1].legend()
    axes[1].grid(True)

    # Plot average total finish times
    axes[2].bar(range(1, num_communities + 1), avg_total_times, color='skyblue')
    axes[2].set_title('Average Total Finish Time')
    axes[2].set_xlabel('Community')
    axes[2].set_ylabel('Average Total Time (s)')
    axes[2].set_xticks(range(1, num_communities + 1))
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout
    plt.tight_layout()
    plt.show()


dublin = load_one_file("assets/hyroxData/S7 2024 Dublin.csv")
dublin = get_division_entry(dublin, "male", "open")
dublin = calculate_performance_ratios(dublin)
network = build_athlete_network(dublin, 0.15)
print('calculating communities')
communities = list(nx.community.louvain_communities(network, weight='weight', resolution=0.4))
community_dfs = extract_community_dataframes(dublin, communities)
profile_communities(community_dfs)

plot_communities(community_dfs)
print('finished calculating communities')
