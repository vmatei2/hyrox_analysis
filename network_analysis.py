import pandas as pd
import os
from hyrox_results_analysis import load_one_file, extract_mean_values_runs_stations, get_division_entry, \
    plot_data_points, line_plot_runs, get_filtered_df, load_all_races
import constants as _ct
import networkx as nx
import seaborn as sns
import numpy as np
from scipy.spatial.distance import euclidean, pdist, squareform
import matplotlib.pyplot as plt
import pickle
import random
from network_helpers import nx2gt
import graph_tool.all as gt
from matplotlib import cm

sns.set_style('darkgrid')

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
    df[_ct.SLEDPULL_2_BURPEE] = df['work_3'] / df['work_4']
    df[_ct.SKI_ERG_TO_ROW_ERG] = df['work_1'] / df['work_5']
    df[_ct.SKI_ERG_TO_WALL_BALL] = df['work_1'] / df['work_8']
    #  calculate the average percentage change between runs for an athlete
    run_columns = [f'run_{i}' for i in range(1, 9)]
    run_changes = df[run_columns].pct_change(axis=1).abs()
    df['avg_run_pacing_change'] = run_changes.mean(axis=1)

    #  we will look at an athlete's performance on the stations in strength vs endurance - i.e. sleds / farmers carry  vs endurance lunges / burpees / wallballs / skierg / rowerg
    work_columns = [f'work_{i}' for i in range(1, 9)]
    for col in work_columns:
        df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
    run_columns = [f'run_{i}' for i in range(1, 9)]
    for col in run_columns:
        df[f'{col}_zscore'] = (df[col] - df[col].mean()) / df[col].std()
    strength_stations = ['work_2', 'work_3', 'work_6', 'work_8']
    endurance_stations = ['work_1', 'work_4', 'work_5', 'work_7']
    df['strength_score'] = df[[f'{station}_zscore' for station in strength_stations]].sum(axis=1)
    df['endurance_score'] = df[[f'{station}_zscore' for station in endurance_stations]].sum(axis=1)

    # Calculate Strength-to-Endurance Balance
    df['strength_to_endurance_balance'] = df['strength_score'] / (
            df['endurance_score'] + 1e-9)  # Avoid division by zero

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
    metrics = df[_ct.NETWORK_ANALYSIS_METRICS].values

    # Calculate all pairwise Euclidean distances
    distances = pdist(metrics, metric='euclidean')
    adjacency_matrix = squareform(distances)

    # Initialize the graph and add nodes
    G = nx.Graph()
    for i, name_i in enumerate(df['name']):
        for j, name_j in enumerate(df['name']):
            if i < j:
                weight = adjacency_matrix[i, j]  # extract the similarity value between the two athletes
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


def plot_communities_large(network, communities, node_sample_size=2201, edge_alpha=0.5, seed=42):
    """
    Visualizes the network with nodes colored by their community membership, optimized for large graphs.

    Parameters:
        network (nx.Graph): The input graph.
        communities (list of sets): List of sets where each set contains nodes in a community.
        node_sample_size (int): Number of nodes to sample for visualization. Default is 500.
        edge_alpha (float): Transparency level for edges. Default is 0.1.
        seed (int): Random seed for layout reproducibility. Default is 42.

    Returns:
        None
    """
    # If sampling, reduce the graph to a subset of nodes
    if len(network.nodes) > node_sample_size:
        sampled_nodes = set(random.sample(list(network.nodes), node_sample_size)) # Take a random set of `node_sample_size` nodes
        network = network.subgraph(sampled_nodes).copy()
        #  convert the networkx graph object to graph-tool object for enhanced plotting
        #  filter the communities as well after sampling nodes!
        updated_communities = [
            {node for node in community if node in sampled_nodes} for community in communities
        ]
        #  remove any empty communities after sampling
        updated_communities = [community for community in updated_communities if len(community)>0]
    else:
        #if not sampling, then simply assign updated communities to be used afterwards
        updated_communities = communities
    gt_graph = nx2gt(network)
    # creating a property map for community-based coloring
    community_property = gt_graph.new_vertex_property("int")
    label_to_vertex = {(gt_graph.vp['name'][v]): v for v in gt_graph.vertices()}


    for community_id, community in enumerate(updated_communities):
        for node in community:
            community_property[label_to_vertex[node]] = community_id

    # Convert community IDs to colors (e.g., RGBA values)
    num_communities = len(set(community_property.a))  # Number of unique communities
    colormap = cm.get_cmap("tab10", num_communities)  # Use a colormap with enough colors

    color_property = gt_graph.new_vertex_property("vector<float>")
    for v in gt_graph.vertices():
        community_id = community_property[v]
        color_property[v] = list(colormap(community_id)[:3])  # Extract RGB from colormap

    #  generate a layout for the graph
    pos = gt.sfdp_layout(gt_graph)

    # create a vertex property map for node sizes
    degree_property = gt_graph.new_vertex_property("float")

    # assign sizes proportional to the degree of each vertex
    for v in gt_graph.vertices():
        degree_property[v] = gt_graph.vertex(v).out_degree() * 1.3  # scale factor

    gt.graph_draw(gt_graph,
                  pos=pos,
                  vertex_fill_color=color_property,
                  C=3.0,
                  output_size=(800, 800)
                  )



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
        for column in _ct.NETWORK_ANALYSIS_METRICS:
            mean_value = community_df[column].mean()
            std_dev_value = community_df[column].std()
            max_value = community_df[column].max()
            min_value = community_df[column].min()
            print(
                f"{column}: Mean = {mean_value:.2f}, Std dev = {std_dev_value:.2f} Max = {max_value:.2f} Min = {min_value:.2f}")
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


def plot_communities_insights(communities):
    num_communities = 0

    # collect data for plotting
    lap_avg_run_times = []
    avg_stations_times = []
    avg_total_times = []
    individual_run_times = []
    individual_station_times = []
    individual_total_times = []
    number_of_athletes_in_community = []
    metrics_data = []

    for i, community_df in enumerate(communities):

            lap_avg_run_time = community_df[[f'run_{i + 1}' for i in range(8)]].median()
            station_avg_times = community_df[[f'work_{i + 1}' for i in range(8)]].median()
            avg_total_time = community_df['total_time'].median()

            if len(community_df) > 1:
                num_communities += 1
                lap_avg_run_times.append(lap_avg_run_time)
                avg_stations_times.append(station_avg_times)
                avg_total_times.append(avg_total_time)
                metrics_avg = community_df[_ct.NETWORK_ANALYSIS_METRICS].median().values
                metrics_data.append(metrics_avg)
                #  getting the fastest athlete in the community to check their performance values
                community_athlete = community_df.iloc[0]
                individual_run_times.append(community_athlete[[f'run_{i+1}' for i in range(8)]])
                individual_station_times.append(community_athlete[[f'work_{i+1}' for i in range(8)]])
                individual_total_times.append(community_athlete['total_time'])
                number_of_athletes_in_community.append(len(community_df))

    num_communities = len(lap_avg_run_times)
    # plt.subplots (nrows, ncols)
    fig, axes = plt.subplots(2, 3, figsize=(12, 12))

    # Plot Median run and station times for each community
    for idx in range(num_communities):
        # Top row: Community Median run times
        axes[0, 0].plot(range(1, 9), lap_avg_run_times[idx], marker='o', label=f'Community {idx + 1}')
        axes[0, 0].set_title('Median Run Time Per Lap')
        axes[0, 0].set_xlabel('Lap Number')
        axes[0, 0].set_ylabel('Median Time (s)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        # Top row: Community Median station times
        axes[0, 1].plot(range(1, 9), avg_stations_times[idx], marker='s', label=f'Community {idx + 1}')
        axes[0, 1].set_title('Median Station Time Per Station')
        axes[0, 1].set_xlabel('Station Number')
        axes[0, 1].set_ylabel('Median Time (s)')
        axes[0, 1].legend()
        axes[0, 1].grid(True)

        # Top row: Community median total finish times
        bars = axes[0, 2].bar(idx + 1, avg_total_times[idx], color='skyblue')
        axes[0, 2].set_title('Median Total Finish Time')
        axes[0, 2].set_xlabel('Community')
        axes[0, 2].set_ylabel('Median Total Time (s)')
        axes[0, 2].set_xticks(range(1, num_communities + 1))
        axes[0, 2].grid(axis='y', linestyle='--', alpha=0.7)
        axes[0, 2].bar_label(bars, fmt='%.2f', padding=2)

    # Plot individual athlete's run times
    # for idx in range(num_communities):
    #     axes[1, 0].plot(range(1, 9), individual_run_times[idx], marker='o', label=f'Athlete {idx + 1}')
    # axes[1, 0].set_title('Individual Athlete Run Times')
    # axes[1, 0].set_xlabel('Lap Number')
    # axes[1, 0].set_ylabel('Time (s)')
    # axes[1, 0].legend()
    # axes[1, 0].grid(True)

    # # Plot individual athlete's station times
    # for idx in range(num_communities):
    #     axes[1, 1].plot(range(1, 9), individual_station_times[idx], marker='s', label=f'Athlete {idx + 1}')
    # axes[1, 1].set_title('Individual Athlete Station Times')
    # axes[1, 1].set_xlabel('Station Number')
    # axes[1, 1].set_ylabel('Time (s)')
    # axes[1, 1].legend()
    # axes[1, 1].grid(True)
    #
    # # Plot individual athlete's total finish times
    # bars = axes[1, 2].bar(range(1, num_communities + 1), individual_total_times, color='lightcoral')
    # axes[1, 2].set_title('Individual Athlete Total Finish Time')
    # axes[1, 2].set_xlabel('Athlete')
    # axes[1, 2].set_ylabel('Total Time (s)')
    # axes[1, 2].set_xticks(range(1, num_communities + 1))
    # axes[1, 2].grid(axis='y', linestyle='--', alpha=0.7)
    # axes[1, 2].bar_label(bars, fmt='%.2f', padding=2)

    # Plot number of athletes in each community
    size_bars = axes[1, 0].bar(range(1, num_communities + 1), number_of_athletes_in_community, color='lightgreen')
    axes[1, 0].set_title('Number of Athletes in Each Community')
    axes[1, 0].set_xlabel('Community')
    axes[1, 0].set_ylabel('Number of Athletes')
    axes[1, 0].set_xticks(range(1, num_communities + 1))
    axes[1, 0].grid(axis='y', linestyle='--', alpha=0.7)
    axes[1, 0].bar_label(size_bars, fmt='%d', padding=3)

    # Plot stacked bar chart of network metrics used in community detection
    # Plot grouped bar chart for community metrics
    bar_width = 0.15
    x = np.arange(len(_ct.NETWORK_ANALYSIS_METRICS))  # the label locations
    colors = plt.cm.tab20.colors  # Use a colormap with enough distinct colors

    for idx in range(num_communities):
        offset = bar_width * idx
        axes[1, 1].bar(x + offset, metrics_data[idx], bar_width, label=f'Community {idx + 1}', color=colors[idx % len(colors)])

    axes[1, 1].set_title('Community Metrics')
    axes[1, 1].set_xlabel('Metrics')
    axes[1, 1].set_ylabel('Median Value')
    axes[1, 1].set_xticks(x + bar_width * (num_communities - 1) / 2)
    axes[1, 1].set_xticklabels(_ct.NETWORK_ANALYSIS_METRICS, rotation=45, ha='right')
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    axes[1, 0].axis('off')
    axes[1, 2].axis('off')
    # Adjust layout
    plt.tight_layout()
    plt.savefig("community_insights.jpeg")
    plt.show()

def generate_fastest_median_report(community_dfs, file_path="assets/reports/perfrormance_report.csv"):
    """
    Function to generate a report on the median values of each community, and the deviation from the fastest community
    :param community_dfs: list of dataframes for each community
    :param file_path: path to save the file
    :return:
    """
    community_names = [f"Community {i+1} " for i in range(len(community_dfs))]
    results = {}
    segment_labels = _ct.RUN_LABELS + _ct.WORK_LABELS
    for segment_label in segment_labels:
        #  Compute median times for each community for the current segment
        medians = {community: df[segment_label].median() for community, df in zip (community_names, community_dfs)}
        fastest_time = min(medians.values())

        row_data = {community: ((median_time - fastest_time) / fastest_time) * 100 for community, median_time in medians.items()}
        results[segment_label] = row_data
    #  convert results to a Dataframe
    results_df = pd.DataFrame(results).T
    results_df.index.name ='Segment'
    results_df.to_csv(file_path)
    return results_df




def main_network_analysis(hyrox_list_file_path):
    dublin = load_one_file("assets/hyroxData/S7 2024 Manchester.csv")
    # dublin = load_all_races()
    dublin = get_division_entry(dublin, "male", "open")
    dublin = calculate_performance_ratios(dublin)
    dublin = get_filtered_df(dublin, "total_time", 65)
    network = build_athlete_network(dublin, 0.18)
    print('calculating communities')
    communities = list(nx.community.louvain_communities(network, weight='weight', seed=20, resolution=0.6))
    community_dfs = extract_community_dataframes(dublin, communities)
    profile_communities(community_dfs)
    with open(hyrox_list_file_path, 'wb') as file:
        pickle.dump(community_dfs, file)
    report_df = generate_fastest_median_report(community_dfs)
    print("plotting community insights")
    plot_communities_insights(community_dfs)
    print("plotting the communities")
    plot_communities_large(network, communities)
    print('finished calculating communities')


file_path = 'assets/hyroxData/community_dfs.pkl'
main_network_analysis(file_path)

