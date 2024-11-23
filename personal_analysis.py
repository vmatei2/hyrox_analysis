from hyrox_results_analysis import load_one_file
from constants import *
import seaborn as sns

sns.set_style('darkgrid')
import matplotlib.pyplot as plt
import pandas as pd


def compare_races(path_df_1, path_df_2, race_1_name, race_2_name, name_1, name_2=""):

    if name_2 == "":  # if not passing in empty name, assume it's the same person
        name_2 = name_1
    df_1 = load_one_file(path_df_1)
    df_2 = load_one_file(path_df_2)

    df_1_user = df_1[df_1['name'].str.contains(name_1, case=False, na=False)]
    df_2_user = df_2[df_2['name'].str.contains(name_2, case=False, na=False)]

    columns_to_extract = RACE_ORDER_LABELS + [ROXZONE_TIME] # ROXZONE_TIME is a string variable hence moving to list for concatenation

    race_1_times = df_1_user[columns_to_extract]
    race_2_times = df_2_user[columns_to_extract]

    race_1_values = race_1_times.iloc[0].values  # extracting the values from a single row dataframe
    race_2_values = race_2_times.iloc[0].values

    differences = race_2_values - race_1_values
    x = range(len(columns_to_extract))

    width = 0.4  # width of each bar in the plot
    fig, ax = plt.subplots(figsize=(10, 6))


    race_1_bars = ax.bar([i - width/2 for i in x], race_1_values, width=width, label=race_1_name)  # assigning ax.bar variables in case needed for further dynamic manipulation of the data
    race_2_Bars = ax.bar([i + width/2 for i in x], race_2_values, width=width, label=race_2_name)  # i+width here so we slightly separate the bars visually

    for i, (race1, race2, diffs) in enumerate(zip(race_1_values, race_2_values, differences)):
        # Calculate the x position for annotation (between the bars)
        x_pos = i
        y_pos = max(race1, race2) + 0.1  # Position the text slightly above the higher bar
        ax.text(x_pos, y_pos, f'{diffs:.2f}', ha='center', fontsize=10, color='black')

    ax.set_xticks(x)
    ax.set_xticklabels(ALL_USER_INPUTS + ['ROXZONE'], rotation=45, ha='right')
    ax.set_ylabel('Time (in minutes) i.e 4.5 = 4min30s')
    if name_2 != name_1:
        name_comparison = f'{name_2} and {name_1}'
    else:
        name_comparison = name_1
    ax.set_title(f'Comparison of Times between {race_2_name} and {race_1_name} for {name_comparison}')
    ax.legend()

    plt.tight_layout()
    plt.show()


compare_races(path_df_1="assets/hyroxData/S6 2024 Rotterdam.csv", path_df_2="assets/hyroxData/S7 2024 Dublin.csv",
              race_1_name="Rotterdam April 2024" , race_2_name="Dublin November 2024", name_1='Matei')

compare_races(path_df_1="assets/hyroxData/S6 2023 London.csv", path_df_2="assets/hyroxData/S7 2024 Birmingham.csv",
              race_1_name="London Excel 2023", race_2_name="Birmingham 2024", name_1="Ingham, James")

compare_races(path_df_1="assets/hyroxData/S6 2023 London.csv", path_df_2="assets/hyroxData/S7 2024 Dublin.csv", race_1_name="London Excel 2023",
              race_2_name="Dublin 2024", name_1="Ingham, James", name_2="Matei, Vlad")
