import pandas as pd
import glob
from os import listdir
from pathlib import Path

barcelona = pd.read_csv("hyrox_analysis/hyroxData/S5 2023 Barcelona.csv")
path2csv = Path("hyrox_analysis/hyroxData")
csvlist = path2csv.glob("*.csv")
csvs = [pd.read_csv(g) for g in csvlist]
allHyroxData = pd.DataFrame()

for file_name in glob.glob(directoryPath+"*.csv"):
    x = pd.read_csv(file_name, low_memory=False)
    allHyroxData = pd.concat([allHyroxData, x],axis=0)

print(len(allHyroxData))