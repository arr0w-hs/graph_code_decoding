import os
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
import json
import csv
import ast
from collections import defaultdict


fs = 15
base_dir = Path(__file__).resolve().parent
dr = base_dir/"ms_rt_data"

# date_list = [""]
time_list = ["200805", '200403', ]

graph_list = []
df_list = []
mol_list = []
out_list = []

b = 'crazy'
b = 'tri'

out_dict = defaultdict(list)
# dir_name = os.path.join(dr, a)
for file in os.listdir(dr):
    if file.endswith(".csv"):

        # if "200805" not in file and '200403' not in file:
        #     continue

        if "_meta" in file or "_map" in file:
            continue
        if "_loss_pattern" not in file:
            continue

        if b not in file:
            continue

        n = int(file[-5:-4])

        if n == 1:
            continue
        # print(n)
        print(file)
        with open(os.path.join(dr, file), "r") as f:
            data = list(csv.reader(f))

        lost_qubit_list = [[int(ele) for ele in row[1:] if ele.strip()] for row in data[1:]]
        out_dict[n].append(lost_qubit_list)

output_path = base_dir / f"ms_tt_{b}.csv"
with output_path.open("w", newline="", encoding="utf-8") as f:
            # print('writing')
            writer = csv.writer(f)
            writer.writerow(out_dict.keys())
            writer.writerows(zip(*out_dict.values()))
