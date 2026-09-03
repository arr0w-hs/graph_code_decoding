import sys
import os
# import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm


plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dr = os.path.dirname(__file__)

def roundoff(ele):
    """function to round off float"""
    ele = int(ele*100)/100
    if ele == 0.44:
        ele = 0.45
    return ele

def mer_edge(ele):
    """function to round off float"""
    g = nx.Graph()
    g.add_edges_from(ele)
    return g.number_of_edges()


form = "pdf"

fs = 15
# data_dir = os.path.join(dir_name, "er_results/2026-01-26_er/")
# data_dir = os.path.join(dir_name, "er_results/2026-03-27_er/")
# # data_dir = os.path.join(dir_name, "er_results/2025-12-02_er/")
# dir_name = os.path.join(dir_name, "er_results/")

dr = os.path.join(dr, "threshold_folder")

a = "minimise"
a = "no_min"

#  filees = ["114507_sc_thres_2000.csv"]
# files = ["153813_sc_thres_2000_nomin_True.csv", "001045_sc_thres_2000_nomin_True.csv",
#          "013821_sc_thres_2000_nomin_True.csv"]
# files =["153637_sc_thres_2000_nomin.csv", "153730_sc_thres_2000_nomin.csv",
#         "153720_sc_thres_2000_nomin.csv"]



# for ele in files:
#     out_list = []
#     out_df = pd.read_csv(os.path.join(dir_name, ele))
#     print(out_df)


# dates = ["2026-08-05",
#         "2026-08-06",
#         "2026-08-08",
#         "2026-08-09"]

# dates = ["2026-08-31"]
dates = ["2026-09-01", "2026-09-02"]
out_list = []

no_targets = [
    "215418_rsc_thres_min_True_1_None",
    "214038_rsc_thres_min_True_1_None",
    "165834_rsc_thres_min_True_1_None",
    "153013_rsc_thres_min_True_1_None",
    "153012_rsc_thres_min_True_1_None",
]

targets = ["104459_rsc_thres_min_True_1_17", "221719_rsc_thres_min_True_1_17"]

for a in dates:
    dir_name = os.path.join(dr, a)

    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:

            if not any(target in file for target in no_targets):
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            out_list.append(df)


out_df = pd.concat(out_list).reset_index(drop=True)
out_df = out_df.drop(columns=["lost_qubits",
                              'Heuristic X logical', 'Heuristic Z logical',
                              'g-SPF X logical', 'g-SPF Z logical',])
out_list = []
for a in dates:
    dir_name = os.path.join(dr, a)

    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:

            if not any(target in file for target in targets):
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            out_list.append(df)

target_df = pd.concat(out_list).reset_index(drop=True)
target_df = target_df.drop(columns=["lost_qubits",
                              'Heuristic X logical', 'Heuristic Z logical',
                              'g-SPF X logical', 'g-SPF Z logical',])

methods = ["Heuristic", "g-SPF"]

out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
out_df = out_df[out_df["Loss probability"] <= 1]
out_df["Distance"] = pd.to_numeric(out_df["Distance"])
distances = sorted(out_df["Distance"].unique())


target_df["Loss probability"] = pd.to_numeric(target_df["Loss probability"], errors="coerce")
target_df = target_df[target_df["Loss probability"] <= 1]
target_df["Distance"] = pd.to_numeric(target_df["Distance"])
distances = sorted(target_df["Distance"].unique())


fig, ax = plt.subplots(figsize=(7, 5))
linestyles = ["-", "--"]
colour = ["#AF4189", "#4171B0", "#4DB041", "#B08A41", "#5B4052"]
colour = ["#CC332D", "#61A6E9", "#FFDDCC", "#DDDFB0", "#276D60"]
markers = [".", "v", "^", "D", "s", "P", "X", "*"]

for i, method in enumerate(methods):
    for dist, distance_df in out_df.groupby("Distance"):
        success_col = method + " success"

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        distance_df = (distance_df.groupby("Loss probability",
                                        as_index=False).agg(success_mean=(success_col, "mean"),
                                            sample_count=(success_col, "count"),).sort_values("Loss probability"))
        distance_df[method + " yerr"] = np.sqrt( distance_df["success_mean"] *
                                            (1 - distance_df["success_mean"])
                                            / distance_df["sample_count"])

        ax.plot(
            distance_df["Loss probability"],
            distance_df["success_mean"],
            linestyle=linestyles[i],
            marker=markers[i],
            label="No fixed target qubit",
            color = colour[0]
        )

    for dist, distance_df in target_df.groupby("Distance"):
        success_col = method + " success"

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        distance_df = (distance_df.groupby("Loss probability",
                                        as_index=False).agg(success_mean=(success_col, "mean"),
                                            sample_count=(success_col, "count"),).sort_values("Loss probability"))
        distance_df[method + " yerr"] = np.sqrt( distance_df["success_mean"] *
                                            (1 - distance_df["success_mean"])
                                            / distance_df["sample_count"])

        ax.plot(
            distance_df["Loss probability"],
            distance_df["success_mean"],
            linestyle=linestyles[i],
            marker=markers[i],
            label="Fixed target qubit",
            color = colour[1]
        )
a = 2
# line style
target_handles = [
    Line2D(
        [0], [0],
        color = colour[0],
        linestyle=linestyles[0],
        linewidth=a,
        markersize=6,
        label="No fixed target qubit",
    ),
    Line2D(
        [0], [0],
        color = colour[1],
        linestyle=linestyles[0],
        linewidth=a,
        markersize=6,
        label="Fixed target qubit",
    ),
]
# Method entries: marker and colours
method_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=markers[i],
        linestyle=linestyles[i],
        linewidth=a,
        markersize=6,
        label=str(method),
    )
    for i, method in enumerate(methods)
]

combined_handles = target_handles + method_handles
target_heading = Line2D(
    [], [], linestyle="None", marker=None, label="Target")

method_heading = Line2D(
    [], [], linestyle="None", marker=None, label="Methods")

combined_handles = ([target_heading]+ target_handles
                    + [method_heading]+ method_handles)

legend = ax.legend(
    handles=combined_handles,
    loc="upper right",
    borderaxespad=0,
    fontsize=13,
    handlelength=2,
    labelspacing=0.5,
        bbox_to_anchor=(0.98, 0.98),
    frameon=True,
)
legend_texts = legend.get_texts()
legend_texts[0].set_weight("bold")
legend_texts[2 + 1].set_weight("bold")

ax.set_title(f"Distance-7 rotated surface code", fontsize=fs)
ax.set_xlabel("Loss probability", fontsize=fs)
ax.set_ylabel("Teleportation rate", fontsize=fs)
ax.grid(True, which="both", alpha=0.3)
ax.tick_params(axis="both", labelsize=fs)
fig.tight_layout()

fig.savefig(os.path.join(dr, f"Target_{method}.pdf"), dpi=800, bbox_inches="tight")
plt.show()
