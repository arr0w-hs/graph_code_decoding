import ast
import sys
import os
# import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import csv
from collections import defaultdict
from pathlib import Path
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


def weighted_mean(group):

    return (
        (group[success_col] * group["config_count"]).sum()
        / group["config_count"].sum()
    )

form = "pdf"
save = True
save = False

fs = 15
dr = os.path.join(dr, "threshold_folder")

a = "minimise"
a = "no_min"
# dates = ["2026-09-01", "2026-09-02"]
dates = ["2026-09-12"]
out_list = []
for a in dates:
    dir_name = os.path.join(dr, a)
    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            if "5" in file:
                df["Distance"] = int(2)
            elif "10" in file:
                df["Distance"] = int(3)
            elif "17" in file:
                df["Distance"] = int(4)
            out_list.append(df)

out_df = pd.concat(out_list).reset_index(drop=True)
out_df = out_df.drop(columns=[
                              'g-SPF X logical', 'g-SPF Z logical',])
                            #   'ILP X logical', 'ILP Z logical'])

# print(out_df)
# methods = ["Heuristic", "ILP", "g-SPF"]
# methods = ["Heuristic", "g-SPF"]
methods = ["g-SPF"]
# plots = [" support size", " runtime", " g", " verify"]
plots = [" support size", " runtime"]


linestyles = ["-", "-", ":", "dashdot"]
colour = ["#AF4189", "#4171B0", "#4DB041", "#B08A41", "#5B4052"]
colour = ["#CC332D", "#61A6E9","#8B7970", "#276D60", "#DDDFB0", "#FFDDCC", "#594D47"]
markers = ["v", "o", "^", "D", "s", "P", "X", "*"]

# out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
# out_df = out_df[out_df["Loss probability"] <= 1]


####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
#                     Runtime fixed p
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
fig, ax = plt.subplots(figsize=(7, 5))
method = 'g-SPF'
i=0
success_col = method + " runtime"
# distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
out_df[success_col] = pd.to_numeric(out_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
out_df = (
    out_df.groupby("Distance", as_index=False)
    .agg(
        success_mean=(success_col, "mean"),
    )
    .sort_values("Distance")
)

ax.plot(
    out_df["Distance"],
    out_df["success_mean"],
    linestyle=linestyles[i],
    marker=markers[i],
    label=f"{method}",
    color = colour[i]
)

base_dir = Path(__file__).resolve().parent
dr = base_dir/"ms_rt_data"
time_list = ["200805", '200403', ]
b = 'crazy'
# b = 'tri'
out_data = []
out_dict = defaultdict(list)
for file in os.listdir(dr):
    if file.endswith(".csv"):

        if "200805" not in file and '200403' not in file:
            continue

        if "_meta" in file or "_map" in file:
            continue
        if "_build" not in file:
            continue

        if b not in file:
            continue
        df = pd.read_csv(os.path.join(dr, file), on_bad_lines="skip", engine="python")
        out_data.append(df)


out_df = pd.concat(out_data).reset_index(drop=True)
out_df = out_df.drop(columns=['stabiliser_build_timed_out',"xz_reps_requested",
                            'xz_reps_done',
                            "n_xz_timeouts","XZ_list_creation_time_std"])
out_df = out_df.drop(index=1)
out_df = out_df.rename(columns={"n": "Distance"})
out_df = out_df.sort_values(by="Distance").reset_index(drop=True)
out_df["MS runtime"] = out_df["stabiliser_build_t"] + out_df["XZ_list_creation_time_av"]


ax.plot(
    out_df["Distance"],
    out_df["stabiliser_build_t"],
    linestyle=linestyles[i],
    marker=markers[1],
    label=f"Morley-Short et al. Build time",
    color = colour[1]
)


# ax.plot(
#     out_df["Distance"],
#     out_df["XZ_list_creation_time_av"],
#     linestyle=linestyles[3],
#     marker=markers[2],
#     label=f"Morley-Short et al. X,Z list creation time",
#     color = colour[1]
# )


b = 'crazy'
# b = 'tri'
out_data = []
out_dict = defaultdict(list)
for file in os.listdir(dr):
    if file.endswith(".csv"):

        if "200805" not in file and '200403' not in file:
            continue

        if "_meta" in file or "_map" in file:
            continue
        if "_min_pair_time" not in file:
            continue

        if b not in file:
            continue
        df = pd.read_csv(os.path.join(dr, file), on_bad_lines="skip", engine="python")
        out_data.append(df)


out_df = pd.concat(out_data).reset_index(drop=True)
# out_df = out_df.drop(columns=['stabiliser_build_timed_out',"xz_reps_requested",
#                             'xz_reps_done',
#                             "n_xz_timeouts","XZ_list_creation_time_std"])


out_df = out_df.drop(index=2)
# print(out_df)
out_df = out_df.rename(columns={"n": "Distance"})
out_df = out_df.sort_values(by="Distance").reset_index(drop=True)
out_df["MS find"] = out_df["min_pair_XZ_list_creation_time_av"] + \
    out_df["min_pair_XZ_list_creation_time_std"]


ax.plot(
    out_df["Distance"],
    out_df["MS find"],
    linestyle='--',
    marker=markers[1],
    label=f"Morley-Short et al. Minimum pair time",
    color = colour[1]
)



base_dir = Path(__file__).resolve().parent
ax.set_xticks([2, 3, 4])
ax.set_title(f"Runtime comparison", fontsize=fs)
ax.set_xlabel("Channel parameter", fontsize=fs)
# ax.set_ylabel("Success rate", fontsize=fs)

ax.set_ylabel("Average runtime (s)", fontsize=fs)
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.5)
ax.tick_params(axis="both", labelsize=fs)
ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)
fig.tight_layout()
if save:
    fig.savefig(os.path.join(base_dir, f"threshold_folder/ms_rt_comparison_crazy.pdf"), dpi=800, bbox_inches="tight")




# for i, method in enumerate(methods):
#     if i >0:
#         continue
#     fig, ax = plt.subplots(figsize=(7, 5))
#     for dist, distance_df in out_df.groupby("Distance"):
#         # success_col = method + " success"
#         success_col = method + " runtime"

#         distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
#         distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
#         # distance_df = (
#         #     distance_df.groupby("Loss probability", as_index=False)
#         #     .agg(
#         #         success_mean=(success_col, "mean"),
#         #         sample_count=(success_col, "count"),
#         #     )
#         #     .sort_values("Loss probability")
#         # )

#         distance_df = distance_df.groupby("Loss probability", as_index=False).sum()


#         ax.plot(
#             distance_df["Loss probability"],
#             distance_df["config_count"],
#             label=f"{dist}",
#         )

#     ax.grid(True, which="both", alpha=0.5)
#     ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)
plt.show()
