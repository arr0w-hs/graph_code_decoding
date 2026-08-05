import sys
import os
# import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D


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
a = "2026-08-03"
dir_name = os.path.join(dr, a)

#  filees = ["114507_sc_thres_2000.csv"]
# files = ["153813_sc_thres_2000_nomin_True.csv", "001045_sc_thres_2000_nomin_True.csv",
#          "013821_sc_thres_2000_nomin_True.csv"]
# files =["153637_sc_thres_2000_nomin.csv", "153730_sc_thres_2000_nomin.csv",
#         "153720_sc_thres_2000_nomin.csv"]



# for ele in files:
#     out_list = []
#     out_df = pd.read_csv(os.path.join(dir_name, ele))
#     print(out_df)


out_list = []
for file in os.listdir(dir_name):
    if file.endswith(".csv"):
        df = pd.read_csv(os.path.join(dir_name, file))
        out_list.append(df)


out_df1 = pd.concat(ele for ele in out_list)
out_df = (
    out_df1.groupby(["method", "loss_prob", "distance"], as_index=False)
    [["runtime", "support_size", "teleportation_rate"]]
    .mean()
)

fig, ax = plt.subplots(figsize=(8.5, 5))
# ax = fig.add_axes([0.10, 0.15, 0.58, 0.75])
distances = sorted(out_df["distance"].unique())
methods = sorted(out_df["method"].unique())

color_map = {
    distance: plt.cm.tab10(i % 10)
    for i, distance in enumerate(distances)}

marker_list = ["o", "s", "^", "D", "v", "P", "X", "*"]
marker_map = {
    method: marker_list[i % len(marker_list)]
    for i, method in enumerate(methods)}

line_styles = [":", "--", "-", ":", ","]
linestyle_map = {
    method: line_styles[i % len(line_styles)]
    for i, method in enumerate(methods)}


for distance, distance_df in out_df.groupby("distance"):
    for method, method_df in distance_df.groupby("method"):
        method_df = method_df.sort_values("loss_prob")

        ax.plot(
            method_df["loss_prob"],
            method_df["runtime"],
            color=color_map[distance],
            marker=marker_map[method],
            linestyle=linestyle_map[method],
        )

ax.set_xlabel("Loss probability", fontsize=fs)
ax.set_ylabel("Average runtime", fontsize=fs)
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.tick_params(axis="both", labelsize=fs)

# Legend for distances: colour
# Distance entries: colour
distance_handles = [
    Line2D(
        [0],
        [0],
        color=color_map[distance],
        linewidth=6,
        label=f"{distance}",
    )
    for distance in distances
]

# Method entries: marker and line style
method_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=marker_map[method],
        linestyle=linestyle_map[method],
        linewidth=1.8,
        markersize=6,
        label=str(method),
    )
    for method in methods
]

combined_handles = distance_handles + method_handles
# Text-only section headings
distance_heading = Line2D(
    [], [], linestyle="None", marker=None, label="Distance"
)

method_heading = Line2D(
    [], [], linestyle="None", marker=None, label="Methods"
)

combined_handles = (
    [distance_heading]
    + distance_handles
    + [method_heading]
    + method_handles
)

legend = ax.legend(
    handles=combined_handles,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),   # outside, to the right
    borderaxespad=0,
    fontsize=13,
    handlelength=1,
    labelspacing=0.5,
    frameon=True,
)

# Make the two section headings bold
legend_texts = legend.get_texts()
legend_texts[0].set_weight("bold")
legend_texts[len(distance_handles) + 1].set_weight("bold")

# Reserve space on the right for the legend
fig.subplots_adjust(right=0.72)


fig.tight_layout()

# fig.savefig(
#     os.path.join(dr, f"{a}_runtime.pdf"),
#     dpi=800,
#     bbox_inches="tight",
# )
# plt.show()



fig, ax = plt.subplots(figsize=(8.5, 5))
# ax = fig.add_axes([0.10, 0.15, 0.58, 0.75])

distances = sorted(out_df["distance"].unique())
methods = sorted(out_df["method"].unique())

color_map = {
    distance: plt.cm.tab10(i % 10)
    for i, distance in enumerate(distances)
}
marker_list = ["o", "s", "^", "D", "v", "P", "X", "*"]
marker_map = {
    method: marker_list[i % len(marker_list)]
    for i, method in enumerate(methods)
}
line_styles = [":", "--", "-", ":", ","]
linestyle_map = {
    method: line_styles[i % len(line_styles)]
    for i, method in enumerate(methods)
}

filter_df = out_df1[out_df1["support_size"] != 0]
# filter_df = (
#     filter_df.groupby(["method", "loss_prob", "distance"], as_index=False)
#     [["runtime", "support_size", "teleportation_rate"]]
#     .mean()
# )

for distance, distance_df in filter_df.groupby("distance"):
    for method, method_df in distance_df.groupby("method"):
        method_df = method_df.sort_values("loss_prob")
        method_df = method_df[
                    method_df["support_size"] != 0
                ].sort_values("loss_prob")

        ax.plot(
            method_df["loss_prob"],
            method_df["support_size"],
            color=color_map[distance],
            marker=marker_map[method],
            linestyle=linestyle_map[method],
        )

ax.set_xlabel("Loss probability", fontsize=fs)
ax.set_ylabel("Average support size", fontsize=fs)
ax.set_yscale("log")
ax.grid(True, which="both", alpha=0.3)
ax.tick_params(axis="both", labelsize=fs)

# Legend for distances: colour
# Distance entries: colour
distance_handles = [
    Line2D(
        [0],
        [0],
        color=color_map[distance],
        linewidth=6,
        label=f"{distance}",
    )
    for distance in distances
]

# Method entries: marker and line style
method_handles = [
    Line2D(
        [0],
        [0],
        color="black",
        marker=marker_map[method],
        linestyle=linestyle_map[method],
        linewidth=1.8,
        markersize=6,
        label=str(method),
    )
    for method in methods
]

combined_handles = distance_handles + method_handles

# Text-only section headings
distance_heading = Line2D(
    [], [], linestyle="None", marker=None, label="Distance"
)

method_heading = Line2D(
    [], [], linestyle="None", marker=None, label="Methods"
)

combined_handles = (
    [distance_heading]
    + distance_handles
    + [method_heading]
    + method_handles
)

legend = ax.legend(
    handles=combined_handles,
    loc="upper left",
    bbox_to_anchor=(1.02, 1.0),   # outside, to the right
    borderaxespad=0,
    fontsize=13,
    handlelength=1,
    labelspacing=0.5,
    frameon=True,
)

# Make the two section headings bold
legend_texts = legend.get_texts()
legend_texts[0].set_weight("bold")
legend_texts[len(distance_handles) + 1].set_weight("bold")

# Reserve space on the right for the legend
fig.subplots_adjust(right=0.72)


fig.tight_layout()

# fig.savefig(
#     os.path.join(dr, f"{a}_support.pdf"),
#     dpi=800,
#     bbox_inches="tight",
# )


num_shots = 1000
for method, method_df in out_df.groupby("method"):
    fig, ax = plt.subplots(figsize=(7, 5))

    # ax = fig.add_axes([0.10, 0.15, 0.58, 0.75])

    for dist, distance_df in method_df.groupby("distance"):
        distance_df = distance_df.sort_values("loss_prob")

        distance_df["teleportation_rate_error"] = np.sqrt(
            distance_df["teleportation_rate"]
            * (1 - distance_df["teleportation_rate"])
            / num_shots
        )
        distance_df = distance_df.sort_values("loss_prob")

        ax.errorbar(
            distance_df["loss_prob"],
            distance_df["teleportation_rate"],
            yerr=distance_df["teleportation_rate_error"],
            marker="o",
            label=str(dist),
        )

    ax.set_title(f"Method used: {method}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Teleportation rate", fontsize=fs)

    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(
        title="Distance",
        fontsize=13,
        title_fontsize=fs,
        handlelength=1.3,
        labelspacing=0.3,
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(dr, f"{a}_Threshold_{method}.pdf"),
    #     dpi=800,
    #     bbox_inches="tight",
    # )


plt.show()