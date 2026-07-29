import sys
import os
# import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import seaborn as sns

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

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

dir_name = os.path.join(dir_name, "threshold_folder/2026-07-28")


files = ["101747_sc_thres.csv"]

for ele in files:
    out_list = []
    out_df = pd.read_csv(os.path.join(dir_name, ele))
    print(out_df)

for distance, distance_df in out_df.groupby("distance"):
    fig, ax = plt.subplots(figsize=(7, 5))

    for method, method_df in distance_df.groupby("method"):
        method_df = method_df.sort_values("loss_prob")

        ax.plot(
            method_df["loss_prob"],
            method_df["runtime"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=str(method),
        )

    ax.set_title(f"Code distance {distance}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Runtime", fontsize=fs)

    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(
        title="Method",
        fontsize=13,
        title_fontsize=fs,
        handlelength=1.3,
        labelspacing=0.3,
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(dir_name, f"runtime_distance_{distance}.pdf"),
    #     dpi=800,
    #     bbox_inches="tight",
    # )


for method, method_df in out_df.groupby("method"):
    fig, ax = plt.subplots(figsize=(7, 5))

    for dist, distance_df in method_df.groupby("distance"):
        distance_df = distance_df.sort_values("loss_prob")

        ax.plot(
            distance_df["loss_prob"],
            distance_df["teleportation_rate"],
            marker="o",
            linewidth=1.8,
            markersize=5,
            label=str(dist),
        )

    ax.set_title(f"Method used {method}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Runtime", fontsize=fs)

    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)

    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(
        title="Method",
        fontsize=13,
        title_fontsize=fs,
        handlelength=1.3,
        labelspacing=0.3,
    )

    fig.tight_layout()

    # fig.savefig(
    #     os.path.join(dir_name, f"runtime_distance_{distance}.pdf"),
    #     dpi=800,
    #     bbox_inches="tight",
    # )


plt.show()