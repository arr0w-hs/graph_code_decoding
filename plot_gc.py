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
dr = os.path.join(dr, "threshold_folder")

a = "minimise"
a = "no_min"
dates = ["2026-09-04"]
out_list = []

for a in dates:
    dir_name = os.path.join(dr, a)
    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:
            if "crazy" not in file and "tree" not in file and "hex" not in file:
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            out_list.append(df)

out_df = pd.concat(out_list).reset_index(drop=True)
out_df = out_df.drop(columns=["lost_qubits",
                              'Heuristic X logical', 'Heuristic Z logical',
                              'g-SPF X logical', 'g-SPF Z logical',])
                            #   'ILP X logical', 'ILP Z logical'])

print(out_df)

# methods = ["Heuristic", "ILP", "g-SPF"]
methods = ["Heuristic", "g-SPF"]
plots = [" support size", " runtime", " g", " verify"]

linestyles = ["-", "-", ":", ","]
colour = ["#AF4189", "#4171B0", "#4DB041", "#B08A41", "#5B4052"]
colour = ["#CC332D", "#61A6E9","#8B7970", "#276D60", "#DDDFB0", "#FFDDCC", "#594D47"]
markers = [".", "v", "^", "D", "s", "P", "X", "*"]

out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
out_df = out_df[out_df["Loss probability"] <= 1]



for i, method in enumerate(methods):
    fig, ax = plt.subplots(figsize=(7, 5))
    for dist, distance_df in out_df.groupby("Distance"):
        j = dist-3
        success_col = method + " success"
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        distance_df = (
            distance_df.groupby("Loss probability", as_index=False)
            .agg(
                success_mean=(success_col, "mean"),
                sample_count=(success_col, "count"),
            )
            .sort_values("Loss probability")
        )

        ax.plot(
            distance_df["Loss probability"],
            distance_df["success_mean"],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"{dist}",
            color = colour[j]
        )

    ax.set_title(f"Method used: {method}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    # ax.set_ylabel("Teleportation rate", fontsize=fs)
    ax.set_ylabel("Runtime (s)", fontsize=fs)
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()

    # fig.savefig(os.path.join(dr, f"Thresholds_{method}.pdf"), dpi=800, bbox_inches="tight")


fig, ax = plt.subplots(figsize=(7, 5))
for i, method in enumerate(methods):
    for dist, distance_df in out_df.groupby("Distance"):
        # if dist!=5:
        #     continue
        # j = dist//2-1-1

        j = dist//2-1
        # success_col = method + " success"
        success_col = method + " runtime"
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        distance_df = (
            distance_df.groupby("Loss probability", as_index=False)
            .agg(
                success_mean=(success_col, "mean"),
                sample_count=(success_col, "count"),
            )
            .sort_values("Loss probability")
        )

        ax.plot(
            distance_df["Loss probability"],
            distance_df["success_mean"],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"{method}",
            color = colour[i]
        )

    ax.set_title(f"Distance-5 rotated surface code", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    # ax.set_ylabel("Teleportation rate", fontsize=fs)
    ax.set_ylabel("Runtime (s)", fontsize=fs)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()

    # fig.savefig(os.path.join(dr, f"Runtime_d5.pdf"), dpi=800, bbox_inches="tight")


# fig, ax = plt.subplots(figsize=(7, 5))
# for i, method in enumerate(methods):
#     for loss, loss_df in out_df.groupby("Loss probability"):
#         if loss > 0.13 or loss < 0.07:
#             continue
#         print(loss)
#         # j = dist//2-1
#         # success_col = method + " success"
#         success_col = method + " runtime"
#         loss_df[success_col] = pd.to_numeric(loss_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
#         loss_df = (
#             loss_df.groupby("Distance", as_index=False)
#             .agg(
#                 success_mean=(success_col, "mean"),
#             )
#             .sort_values("Distance")
#         )

#         ax.plot(
#             loss_df["Distance"],
#             loss_df["success_mean"],
#             linestyle=linestyles[i],
#             marker=markers[i],
#             label=f"{method}",
#             color = colour[i]
#         )

#     ax.set_title(f"Runtime rotated surface code", fontsize=fs)
#     ax.set_xlabel("Distance", fontsize=fs)
#     # ax.set_ylabel("Teleportation rate", fontsize=fs)
#     ax.set_ylabel("Runtime (s)", fontsize=fs)
#     ax.set_yscale("log")
#     ax.grid(True, which="both", alpha=0.3)
#     ax.tick_params(axis="both", labelsize=fs)
#     ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

#     fig.tight_layout()

#     # fig.savefig(os.path.join(dr, f"p_0.1.pdf"), dpi=800, bbox_inches="tight")


plt.show()
