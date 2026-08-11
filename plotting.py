import ast
import sys
import os
# import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
from matplotlib.lines import Line2D
from matplotlib.colors import LogNorm
import math


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


dates = ["2026-08-05", "2026-08-06", "2026-08-08", "2026-08-09"]
out_list = []
skip_d3_files = ["224415_sc_thres_min_True_1.csv",
                "224421_sc_thres_min_True_1.csv",
                 "224422_sc_thres_min_True_1.csv",
                 "224435_sc_thres_min_True_1.csv",
                 "224522_sc_thres_min_True_1.csv",
                 ]
for a in dates:
    dir_name = os.path.join(dr, a)
    # for file in os.listdir(dir_name):
    #     if file.endswith(".csv"):
    #         if "True" in file:
    #             if "_5" in file:
    #                 continue
    #             # print(file)
    #             df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine='python')
    #             out_list.append(df)
    #         # print(file)

    #         skip_d3_files = ["file1.csv", "file2.csv"]

    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:
            if "_5" in file:
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")

            # if file in skip_d3_files:
            if file in skip_d3_files:
                df = df[pd.to_numeric(df["Distance"], errors="coerce") != 3]

            out_list.append(df)

# out_df = pd.concat(ele for ele in out_list)
# print(out_df)
out_df = pd.concat(out_list).reset_index(drop=True)
out_df["Lost qubits"] = out_df["lost_qubits"].apply(
    lambda x: len(ast.literal_eval(x.replace("frozenset", ""))) if isinstance(x, str) else 0
)

out_df = out_df.drop(columns=["lost_qubits",
                              'Heuristic X logical', 'Heuristic Z logical',
                              'g-SPF X logical', 'g-SPF Z logical',
                              'ILP X logical', 'ILP Z logical'])

print(out_df.keys())

methods = ["Heuristic", "ILP", "g-SPF"]
plots = [" support size", " runtime", " g", " verify"]
# plots = [" support size"]
# plots = [" runtime"]
# plots = [" g"]
# plots = [" support size", " runtime"]

markers = ["o", "v", "^"]
linestyles = [":", "--", "-"]

# print(out_df["Loss probability"])
# out_df = out_df[out_df["Loss probability"] <= 1]
out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
out_df = out_df[out_df["Loss probability"] <= 1]

# print(out_df[pd.to_numeric(out_df["Distance"], errors="coerce").isna()])
out_df["Distance"] = pd.to_numeric(out_df["Distance"])
distances = sorted(out_df["Distance"].unique())

plots = [" support size"]


methods = ["Heuristic", "ILP", "g-SPF"]
plots = [" support size", " runtime", " g", " verify"]
# plots = [" support size"]
# plots = [" runtime"]
# plots = [" g"]
plots = [" support size", " runtime"]


# print(out_df["Loss probability"])
# out_df = out_df[out_df["Loss probability"] <= 1]
out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
out_df = out_df[out_df["Loss probability"] <= 1]

# print(out_df[pd.to_numeric(out_df["Distance"], errors="coerce").isna()])
out_df["Distance"] = pd.to_numeric(out_df["Distance"])
distances = sorted(out_df["Distance"].unique())

for p in plots:

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
    axes = axes.flatten()

    color_map = {
        distance: plt.cm.tab10(i % 10)
        for i, distance in enumerate(distances)
    }

    # marker_list = ["o", "v", "^", "D", "s", "P", "X", "*"]
    # marker_map = {
    #     method: marker_list[i % len(marker_list)]
    #     for i, method in enumerate(methods)
    # }

    line_styles = [":", "--", "-", "-."]
    linestyle_map = {
        method: line_styles[i % len(line_styles)]
        for i, method in enumerate(methods)
    }

    for i, (d, distance_df) in enumerate(out_df.groupby("Distance")):

        distance_df["Lost qubits"] = (distance_df["Lost qubits"]/ (2*d**2+2*d-1))

        ax = axes[i]
        distance_df = distance_df.copy()

        for method in methods:
            col = method + p
            distance_df[col] = pd.to_numeric(
                distance_df[col].replace({"True": 1, "False": 0}),
                errors="coerce"
            )

        cols = ["Lost qubits"] + [method + p for method in methods]

        distance_df[cols[1:]] = distance_df[cols[1:]].apply(
            pd.to_numeric, errors="coerce"
        )

        distance_df = (
            distance_df[cols]
            .groupby("Lost qubits", as_index=False)
            .mean()
                )
        heuristic = distance_df["Heuristic" + p].copy()

        for method in methods:
            # distance_df[method + p] = distance_df[method + p] / heuristic

            ax.plot(
                distance_df["Lost qubits"],
                distance_df[method + p],
                # marker=marker_map[method],
                linestyle=linestyle_map[method],
                label=method
            )

        ax.set_title(f"Distance {d}", fontsize=fs)
        ax.set_xlabel("Lost qubits", fontsize=fs)
        ax.grid(True, which="both", alpha=0.3)
        ax.tick_params(axis="both", labelsize=fs)

        ylabel = "logical overlap" if p == " g" else p
        ax.set_ylabel(f"Average {ylabel}", fontsize=fs)

        if p == " runtime":
            ax.set_yscale("log")

    # one common legend
    method_handles = [
        Line2D(
            [0], [0],
            color="black",
            # marker=marker_map[method],
            linestyle=linestyle_map[method],
            label=method
        )
        for method in methods
    ]

    fig.legend(
        handles=method_handles,
        loc="center right",
        bbox_to_anchor=(0.99, 0.5),
        fontsize=13
    )

    fig.tight_layout(rect=[0, 0, 0.88, 1])


    # fig.savefig(os.path.join(dr, f"Threshold_{method}_test1.pdf"), dpi=800, bbox_inches="tight")

# for p in plots:

#     fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True, sharey=True)
#     axes = axes.flatten()


#     for i, (d, distance_df) in enumerate(out_df.groupby("Distance")):

#         ax = axes[i]
#         distance_df = distance_df.copy()

#         for method in methods:
#             col = method + p
#             distance_df[col] = pd.to_numeric(
#                 distance_df[col].replace({"True": 1, "False": 0}),
#                 errors="coerce"
#             )

#         cols = ["Loss probability"] + [method + p for method in methods]

#         distance_df[cols[1:]] = distance_df[cols[1:]].apply(
#             pd.to_numeric, errors="coerce"
#         )


#         # distance_df["Loss probability"] = (distance_df["Loss probability"]/ (2*d**2+2*d-1))

#         distance_df = (
#             distance_df[cols]
#             .groupby("Loss probability", as_index=False)
#             .mean()
#                 )
#         heuristic = distance_df["Heuristic" + p].copy()

#         for method in methods:
#             distance_df[method + p] = distance_df[method + p] / heuristic

#             ax.plot(
#                 distance_df["Loss probability"],
#                 distance_df[method + p],
#                 label=method
#             )

#         ax.set_title(f"Distance {d}", fontsize=fs)
#         ax.set_xlabel("Loss probability", fontsize=fs)
#         ax.grid(True, which="both", alpha=0.3)
#         ax.tick_params(axis="both", labelsize=fs)

#         ylabel = "logical overlap" if p == " g" else p
#         ax.set_ylabel(f"Average {ylabel}", fontsize=fs)

#         if p == " runtime":
#             ax.set_yscale("log")

#     # one common legend
#     method_handles = [
#         Line2D(
#             [0], [0],
#             color="black",
#             label=method
#         )
#         for method in methods
#     ]

#     fig.legend(
#         handles=method_handles,
#         loc="center right",
#         bbox_to_anchor=(0.99, 0.5),
#         fontsize=13
#     )

#     fig.tight_layout(rect=[0, 0, 0.88, 1])


#     fig.savefig(os.path.join(dr, f"Threshold_{method}_test1.pdf"), dpi=800, bbox_inches="tight")

for method in methods:
    fig, ax = plt.subplots(figsize=(7, 5))
    for d, distance_df in out_df.groupby("Distance"):
        success_col = method + " success"

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")

        # distance_df["Loss probability"] = (distance_df["Loss probability"]/ (2*d**2+2*d-1))

        distance_df = (
            distance_df.groupby("Loss probability", as_index=False)
            .agg(
                success_mean=(success_col, "mean"),
                sample_count=(success_col, "count"),
            )
            .sort_values("Loss probability")
        )
        distance_df[method + " yerr"] = np.sqrt( distance_df["success_mean"] *
                                            (1 - distance_df["success_mean"])
                                            / distance_df["sample_count"])

        ax.errorbar(
            distance_df["Loss probability"],
            distance_df["success_mean"],
            yerr=distance_df[method + " yerr"],
            label=str(d),
        )

    ax.set_title(f"Method used: {method}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Teleportation rate", fontsize=fs)
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()



for method in methods:
    fig, ax = plt.subplots(figsize=(7, 5))
    for d, distance_df in out_df.groupby("Distance"):
        success_col = method + " success"

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")

        distance_df["Lost qubits"] = (distance_df["Lost qubits"]/ (2*d**2+2*d-1))

        distance_df = (
            distance_df.groupby("Lost qubits", as_index=False)
            .agg(
                success_mean=(success_col, "mean"),
                sample_count=(success_col, "count"),
            )
            .sort_values("Lost qubits")
        )
        distance_df[method + " yerr"] = np.sqrt( distance_df["success_mean"] *
                                            (1 - distance_df["success_mean"])
                                            / distance_df["sample_count"])

        ax.errorbar(
            distance_df["Lost qubits"],
            distance_df["success_mean"],
            yerr=distance_df[method + " yerr"],
            label=str(d),
        )

    ax.set_title(f"Method used: {method}", fontsize=fs)
    ax.set_xlabel("Lost qubits", fontsize=fs)
    ax.set_ylabel("Teleportation rate", fontsize=fs)
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()


plt.show()
