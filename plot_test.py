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

# skip_d3_files = ["224415_sc_thres_min_True_1.csv",
#                 "224421_sc_thres_min_True_1.csv",
#                  "224422_sc_thres_min_True_1.csv",
#                  "224435_sc_thres_min_True_1.csv",
#                  "224522_sc_thres_min_True_1.csv",
#                  ]
skip_d3_files = ["104748_sc_thres_min_True_3.csv"]

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
            if "_3" in file or "bb" in file:
                continue
            elif "_5" in file or "_17" in file:
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")

            # if file in skip_d3_files:
            if file in skip_d3_files:
                df = df[pd.to_numeric(df["Distance"], errors="coerce") != 3]

            out_list.append(df)

# out_df = pd.concat(ele for ele in out_list)
# print(out_df)
out_df = pd.concat(out_list).reset_index(drop=True)
out_df = out_df.drop(columns=["lost_qubits",
                              'Heuristic X logical', 'Heuristic Z logical',
                              'g-SPF X logical', 'g-SPF Z logical',])
                            #   'ILP X logical', 'ILP Z logical'])


# methods = ["Heuristic", "ILP", "g-SPF"]
methods = ["Heuristic", "g-SPF"]
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

for p in plots:
    fig, ax = plt.subplots(figsize=(8.5, 5))
    # out_df["Distance"] = pd.to_numeric(out_df["Distance"])
    # distances = sorted(out_df["Distance"].unique())
    # distances = sorted(out_df["Distance"].unique())

    color_map = {
        distance: plt.cm.tab10(i % 10)
        for i, distance in enumerate(distances)}

    marker_list = ["o", "v", "^", "D", "s", "P", "X", "*"]
    marker_map = {
        method: marker_list[i % len(marker_list)]
        for i, method in enumerate(methods)}

    line_styles = [":", "--", "-", ":", ","]
    linestyle_map = {
        method: line_styles[i % len(line_styles)]
        for i, method in enumerate(methods)}


    for i, (distance, distance_df) in enumerate(out_df.groupby("Distance")):
        distance_df = distance_df.copy()
        # print(distance_df.keys())
        for method in methods:
            col = method + p
            # distance_df[col] = distance_df[col].replace({"True": 1, "False": 0, True: 1, False: 0})
            distance_df[col] = pd.to_numeric(distance_df[col].replace({"True": 1, "False": 0}), errors="coerce")

        cols = ["Loss probability"] + [method + p for method in methods]
        # print(distance_df)
        distance_df = distance_df[cols]
        # print(distance_df)
        # for col in cols[1:]:
        #     print(col, distance_df.loc[pd.to_numeric(distance_df[col], errors="coerce").isna() & distance_df[col].notna(), col].unique())

        # distance_df = distance_df[cols].groupby("Loss probability", as_index=False).mean()

        distance_df[cols[1:]] = distance_df[cols[1:]].apply(pd.to_numeric, errors="coerce")
        distance_df = distance_df[cols].groupby("Loss probability", as_index=False).mean()
        for j, method in enumerate(methods):
            ax.plot(distance_df["Loss probability"], distance_df[method + p], color=plt.cm.tab10(i % 10), marker=markers[j], linestyle=linestyles[j], label=f"{method}, distance {distance}")

    ax.grid()
    ax.legend()

    ax.set_xlabel("Loss probability", fontsize=fs)
    if p == " g":
        p = "logical overlap"
    ax.set_ylabel(f"Average {p}", fontsize=fs)
    if p == " runtime":
        ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs)

    ax.set_xlabel("Loss probability", fontsize=fs)
    if p == " g":
        p = "logical overlap"
    ax.set_ylabel(f"Average {p}", fontsize=fs)
    if p == " runtime":
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
    distance_heading = Line2D(
        [], [], linestyle="None", marker=None, label="Distance")

    method_heading = Line2D(
        [], [], linestyle="None", marker=None, label="Methods")

    combined_handles = (
        [distance_heading]
        + distance_handles
        + [method_heading]
        + method_handles)

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
    # fig.savefig(os.path.join(dr, f"{p}_{method}.pdf"), dpi=800, bbox_inches="tight")


# # # plt.show()
# # plots = [" support size"]
# # df = (out_df[np.isclose(out_df["Loss probability"], 0.6) & (out_df["Distance"] == 7)])
# # print(df["ILP support size"])

# # for p in plots:

# #     fig, ax = plt.subplots(figsize=(8.5, 5))

# #     for i, (distance, distance_df) in enumerate(out_df.groupby("Distance")):
# #         if distance != 7:
# #             continue
# #         distance_df = distance_df.copy()
# #         for method in methods:
# #             col = method + p
# #             distance_df[col] = pd.to_numeric(distance_df[col].replace({"True": 1, "False": 0}), errors="coerce")

# #         cols = ["Loss probability"] + [method + p for method in methods]

# #         distance_df = distance_df[cols]
# #         # print(distance_df[distance_df["Loss probability"==0.6]])

# #         distance_df[cols[1:]] = distance_df[cols[1:]].apply(pd.to_numeric, errors="coerce")
# #         distance_df = distance_df[cols].groupby("Loss probability", as_index=False).mean()
# #         for j, method in enumerate(methods):
# #             ax.plot(distance_df["Loss probability"], distance_df[method + p], color=plt.cm.tab10(i % 10), marker=markers[j], linestyle=linestyles[j], label=f"{method}, distance {distance}")

# #     ax.grid()
# #     ax.legend()

# #     fig.tight_layout()


# # num_shots = 1000
for method in methods:
    fig, ax = plt.subplots(figsize=(7, 5))
    for dist, distance_df in out_df.groupby("Distance"):
        success_col = method + " success"

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        # distance_df[success_col] = distance_df[success_col].astype(int)
        # distance_df = (distance_df.groupby("Loss probability", as_index=False).agg(
        #     success_mean=(success_col, "mean"),
        #     sample_count=(success_col, "count"),).sort_values("Loss probability"))
        # num_shots = 1000   # or 2000

        # distance_df = (
        #     distance_df.groupby("Loss probability", group_keys=False)
        #     .apply(lambda x: x.sample(n=min(len(x), num_shots)))
        # )

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
            marker="o",
            label=str(dist),
        )

    ax.set_title(f"Method used: {method}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Teleportation rate", fontsize=fs)
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()

    # fig.savefig(os.path.join(dr, f"Threshold_{method}.pdf"), dpi=800, bbox_inches="tight")


# plt.show()
fig, ax = plt.subplots(figsize=(7, 5))

for i, method in enumerate(methods):
    if i >0:
        continue
    for dist, distance_df in out_df.groupby("Distance"):
        success_col = method + " success"

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        # distance_df[success_col] = distance_df[success_col].astype(int)

        # distance_df = (
        #     distance_df.groupby("Loss probability", group_keys=False)
        #     .apply(lambda x: x.sample(n=min(len(x), num_shots), random_state=1)))

        distance_df = (distance_df.groupby("Loss probability", as_index=False).agg(
            success_mean=(success_col, "mean"),
            sample_count=(success_col, "count"),).sort_values("Loss probability"))

        ax.plot(
            distance_df["Loss probability"],
            distance_df["sample_count"],
            marker="o",
            label=str(dist),
        )

    ax.set_title(f"Method used: {method}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Count", fontsize=fs)
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()


# # # print(out_df.keys())

# # for dist, distance_df in out_df.groupby("Distance"):
# #     distance_df = distance_df[np.isclose(distance_df["Loss probability"], 0.55)]
# #     plt.figure()
# #     plt.title(f"Runtime for {dist}")
# #     plt.hist(distance_df["g-SPF runtime"], bins=30)
# #     plt.xlabel("g-SPF runtime")
# #     plt.ylabel("Number of occurrences")


# for dist, distance_df in out_df.groupby("Distance"):

#     probabilities = np.sort(distance_df["Loss probability"].round(2).unique())
#     runtime_edges = np.histogram_bin_edges(
#         distance_df["g-SPF runtime"].dropna(),
#         bins=30,
#     )

#     histogram_matrix = []

#     for probability in probabilities:
#         runtimes = distance_df.loc[
#             np.isclose(distance_df["Loss probability"], probability),
#             "g-SPF runtime",
#         ]

#         counts, _ = np.histogram(runtimes, bins=runtime_edges)
#         histogram_matrix.append(counts)

#     histogram_matrix = np.array(histogram_matrix).T

#     plt.figure()
#     plt.imshow(
#     histogram_matrix,
#     aspect="auto",
#     origin="lower",
#     extent=[
#         probabilities.min(),
#         probabilities.max(),
#         runtime_edges.min(),
#         runtime_edges.max(),
#     ],
# )
#     plt.xlabel("Loss probability")
#     plt.ylabel("g-SPF runtime")
#     plt.title(f"Distance = {dist}")
#     plt.colorbar(label="Number of occurrences")

# probability = 0.05

# fig, ax = plt.subplots(figsize=(8.5, 5))

# plot_df = out_df[
#     np.isclose(out_df["Loss probability"], probability)
# ].copy()

# for method in methods:
#     col = method + " runtime"
#     plot_df[col] = pd.to_numeric(plot_df[col], errors="coerce")

# plot_df = (
#     plot_df.groupby("Distance", as_index=False)[
#         [method + " runtime" for method in methods]
#     ]
#     .mean()
# )

# plot_df["Distance"] = 2*plot_df["Distance"] ** 2-2*plot_df["Distance"]+1

# for j, method in enumerate(methods):
#     ax.plot(
#         plot_df["Distance"],
#         plot_df[method + " runtime"],
#         marker=markers[j],
#         linestyle=linestyles[j],
#         label=method,
#     )

# ax.set_xlabel("Distance")
# ax.set_ylabel("Runtime")
# ax.set_yscale("log")
# ax.set_title(f"Loss probability = {probability}")
# ax.grid()
# ax.legend()


plt.show()
