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


def weighted_mean(group, success_col):

    return (
        (group[success_col] * group["config_count"]).sum()
        / group["config_count"].sum()
    )

def weighted_sem(group, success_col):
    x = group[success_col].to_numpy()
    w = group["config_count"].to_numpy()

    n = np.sum(w)
    mean = np.sum(w * x) / n
    # print(n)
    variance = np.sum(w * (x - mean) ** 2) / (n - 1)

    return np.sqrt(variance / n)


# def weighted_sem(group):
#     x = group[success_col].to_numpy()
#     w = group["config_count"].to_numpy()

#     mean = np.sum(w * x) / np.sum(w)

#     weighted_variance = (
#         np.sum(w * (x - mean) ** 2)
#         / (np.sum(w) - np.sum(w ** 2) / np.sum(w))
#     )

#     n_eff = np.sum(w) ** 2 / np.sum(w ** 2)

#     return np.sqrt(weighted_variance / n_eff)

form = "pdf"
save = True
save = False

fs = 15
dr = os.path.join(dr, "threshold_folder")

a = "minimise"
a = "no_min"
# dates = ["2026-09-01", "2026-09-02"]
dates = ["2026-09-09", "2026-09-14", "2026-09-15", ]
out_list = []
for a in dates:
    dir_name = os.path.join(dr, a)
    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:
            if "_tri" in file or "bb" in file:
                continue
            elif "crazy" in file or "_17" in file:
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            out_list.append(df)

out_df = pd.concat(out_list).reset_index(drop=True)
out_df = out_df.drop(columns=["lost_qubits",
                              'Heuristic X logical', 'Heuristic Z logical',
                              'g-SPF X logical', 'g-SPF Z logical',])
                            #   'ILP X logical', 'ILP Z logical'])

# methods = ["Heuristic", "ILP", "g-SPF"]
# methods = ["Heuristic", "g-SPF"]
methods = ["g-SPF", "Heuristic"]
# plots = [" support size", " runtime", " g", " verify"]
plots = [" support size", " runtime"]


linestyles = ["-", "-", ":", "-."]
colour = ["#AF4189", "#4171B0", "#4DB041", "#B08A41", "#5B4052"]
colour = ["#CC332D", "#61A6E9","#8B7970", "#276D60", "#DDDFB0", "#FFDDCC", "#594D47"]
markers = ["o", "v", "^", "D", "s", "P", "X", "*"]
# markers = [" ", " ", "^", "D", "s", "P", "X", "*"]

out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
out_df = out_df[out_df["Loss probability"] <= 1]



####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
#                    Threshold                         #
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
for i, method in enumerate(methods):
    fig, ax = plt.subplots(figsize=(7, 5))
    for dist, distance_df in out_df.groupby("Distance"):
        j = dist//2-1
        success_col = method + " success"

        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")

        mean_df = (
            distance_df.groupby("Loss probability")
            .apply(weighted_mean, success_col, include_groups=False)
            .reset_index(name=f"success_mean_{method}")
        )

        sem_df = (
            distance_df.groupby("Loss probability")
            .apply(weighted_sem, success_col, include_groups=False)
            .reset_index(name=f"success_sem_{method}")
        )

        distance_df = (
            mean_df
            .merge(sem_df, on="Loss probability")
            .sort_values("Loss probability")
        )

        ax.errorbar(
            distance_df["Loss probability"],
            distance_df[f"success_mean_{method}"],
            yerr=distance_df[f"success_sem_{method}"],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"{dist}",
            color = colour[j]
        )

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(f"Method used: {method}", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    # ax.set_ylabel("Success rate", fontsize=fs)
    ax.set_ylabel("Success rate", fontsize=fs)
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.5)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(dr, f"Thresholds_{method}.pdf"), dpi=800, bbox_inches="tight")


####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
#                     Overlap-5
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
codedd = 5
fig, ax = plt.subplots(figsize=(7, 5))
for i, method in enumerate(methods):
    success_col = method + " support size"
    for dist, distance_df in out_df.groupby("Distance"):
        if dist!=codedd:
            continue
        j = dist//2-1-1
        # success_col = method + " success"
        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")

        # distance_df[success_col] = distance_df.apply(lambda row: (row[success_col]*row[method+" success"]
        #                                                     if row[method+" success"] is True
        #                                                     else 0),
        #                                 axis=1)
        distance_df = distance_df[distance_df["Heuristic success"]]
        distance_df = distance_df[distance_df["g-SPF success"]]
        # print(distance_df.keys())
        mean_df = (
            distance_df.groupby("Loss probability")
            .apply(weighted_mean, success_col, include_groups=False)
            .reset_index(name=f"success_mean")
            .sort_values("Loss probability")
        )

        sem_df = (
            distance_df.groupby("Loss probability")
            .apply(weighted_sem, success_col, include_groups=False)
            .reset_index(name=f"success_sem_{method}")
            .sort_values("Loss probability")
        )

        # print(dist)
        print(max(mean_df["success_mean"])/(2*dist-1))
        print(max(mean_df["success_mean"]))

        # ax.errorbar(
        #             distance_df["Loss probability"],
        #             distance_df[f"success_mean_{method}"],
        #             yerr=distance_df[f"success_sem_{method}"],
        #             linestyle=linestyles[i],
        #             marker=markers[i],
        #             label=f"{dist}",
        #             color = colour[j]
        #         )

        ax.errorbar(
            mean_df["Loss probability"],
            mean_df[f"success_mean"],
            # distance_df[f"success_mean"],
            yerr=sem_df[f"success_sem_{method}"],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"{method}",
            color = colour[i],
        )

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(f"Distance-{codedd} rotated surface code", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Average logical support size", fontsize=fs)
    ax.grid(True, which="both", alpha=0.5)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(dr, f"sup_size_d5.pdf"), dpi=800, bbox_inches="tight")


####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
#                     Overlap-5
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
fig, ax = plt.subplots(figsize=(7, 5))
for i, method in enumerate(methods):
    success_col = method + " support size"
    out_df["Loss probability"] = out_df["Loss probability"].round(4)
    for prob, distance_df in out_df.groupby("Loss probability"):
        # j = dist//2-1-1
        s = 0.2
        if prob != s:
            continue
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        distance_df = distance_df[distance_df["Heuristic success"]]
        distance_df = distance_df[distance_df["g-SPF success"]]
        distance_df[success_col] = 100*distance_df[success_col]/(2 * distance_df["Distance"]-1)-100

        # distance_df = distance_df[distance_df["Loss probability"] == 0.1]
        # distance_df = (
        #     distance_df.groupby("Distance")
        #     .apply(weighted_mean, success_col, include_groups=False)
        #     .reset_index(name=f"success_mean")
        #     .sort_values("Distance")
        # )
        # ax.plot(
        #     distance_df["Distance"],
        #     100*distance_df[f"success_mean"]/(2 * distance_df["Distance"]-1)-100,
        #     # distance_df[f"success_mean"],
        #     linestyle=linestyles[i],
        #     marker=markers[i],
        #     label=f"{method}",
        #     color = colour[i]
        # )

        mean_df = (
            distance_df.groupby("Distance")
            .apply(weighted_mean, success_col, include_groups=False)
            .reset_index(name=f"success_mean")
            .sort_values("Distance")
        )
        sem_df = (
            distance_df.groupby("Distance")
            .apply(weighted_sem, success_col, include_groups=False)
            .reset_index(name=f"success_sem")
            .sort_values("Distance")
        )
        ax.errorbar(
            mean_df["Distance"],
            mean_df[f"success_mean"],
            # distance_df[f"success_mean"],
            yerr=sem_df[f"success_sem"],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"{method}",
            color = colour[i],
        )

    ax.set_xticks(np.arange(3, 11, 2))
    ax.set_title(f"Logical support size (p = {s})", fontsize=fs)
    ax.set_xlabel("Distance", fontsize=fs)
    ax.set_ylabel("Increase in average\nsupport size (%)", fontsize=fs)
    ax.grid(True, which="both", alpha=0.5)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(dr, f"sup_size_d5_p01.pdf"), dpi=800, bbox_inches="tight")



###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
#                    Runtime-5                        #
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
###%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####

fig, ax = plt.subplots(figsize=(7, 5))
for i, method in enumerate(methods):
    success_col = method + " runtime"
    for dist, distance_df in out_df.groupby("Distance"):
        if dist!=5:
            continue
        j = dist//2-1-1
        # success_col = method + " success"
        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")

        # distance_df = distance_df[distance_df["early flag"] == False]

        # distance_df = (
        #     distance_df.groupby("Loss probability", as_index=False)
        #     .agg(
        #         success_mean=(success_col, "mean"),
        #         sample_count=(success_col, "count"),
        #     )
        #     .sort_values("Loss probability")
        # )
        # ax.plot(
        #     distance_df["Loss probability"],
        #     distance_df[f"success_mean"],
        #     linestyle=linestyles[i],
        #     marker=markers[i],
        #     label=f"{method}",
        #     color = colour[i]
        # )


        mean_df = (
            distance_df.groupby("Loss probability")
            .apply(weighted_mean, success_col, include_groups=False)
            .reset_index(name=f"success_mean")
            .sort_values("Loss probability")
        )
        sem_df = (
            distance_df.groupby("Loss probability")
            .apply(weighted_sem, success_col, include_groups=False)
            .reset_index(name=f"success_sem")
            .sort_values("Loss probability")
        )
        ax.errorbar(
            mean_df["Loss probability"],
            mean_df[f"success_mean"],
            yerr=sem_df[f"success_sem"],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"{method}",
            color = colour[i],
        )

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_title(f"Distance-5 rotated surface code", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_yscale("log")
    ax.set_ylabel("Average runtime (s)", fontsize=fs)
    ax.grid(True, which="both", alpha=0.5)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(dr, f"Runtime_d5.pdf"), dpi=800, bbox_inches="tight")


####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
#                     Runtime fixed p
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
fig, ax = plt.subplots(figsize=(7, 5))
for i, method in enumerate(methods):
    for loss, loss_df in out_df.groupby("Loss probability"):
        if loss > 0.13 or loss < 0.07:
            continue
        success_col = method + " runtime"

        loss_df["Loss probability"] = loss_df["Loss probability"].round(4)
        loss_df[success_col] = pd.to_numeric(loss_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        # loss_df = (
        #     loss_df.groupby("Distance", as_index=False)
        #     .agg(success_mean=(success_col, "mean"),)
        #     .sort_values("Distance")
        # )
        # ax.plot(
        #     loss_df["Distance"],
        #     loss_df["success_mean"],
        #     linestyle=linestyles[i],
        #     marker=markers[i],
        #     label=f"{method}",
        #     color = colour[i]
        # )


        mean_df = (
            loss_df.groupby("Distance")
            .apply(weighted_mean, success_col, include_groups=False)
            .reset_index(name=f"success_mean")
            .sort_values("Distance")
        )
        sem_df = (
            loss_df.groupby("Distance")
            .apply(weighted_sem, success_col, include_groups=False)
            .reset_index(name=f"success_sem")
            .sort_values("Distance")
        )
        ax.errorbar(
            mean_df["Distance"],
            mean_df[f"success_mean"],
            yerr=sem_df[f"success_sem"],
            linestyle=linestyles[i],
            marker=markers[i],
            label=f"{method}",
            color = colour[i],
        )

    ax.set_xticks(np.arange(3, 11, 2))
    ax.set_title(f"Runtime rotated surface code", fontsize=fs)
    ax.set_xlabel("Distance", fontsize=fs)
    # ax.set_ylabel("Success rate", fontsize=fs)
    ax.set_ylabel("Average runtime (s)", fontsize=fs)
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.5)
    ax.tick_params(axis="both", labelsize=fs)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()

    if save:
        fig.savefig(os.path.join(dr, f"p_0.1.pdf"), dpi=800, bbox_inches="tight")




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
