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


form = "pdf"

fs = 15
dr = os.path.join(dr, "threshold_folder")

a = "minimise"
a = "no_min"
# dates = ["2026-09-01", "2026-09-02"]
dates = ["2026-09-09", "2026-09-10", "2026-09-11"]
out_list = []
for a in dates:
    dir_name = os.path.join(dr, a)
    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:
            if "_tri" not in file:
                continue
            if "None" not in file:
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            out_list.append(df)

out_df = pd.concat(out_list).reset_index(drop=True)
out_df = out_df.drop(columns=["lost_qubits",
                              'g-SPF X logical', 'g-SPF Z logical',"early runtime"
                              ,"early flag", "g-SPF support size", "g-SPF g",
                              "g-SPF status", ])
                            #   'ILP X logical', 'ILP Z logical'])

out_list = []
for a in dates:
    dir_name = os.path.join(dr, a)
    for file in os.listdir(dir_name):
        if file.endswith(".csv"):
            if "tri" not in file:
                continue
            if "None" in file:
                continue

            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            out_list.append(df)


target_df = pd.concat(out_list).reset_index(drop=True)
target_df = target_df.drop(columns=["lost_qubits",
                              'g-SPF X logical', 'g-SPF Z logical',"early runtime"
                              ,"early flag", "g-SPF support size", "g-SPF g",
                              "g-SPF status", ])
                            #   'ILP X logical', 'ILP Z logical'])
# methods = ["Heuristic", "ILP", "g-SPF"]
methods = ["g-SPF"]
plots = [" support size", " runtime", " g", " verify"]

linestyles = ["-", "-", ":", ","]
colour = ["#AF4189", "#4171B0", "#4DB041", "#B08A41", "#5B4052"]
colour = ["#CC332D", "#61A6E9","#8B7970", "#276D60", "#DDDFB0", "#FFDDCC", "#594D47"]
markers = [".", "v", "^", "D", "s", "P", "X", "*"]

out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
out_df = out_df[out_df["Loss probability"] <= 1]


####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
#                   t-spf vs 1-spf                     #
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####


target_df["Loss probability"] = pd.to_numeric(target_df["Loss probability"], errors="coerce")
target_df = target_df[target_df["Loss probability"] <= 1]
target_df["Distance"] = pd.to_numeric(target_df["Distance"])
distances = sorted(target_df["Distance"].unique())


fig, ax = plt.subplots(figsize=(7, 5))
linestyles = ["-", "--"]
colour = ["#AF4189", "#4171B0", "#4DB041", "#B08A41", "#5B4052"]
colour = ["#CC332D", "#61A6E9", "#FFDDCC", "#DDDFB0", "#276D60"]
markers = ["o", "v", "^", "D", "s", "P", "X", "*"]

for i, method in enumerate(methods):
    for dist, distance_df in out_df.groupby("Distance"):
        success_col = method + " success"
        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        # distance_df = (distance_df.groupby("Loss probability",
        #                                 as_index=False).agg(success_mean=(success_col, "mean"),
        #                                     sample_count=(success_col, "count"),).sort_values("Loss probability"))
        # distance_df[method + " yerr"] = np.sqrt( distance_df["success_mean"] *
        #                                     (1 - distance_df["success_mean"])
        #                                     / distance_df["sample_count"])

        # ax.plot(
        #     distance_df["Loss probability"],
        #     distance_df["success_mean"],
        #     linestyle=linestyles[i],
        #     marker=markers[i],
        #     label="No fixed target qubit",
        #     color = colour[0]
        # )


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
            label="Not fixed",
            color = colour[0]
        )


    for dist, distance_df in target_df.groupby("Distance"):
        if dist !=5:
            continue
        success_col = method + " success"

        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        # distance_df = (distance_df.groupby("Loss probability",
        #                                 as_index=False).agg(success_mean=(success_col, "mean"),
        #                                     sample_count=(success_col, "count"),).sort_values("Loss probability"))
        # distance_df[method + " yerr"] = np.sqrt( distance_df["success_mean"] *
        #                                     (1 - distance_df["success_mean"])
        #                                     / distance_df["sample_count"])

        # ax.plot(
        #     distance_df["Loss probability"],
        #     distance_df["success_mean"],
        #     linestyle=linestyles[i],
        #     marker=markers[i],
        #     label="Fixed target qubit",
        #     color = colour[1]
        # )


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
            label="Fixed",
            color = colour[1]
        )



if method == "Heuristic":
    method = "H-LoFi"
elif method == "g-SPF":
    method = "D-LoFi"

ax.set_title(f"5x5 triangular lattice graph", fontsize=fs)
ax.set_xlabel("Loss probability", fontsize=fs)
ax.set_ylabel("Success rate", fontsize=fs)
ax.grid(True, which="both", alpha=0.5)
ax.tick_params(axis="both", labelsize=fs)
fig.tight_layout()
ax.set_xticks(np.arange(0, 1.1, 0.1))
ax.legend(title="Target qubit", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)
fig.tight_layout()

# fig.savefig(os.path.join(dr, f"Target_{method}_gc.pdf"), dpi=800, bbox_inches="tight")


for i, method in enumerate(methods):
    if i >0:
        continue
    fig, ax = plt.subplots(figsize=(7, 5))
    for dist, distance_df in out_df.groupby("Distance"):
        # if dist <8:continue
        # success_col = method + " success"
        success_col = method + " runtime"

        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
        # distance_df = distance_df[distance_df["Loss probability"] <= 0.3]

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        # distance_df = (
        #     distance_df.groupby("Loss probability", as_index=False)
        #     .agg(
        #         success_mean=(success_col, "mean"),
        #         sample_count=(success_col, "count"),
        #     )
        #     .sort_values("Loss probability")
        # )

        distance_df = distance_df.groupby("Loss probability", as_index=False).sum()
        # print(distance_df[ "config_count"])

        ax.plot(
            distance_df["Loss probability"],
            distance_df["config_count"],
            label=f"{dist}",
        )

    ax.grid(True, which="both", alpha=0.5)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)


for i, method in enumerate(methods):
    if i >0:
        continue
    fig, ax = plt.subplots(figsize=(7, 5))
    for dist, distance_df in target_df.groupby("Distance"):
        # if dist <8:continue
        # success_col = method + " success"
        success_col = method + " runtime"

        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
        # distance_df = distance_df[distance_df["Loss probability"] <= 0.3]

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")
        # distance_df = (
        #     distance_df.groupby("Loss probability", as_index=False)
        #     .agg(
        #         success_mean=(success_col, "mean"),
        #         sample_count=(success_col, "count"),
        #     )
        #     .sort_values("Loss probability")
        # )

        distance_df = distance_df.groupby("Loss probability", as_index=False).sum()
        # print(distance_df[ "config_count"])

        ax.plot(
            distance_df["Loss probability"],
            distance_df["config_count"],
            label=f"{dist}",
        )

    ax.grid(True, which="both", alpha=0.5)
    ax.legend(title="Distance", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

plt.show()
