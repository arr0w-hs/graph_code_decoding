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


def weighted_mean(group):

    return (
        (group[success_col] * group["config_count"]).sum()
        / group["config_count"].sum()
    )

form = "pdf"

fs = 15
dr = os.path.join(dr, "threshold_folder")

a = "minimise"
a = "no_min"
# dates = ["2026-09-01", "2026-09-02"]
dates = ["2026-09-09"]
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

out_list = []
for a in dates:
    dir_name = os.path.join(dr, a)
    for file in os.listdir(dir_name):
        if file.endswith(".csv") and "True" in file:
            if "_17" not in file:
                continue
            df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
            out_list.append(df)


target_df = pd.concat(out_list).reset_index(drop=True)
target_df = target_df.drop(columns=["lost_qubits",
                              'Heuristic X logical', 'Heuristic Z logical',
                              'g-SPF X logical', 'g-SPF Z logical',])
                            #   'ILP X logical', 'ILP Z logical'])
# methods = ["Heuristic", "ILP", "g-SPF"]
methods = ["Heuristic", "g-SPF"]
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
markers = [".", "v", "^", "D", "s", "P", "X", "*"]

for i, method in enumerate(methods):
    for dist, distance_df in out_df.groupby("Distance"):
        if dist !=7:
            continue
        success_col = method + " success"
        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)
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
ax.set_ylabel("Success rate", fontsize=fs)
ax.grid(True, which="both", alpha=0.5)
ax.tick_params(axis="both", labelsize=fs)
fig.tight_layout()

# fig.savefig(os.path.join(dr, f"Target_{method}.pdf"), dpi=800, bbox_inches="tight")


plt.show()
