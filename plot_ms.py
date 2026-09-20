# Import Python packages
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def roundoff(ele):
    """function to round off float"""
    ele = int(ele*100)/100
    if ele == 0.44:
        ele = 0.45
    return ele

def frozenset_size(x):
    x = str(x).strip()

    if x == "frozenset()":
        return 0

    inside = x.removeprefix("frozenset({").removesuffix("})")
    return len(inside.split(","))

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


def plot_spf_and_gpf_thresholds(in_files, styles, ax, colour, b):
    """
        Plots SPF and GPF teleportation lattice channel rates across given
        sizes.
        Line colors, styles and labels should be provided in a styles dict.
        Adds title and outputs PNG to out_file
    """
    # Initialise figure
    # plt.figure()
    # fig, ax = plt.subplots()
    # Plots input datasets
    for i, spf_in_file in enumerate(in_files):
        if b=="crazy":
            i+=2
        spf_in_file=base_dir/spf_in_file

        # Reads in CSV data into pandas dataframe object
        spf_df = pd.read_csv(spf_in_file, index_col=[0])
        # gpf_df = pd.read_csv(gpf_in_file, index_col=[0])
        # Creates x- and y-value arrays and their associated errors
        spf_x_vals = np.array(spf_df.index.values)
        # gpf_x_vals = np.array(gpf_df.index.values)
        spf_y_vals = np.array(spf_df.prob_tel)
        # gpf_y_vals = np.array(gpf_df.prob_tel)
        spf_y_errs = np.array(spf_df.prob_tel_std)
        # gpf_y_errs = np.array(gpf_df.prob_tel_std)
        # Plots SPF and GPF lines
        ax.errorbar(spf_x_vals, spf_y_vals,
                    yerr = spf_y_errs,
                color=colour[i],
                linestyle='-',
                # marker = 'v',
                label="",)


if __name__ == '__main__':


    plt.rcParams.update({'font.size': 12})
    sys.path.append(os.path.dirname(__file__))
    dr = os.path.dirname(__file__)
    base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()

    fig, ax = plt.subplots(figsize=(7, 5))
    if not os.path.exists('ms_plot'):
        os.makedirs('ms_plot')

    b = "crazy"
    b = "tri"

    if b == "tri":
        in_files =  \
            ['ms_data/W2xL2_TriangularLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
             'ms_data/W3xL3_TriangularLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',
            'ms_data/W4xL4_TriangularLatticeChannel_MW5_10000MC_SPF_TEL_RATE.csv',]
    else:
        in_files =  \
            [
            # 'ms_data/W2xL2_CrazyGraphChannel_MW5_10000MC_SPF_TEL_RATE.csv',
            #  'ms_data/W3xL3_CrazyGraphChannel_MW5_10000MC_SPF_TEL_RATE.csv',
            'ms_data/W4xL4_CrazyGraphChannel_MW5_10000MC_SPF_TEL_RATE.csv',]
    styles = [{'color': 'b', 'line': '-', 'label': '$4 \\times 4$'},
              {'color': 'r', 'line': '-', 'label': '$3 \\times 3$'},
              {'color': 'g', 'line': '-', 'label': '$2 \\times 2$'}]
    out_file = 'ms_plot/tri_SPF_and_GPF_thresholds.pdf'
    title = 'Triangular Lattice'



    i=0
    form = "pdf"
    fs = 15
    dr = os.path.join(dr, "threshold_folder")

    a = "minimise"
    a = "no_min"
    # dates = ["2026-09-06", "2026-09-07"]
    dates = ["2026-09-08", "2026-09-09", "2026-09-10", "2026-09-11",
            "2026-09-18"]
    out_list = []

    if b =="crazy":
        cc = "Crazy"
    elif b == "tri":
        cc = "Triangular"

    for a in dates:
        dir_name = os.path.join(dr, a)
        for file in os.listdir(dir_name):
            if file.endswith(".csv") and "True" in file:
                if b not in file:# and "tree" not in file and "hex" not in file:
                    continue
                if "_None" in file:
                    continue
                df = pd.read_csv(os.path.join(dir_name, file), on_bad_lines="skip", engine="python")
                out_list.append(df)

    out_df = pd.concat(out_list).reset_index(drop=True)

    out_df = out_df.drop(columns=["lost_qubits",
                                'g-SPF X logical', 'g-SPF Z logical',"early runtime"
                                ,"early flag", "g-SPF support size", "g-SPF g",
                                "g-SPF status", ])

    methods = ["g-SPF"]
    plots = [" support size", " runtime", " g", " verify"]

    linestyles = ["-", "-", ":", ","]
    colour = ["#003809", "#CC332D", "#61A6E9","#8B7970", "#276D60", "#02AB99", "#868D00", "#FFDDCC", "#594D47"]

    out_df["Loss probability"] = pd.to_numeric(out_df["Loss probability"], errors="coerce")
    out_df = out_df[out_df["Loss probability"] <= 1]

    ####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
    ####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
    #                    Threshold                         #
    ####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
    ####%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%####
    # for i, method in enumerate(methods):
    method = 'g-SPF'
    for dist, distance_df in out_df.groupby("Distance"):

        j = dist-2

        success_col = method + " success"
        distance_df["Loss probability"] = distance_df["Loss probability"].round(4)

        distance_df[success_col] = pd.to_numeric(distance_df[success_col].replace({"True": 1, "False": 0}), errors="coerce")

        # distance_df = (
        #     distance_df.groupby("Loss probability")
        #     .apply(weighted_mean, success_col, include_groups=False)
        #     .reset_index(name="success_mean")
        #     .sort_values("Loss probability")
        # )
        # ax.plot(
        #     distance_df["Loss probability"],
        #     distance_df["success_mean"],
        #     # linestyle=linestyles[i],
        #     linestyle = "--",
        #     marker="v",
        #     color = colour[j],
        #     alpha=0.4,
        # )
        # ax.plot(
        #     distance_df["Loss probability"],
        #     distance_df["success_mean"],
        #     # linestyle=linestyles[i],
        #     linestyle = "",
        #     marker="v",
        #     label=f"Channel is {dist}x{dist}",
        #     color = colour[j],
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
            linestyle="--",
            marker='o',
            alpha = 0.4,
            color = colour[j]
        )


        ax.errorbar(
            distance_df["Loss probability"],
            distance_df[f"success_mean_{method}"],
            yerr=distance_df[f"success_sem_{method}"],
            # linestyle=linestyles[i],
            linestyle = "",
            marker="o",
            label=f"{dist}x{dist}",
            color = colour[j],
        )


    ax.set_title(f"{cc} lattice graph", fontsize=fs)
    ax.set_xlabel("Loss probability", fontsize=fs)
    ax.set_ylabel("Success rate", fontsize=fs)
    # ax.set_ylabel("Runtime (s)", fontsize=fs)
    # ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.5)
    ax.tick_params(axis="both", labelsize=fs)
    # ax.legend(title="Channel", fontsize=13, title_fontsize=fs, handlelength=1.3, labelspacing=0.3,)

    fig.tight_layout()


    plot_spf_and_gpf_thresholds(in_files, styles, ax, colour,b)

    # techs = ["PyZX", "Qiskit", "Pauli gadget", "Raw"]
    techs = ["Morley-Short et al.", 'D-LoFi']
    markers = ['', 'o']
    lines = ['-', '--']
    tech_handles = [
        Line2D(
            [0],
            [0],
            linewidth=1.5,
            label=f"{distance}",
            color = colour[i],
            marker = markers[i],
            linestyle=lines[i],
        )
        for i, distance in enumerate(techs)
    ]

    if b== "tri":
        channels = [2,3,4,5,6,7]
        # Method entries: marker and line style
        method_handles = [
            Line2D(
                [0],
                [0],
                color=colour[i],
                # linestyle=lines[i],
                linewidth=6.8,
                # markersize=10.5,
                marker = '',
                label=f'{cha}x{cha}',
            )
            for i, cha in enumerate(channels)
        ]
    else:
        channels = [2,3,4,5,6,7]
        # Method entries: marker and line style
        method_handles = [
            Line2D(
                [0],
                [0],
                color=colour[i],
                # linestyle=lines[i],
                linewidth=6.8,
                # markersize=10.5,
                marker = '',
                label=f'{cha}x{cha}',
            )
            for i, cha in enumerate(channels)
        ]


    legend1 = ax.legend(
        handles=tech_handles,
        # title_fontsize=fs,
        # title="Optimiser",
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
        fontsize=13,
        frameon=True,
    )

    ax.add_artist(legend1)

    legend2 = ax.legend(
        handles=method_handles,
        title_fontsize=fs,
        handlelength=1,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.80),
        fontsize=13,
        frameon=True,
    )

    fig.savefig(os.path.join(dr, f"Thresholds_method_{b}.pdf"), dpi=800, bbox_inches="tight")
    plt.show()
