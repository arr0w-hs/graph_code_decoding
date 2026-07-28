from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from galois import GF2
import pandas as pd
from threshold import compute_teleportation_rate
import stabiliser_code as sc
import tableau as ta
from code_importer import rotated_surface_code, surface_code
from collections import defaultdict



ts = pd.Timestamp.now(tz="Europe/Stockholm")
date_str = ts.strftime("%Y-%m-%d")
time_str = ts.strftime("%H%M%S")
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_directory = base_dir / "threshold_folder" / f"{date_str}"
data_directory.mkdir(parents=True, exist_ok=True)


out = defaultdict(list)
m = ["heuristic" ,"gspf" ,"ilp"]
code = "sc"

num_shots = 20
lost_prob = np.linspace(0,1,21)

for i in [3,5]:

    fail_list = []
    _,_,H,xlogi, zlogi = surface_code(i)
    stabi = GF2(H)
    # _,_,logicals = sc.find_logical_op_basis(stabi, i)
    # xlogi=ta.to_gf2_tableau(logicals[0])
    # zlogi=ta.to_gf2_tableau(logicals[1])

    for method in m:


        for p in lost_prob:
            tele_rate, _, rt = compute_teleportation_rate(stabi, p, g=1, method = method, max_cache_size=num_shots, num_shots=num_shots)

            out["loss_prob"].append(p)
            out["distance"].append(i)
            out["runtime"].append(rt)
            out["teleportation_rate"].append(tele_rate)
            out["method"].append(method)

            output_path = data_directory / f"{time_str}_{code}_thres.csv"
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(out.keys())
                writer.writerows(zip(*out.values()))

#         fail_list.append(t)


#     fail_list = np.asarray(fail_list)
#     yerr = np.sqrt(fail_list * (1 - fail_list) / num_shots)

#     # plt.title(f"Threshold plot for {cha} channel")
#     plt.title(f"Threshold plot for rotated-surface code")
#     plt.errorbar(
#         lost_prob,
#         fail_list,
#         yerr=yerr,
#         fmt="o-",
#         label=f"Code is {i}x{i}"
#     )

# plt.xlabel("Loss Rate")
# plt.ylabel("Rate of teleportation")

# plt.legend()
# plt.grid()
# # plt.savefig(f"plots/rotated_surfacecode_threshold_5_heu_g{gg}_{num_shots}"+".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
# plt.show()
