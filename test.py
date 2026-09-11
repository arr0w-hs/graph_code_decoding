import time
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from galois import GF2
import pandas as pd
# from threshold import compute_teleportation_rate
from generalised_spf import generalised_spf_logical, gspf_ilp ,generalised_spf_logical_heuristic
import stabiliser_code as sc
import tableau as ta
from code_importer import rotated_surface_code, surface_code,bb_tableau
from collections import defaultdict
from gspf_tests import test_gspf


import time
import random
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from galois import GF2
import pandas as pd
# from threshold import compute_teleportation_rate
from generalised_spf import generalised_spf_logical, gspf_ilp ,generalised_spf_logical_heuristic
import stabiliser_code as sc
import tableau as ta
from spf_graphs import crazy_graph, hexagonal_lattice, triangular_lattice
from collections import defaultdict
from gspf_tests import test_gspf
import argparse
from collections import Counter


time.sleep(random.uniform(1, 1))


def sample_lost_masks(num_qubits, p, num_shots = 1, exclude=None, rng=None):
    rng = np.random.default_rng(rng)
    # independent Bernoulli(p) loss per qubit
    lost_masks = rng.random(size = (num_shots, num_qubits)) < p

    for qu in exclude:
        if qu is not None:
            lost_masks[:, qu] = False
    # lost_qubits = np.flatnonzero(lost_masks)
    return lost_masks


parser = argparse.ArgumentParser()

parser.add_argument("--size", type=int, default=3)
parser.add_argument(
    "--code",
    type=str,
    choices=["rsc", "bb", "sc"],
    default="rsc",
)
parser.add_argument(
    "--p_add",
    type=float,
    default=0.0,
    help="Add p to each loss probability",
)
parser.add_argument(
    "--target",
    type=int,
    default=None,
    help="specify target qubit",
)

parser.add_argument(
    "--num_shots",
    type=int,
    default=100,)

args = parser.parse_args()

num_shots = args.num_shots
code = args.code
code_size = args.size
p_add = args.p_add
target_qubit = args.target


ts = pd.Timestamp.now(tz="Europe/Stockholm")
date_str = ts.strftime("%Y-%m-%d")
time_str = ts.strftime("%H%M%S")
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_directory = base_dir / "threshold_folder" / f"{date_str}"
data_directory.mkdir(parents=True, exist_ok=True)

bb_tuples=[(6,6), (15,3), (9,6), (12,6),]#, (12,12)]
gg = 1
lost_prob = np.linspace(0,1,11)
lost_prob = lost_prob+p_add

ms = True
out_dict = defaultdict(list)
cache = []
fail_list = []

if code == "bb":
    _,_,H,_,_ = bb_tableau(bb_tuples[code_size])
elif code == "rsc":
    _,_,H,_,_ = rotated_surface_code(code_size)
elif code == "sc":
    _,_,H,_,_ = surface_code(code_size)

T = ta.to_gf2_tableau(H)
num_qubits=T.shape[1]//2
k = num_qubits - T.shape[0]
print(f"number of physical qubits: {num_qubits}")
print(f"number of logical qubits: {k}")

for p in lost_prob:
    # print('p: ',p)
    if p>1:
        continue
    support_list = []
    seen = set()

    config_counts = Counter()

    lost_mask = sample_lost_masks(num_qubits, p, int(num_shots), exclude=[target_qubit])

    for shot in range(num_shots):
        lq = np.flatnonzero(lost_mask[shot])
        lq = [int(q) for q in lq]
        lost_qubits = frozenset(lq)
        config_counts[lost_qubits] += 1

    for lq, count in config_counts.items():

        lq = np.array(list(lq), dtype=int)

        if code == "bb":
            out_dict["BB-type"].append(f"[[{num_qubits}, {k}]]")
        else:
            out_dict["Distance"].append(code_size)
        out_dict['Loss probability'].append(p)
        # out_dict["Shot number"].append(shot)
        out_dict['lost_qubits'].append(lq)
        out_dict["config_count"].append(count)

        t1 = time.time()
        res_heu = generalised_spf_logical_heuristic(T, lq, gg,target_qubit=target_qubit) # this assumes no meas have happened
        rt_heu = (time.time()-t1)
        # print("heu", rt_heu)
        # print(res_heu)
        supp = sc.pair_support(res_heu["x"], res_heu["z"], target_qubit, size=True)
        test = test_gspf(T, res_heu["x"], res_heu["z"], [], lq, g =gg)
        out_dict["Heuristic success"].append(res_heu['success'])
        out_dict["Heuristic verify"].append(test)
        out_dict['Heuristic runtime'].append(rt_heu)
        out_dict["Heuristic support size"].append(supp)
        out_dict["Heuristic X logical"].append(ta.tableau2paulistring(res_heu["x"]))
        out_dict["Heuristic Z logical"].append(ta.tableau2paulistring(res_heu["z"]))
        a = ta.qubit_wise_commutation(res_heu["x"], res_heu["z"])
        if a is not None:
            out_dict["Heuristic g"].append(len(a))
        else:
            out_dict["Heuristic g"].append(a)


        t2 = time.time()
        res_gspf, ea_flag, rt_early = generalised_spf_logical(T, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
        rt_gspf = (time.time()-t2)
        # print("gsf",rt_gspf)
        # print(res_gspf)
        supp = sc.pair_support(res_gspf["x"], res_gspf["z"], target_qubit, size=True)
        test = test_gspf(T, res_gspf["x"], res_gspf["z"], [], lq, g =gg)
        out_dict["g-SPF success"].append(res_gspf['success'])
        out_dict["g-SPF status"].append(res_gspf['status'])
        out_dict["g-SPF verify"].append(test)
        out_dict['g-SPF runtime'].append(rt_gspf)
        out_dict["g-SPF support size"].append(supp)
        out_dict["g-SPF X logical"].append(ta.tableau2paulistring(res_gspf["x"]))
        out_dict["g-SPF Z logical"].append(ta.tableau2paulistring(res_gspf["z"]))
        out_dict["early flag"].append(ea_flag)
        out_dict["early runtime"].append(rt_early)
        a = ta.qubit_wise_commutation(res_gspf["x"], res_gspf["z"])
        if a is not None:
            out_dict["g-SPF g"].append(len(a))
        else:
            out_dict["g-SPF g"].append(a)



        supp = sc.pair_support(res_gspf["x"], res_gspf["z"], target_qubit)
        if supp is not None and supp not in cache:
            cache.append(supp)

        output_path = data_directory / f"{time_str}_{code}_thres_min_{ms}_{gg}_{target_qubit}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            # print('writing')
            writer = csv.writer(f)
            writer.writerow(out_dict.keys())
            writer.writerows(zip(*out_dict.values()))

        # print(out_dict)
