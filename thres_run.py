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
from code_importer import rotated_surface_code, surface_code
from collections import defaultdict
from gspf_tests import test_gspf


def sample_lost_masks(num_qubits, p, num_shots = 1, exclude=None, rng=None):

    rng = np.random.default_rng(rng)

    # independent Bernoulli(p) loss per qubit
    lost_masks = rng.random(size = (num_shots, num_qubits)) < p

    if exclude is not None:
        lost_masks[:, exclude] = False

    # lost_qubits = np.flatnonzero(lost_masks)

    return lost_masks


ts = pd.Timestamp.now(tz="Europe/Stockholm")
date_str = ts.strftime("%Y-%m-%d")
time_str = ts.strftime("%H%M%S")
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_directory = base_dir / "threshold_folder" / f"{date_str}"
data_directory.mkdir(parents=True, exist_ok=True)



m = ["Heuristic" ,"g-SPF" ,"ILP"]
m = ["Heuristic" ,"g-SPF" ,"ILP"]
# m = ['Heuristic']
code = "sc"

num_shots = 5
gg = 1
lost_prob = np.linspace(0,1,0)
# print(list(lost_prob))
# lost_prob = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#             0.55, 0.6000000000000001, 0.65, 0.7000000000000001,
#             0.75, 0.8, 0.8500000000000001, 0.9, 0.9500000000000001, 1.0]

# lost_prob = [0.05]
ms = True
out_dict = defaultdict(list)
for i in [3,5, 7,9]:
    seen = set()
    cache = []
    fail_list = []
    _,_,H,xlogi, zlogi = surface_code(i)
    T = GF2(H)
    target_qubit = None
    num_qubits=T.shape[1]//2
    print(num_qubits)
    for p in lost_prob:
        support_list = []
        # hits = 0
        # solves = 0
        # success = 0
        # num_runs = 0
        lost_mask = sample_lost_masks(num_qubits, p, int(10*num_shots), exclude=target_qubit)

        for shot in range(num_shots):

            lq = np.flatnonzero(lost_mask[shot])
            lost_set = frozenset(int(c) for c in lq)

            if lost_set in seen:
                continue

            seen.add(lost_set)

            # hit = False
            # for supp in cache:
            #     if lost_set.isdisjoint(supp):
            #         hit = True
            #         break

            # if hit:
            #     # hits += 1
            #     # success += 1                      # success (a valid pattern survives)
            #     continue

            # solves += 1
            out_dict["Distance"].append(i)
            out_dict['Loss probability'].append(p)
            out_dict["Shot number"].append(shot)
            out_dict['lost_qubits'].append(lost_set)

            t1 = time.time()
            res_heu = generalised_spf_logical_heuristic(T, lq, target_qubit=target_qubit) # this assumes no meas have happened
            rt_heu = (time.time()-t1)
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
            supp = sc.pair_support(res_gspf["x"], res_gspf["z"], target_qubit, size=True)
            test = test_gspf(T, res_gspf["x"], res_gspf["z"], [], lq, g =gg)
            out_dict["g-SPF success"].append(res_gspf['success'])
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


            t3 = time.time()
            T = GF2(T)
            _,_,logicals = sc.find_logical_op_basis(T, num_qubits)
            xlogi = ta.to_gf2_tableau(logicals[0])
            zlogi = ta.to_gf2_tableau(logicals[1])
            res_ilp = gspf_ilp(T, xlogi, zlogi, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
            rt_ilp = (time.time()-t3)
            supp = sc.pair_support(res_ilp["x"], res_ilp["z"], target_qubit, size=True)
            test = test_gspf(T, res_ilp["x"], res_ilp["z"], [], lq, g =gg)
            out_dict["ILP success"].append(res_ilp['success'])
            out_dict["ILP verify"].append(test)
            out_dict['ILP runtime'].append(rt_ilp)
            out_dict["ILP support size"].append(supp)
            out_dict["ILP X logical"].append(ta.tableau2paulistring(res_ilp["x"]))
            out_dict["ILP Z logical"].append(ta.tableau2paulistring(res_ilp["z"]))
            a = ta.qubit_wise_commutation(res_ilp["x"], res_ilp["z"])
            if a is not None:
                out_dict["ILP g"].append(len(a))
            else:
                out_dict["ILP g"].append(a)


            supp = sc.pair_support(res_ilp["x"], res_ilp["z"], target_qubit)
            if supp is not None and supp not in cache:
                cache.append(supp)

            output_path = data_directory / f"{time_str}_{code}_thres_min_{ms}_{gg}.csv"
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(out_dict.keys())
                writer.writerows(zip(*out_dict.values()))
