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
from spf_graphs import crazy_graph, hexagonal_lattice, triangular_lattice
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

#----------------------------------------------------PARAMETERS----------------------------------------------------------------------------------#
ts = pd.Timestamp.now(tz="Europe/Stockholm")
date_str = ts.strftime("%Y-%m-%d")
time_str = ts.strftime("%H%M%S")
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_directory = base_dir / "threshold_folder" / f"{date_str}"
data_directory.mkdir(parents=True, exist_ok=True)
bb_tuples=[(6,6), (15,3), (9,6), (12,6),]#, (12,12)]
m = ["Heuristic" ,"g-SPF" ,"ILP"]
m = ["Heuristic" ,"g-SPF" ,"ILP"]
# m = ['Heuristic']
code = "hex"
code = "tri"
code = "crazy"

num_shots = 20
gg = 1
lost_prob = np.linspace(0,1,11)
# print(list(lost_prob))
# lost_prob = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5,
#             0.55, 0.6000000000000001, 0.65, 0.7000000000000001,
#             0.75, 0.8, 0.8500000000000001, 0.9, 0.9500000000000001, 1.0]

# lost_prob = [0.05]
#----------------------------------------------------PARAMETERS----------------------------------------------------------------------------------#



ms = True
spf_like = True
out_dict = defaultdict(list)

for i in [3]:
    print(i)
    cache = []
    fail_list = []

    if code == "hex":
        graph = hexagonal_lattice(i,i)
    elif code == "tri":
        graph = triangular_lattice(i,i)
    elif code == "crazy":
        graph = crazy_graph(i,i)

    xl, zl, H = sc.create_graph_code(graph, spf=spf_like)

    T = ta.to_gf2_tableau(H)
    num_qubits=T.shape[1]//2
    target_qubit = num_qubits-1
    k = num_qubits - T.shape[0]
    print(f"number of physical qubits: {num_qubits}")
    print(f"number of logical qubits: {k}")
    for p in lost_prob:
        print('p: ',p)
        support_list = []
        seen = set()

        # hits = 0
        # solves = 0
        # success = 0
        # num_runs = 0
        lost_mask = sample_lost_masks(num_qubits, p, int(num_shots), exclude=target_qubit)

        for shot in range(num_shots):


            # print("shot: ",shot)

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
            if code == "bb":
                out_dict["BB-type"].append(f"[[{num_qubits}, {k}]]")
            else:
                out_dict["Distance"].append(i)
            out_dict['Loss probability'].append(p)
            out_dict["Shot number"].append(shot)
            out_dict['lost_qubits'].append(lost_set)

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
            T = ta.to_gf2_tableau(T) #is this necessary?
            _,xlogi,zlogi = sc.initialise_logical_basis(T) #needs to call this function for more than one qubits, otherwise
            # it is not guaranteed that xlogi and zlogic anti-commute

            res_ilp = gspf_ilp(T, xlogi, zlogi, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
            rt_ilp = (time.time()-t3)
            # print("ilp",rt_ilp)
            # print(res_ilp)
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
