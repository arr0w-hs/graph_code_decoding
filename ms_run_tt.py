import time
import os
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from galois import GF2
import pandas as pd
import ast
from generalised_spf import generalised_spf_logical, gspf_ilp ,generalised_spf_logical_heuristic
import stabiliser_code as sc
import tableau as ta
from code_importer import rotated_surface_code, surface_code,bb_tableau
from collections import defaultdict
from gspf_tests import test_gspf
import random
from spf_graphs import crazy_graph, hexagonal_lattice, triangular_lattice

time.sleep(random.uniform(1, 10))
base_dir = Path(__file__).resolve().parent
file = base_dir/"ms_tt_crazy.csv"

with open(file, "r") as f:
    data = list(csv.reader(f))
nlist = [int(ele) for ele in data[0]]
print(nlist)
llist = data[1]
for i in data[1]:
    i = ast.literal_eval(i)

lq_list = []
for i, n in enumerate(nlist):
    lqq = ast.literal_eval(data[1][i])
    lq_list.append(lqq)
    print(len(lqq))
    # unique_elements = [list(x) for x in dict.fromkeys(map(tuple, lq_list))]
    # print(len(unique_elements))

# print(len(lq_list))


gg = 1
ms = True
code = 'crazy'
out_dict = defaultdict(list)

for code_size, lql in zip(nlist, lq_list):
    # if code_size != 4:
    #     continue
    # print
    if code == "hex":
        graph = hexagonal_lattice(code_size,code_size)
    elif code == "tri":
        graph = triangular_lattice(code_size,code_size)
    elif code == "crazy":
        graph = crazy_graph(code_size,code_size)

    xl, zl, H = sc.create_graph_code(graph, spf=True)

    T = ta.to_gf2_tableau(H)
    num_qubits=T.shape[1]//2
    target_qubit = num_qubits-1

    # lq_list = ast.literal_eval(data[1][code_size-2])
    # print(len(lql))

    for lq in lql:

        lq = np.array(list(lq), dtype=int)
        # print(lq)

        # t1 = time.time()
        # res_heu = generalised_spf_logical_heuristic(T, lq, gg,target_qubit=target_qubit) # this assumes no meas have happened
        # rt_heu = (time.time()-t1)
        # # print("heu", rt_heu)
        # # print(res_heu)
        # supp = sc.pair_support(res_heu["x"], res_heu["z"], target_qubit, size=True)
        # test = test_gspf(T, res_heu["x"], res_heu["z"], [], lq, g =gg)
        # out_dict["Heuristic success"].append(res_heu['success'])
        # out_dict["Heuristic verify"].append(test)
        # out_dict['Heuristic runtime'].append(rt_heu)
        # out_dict["Heuristic support size"].append(supp)
        # out_dict["Heuristic X logical"].append(ta.tableau2paulistring(res_heu["x"]))
        # out_dict["Heuristic Z logical"].append(ta.tableau2paulistring(res_heu["z"]))
        # a = ta.qubit_wise_commutation(res_heu["x"], res_heu["z"])
        # if a is not None:
        #     out_dict["Heuristic g"].append(len(a))
        # else:
        #     out_dict["Heuristic g"].append(a)


        t2 = time.time()
        res_gspf, ea_flag, rt_early = generalised_spf_logical(T, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
        rt_gspf = (time.time()-t2)
        # print("gsf",rt_gspf)
        # print(res_gspf)
        out_dict['lost_qubits'].append(lq)
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

        output_path = base_dir / f"ms_run_thres_min_{ms}_{gg}_{target_qubit}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            # print('writing')
            writer = csv.writer(f)
            writer.writerow(out_dict.keys())
            writer.writerows(zip(*out_dict.values()))

        # print(out_dict)
