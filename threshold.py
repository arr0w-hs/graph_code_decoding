import time
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from galois import GF2
import tableau as ta
from generalised_spf import generalised_spf_logical, gspf_ilp ,generalised_spf_logical_heuristic
from spf_graphs import crazy_graph, sample_lost_nodes
from decoder_methods import create_graph_code

import stabiliser_code as sc
from galois import GF2
import codetable_sampler as cs
from code_importer import rotated_surface_code, surface_code

from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed
plt.rcParams.update({'font.size': 14})


def run_p(args):
    p, stabi, xlogi, zlogi, previous_meas, numq, gg, num_shots = args

    success = 0

    for _ in range(num_shots):
        # lq = sample_lost_nodes(g, p)
        # lq = [node - 1 for node in lq]
        # lost_qubits = [0] * (2 * numq)
        # for c in lost_cols:
        #     lost_qubits[c] = 1
        #     lost_qubits[c + numq] = 1



        rng = random.Random()
        # lq =  [node for node in range(numq) if rng.random() < p and node != numq-1]
        lq =  [node for node in range(numq) if rng.random() < p]



        res = generalised_spf_logical(
            stabi,
            previous_meas,
            lq,
            gg,
            target_qubit=None,
            minimise_support=False
        )

        # res = generalised_spf_logical_heuristic(
        #     stabi,
        #     lq,
        #     target_qubit=None,
        # )


        # res = generalised_spf_logical_heuristic(
        #     stabi,
        #     lq,
        #     target_qubit=numq - 1,
        # )

        success += res["success"]

    return success / num_shots



def sample_lost_masks(num_qubits, p, num_shots = 1, exclude=None, rng=None):

    rng = np.random.default_rng(rng)

    # independent Bernoulli(p) loss per qubit
    lost_masks = rng.random(size = (num_shots, num_qubits)) < p

    if exclude is not None:
        lost_masks[:, exclude] = False

    # lost_qubits = np.flatnonzero(lost_masks)

    return lost_masks


def tele_rate_plot(gg, num_shots, save=False):

    lost_prob = np.linspace(0,1,21)
    for i in [3,5,7]:
        # cha = "hexagonal"
        # cha = "crazy graph"
        # g = hexagonal_lattice(i,i)
        # g = crazy_graph(i,i)
        # g = nx.to_numpy_array(g, dtype = np.uint16)
        # xlogi, zlogi, stabi = create_graph_code(g)

        # _,_,H,xlogi, zlogi = rotated_surface_code(i)
        _,_,H,xlogi, zlogi = surface_code(i)
        stabi = GF2(H)
        _,_,logicals = sc.find_logical_op_basis(stabi, i)
        xlogi=ta.to_gf2_tableau(logicals[0])
        zlogi=ta.to_gf2_tableau(logicals[1])

        numq=stabi.shape[1]//2
        print(numq)
        previous_meas = []
        # previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]


        fail_list = []
        fit = []
        args = [
            (p, stabi, xlogi, zlogi,
                previous_meas, numq, gg, num_shots)
            for p in lost_prob
        ]

        # i cannot import surface code but here is how you implement this with the functin abive
        #args=[(T,p,None,gg) for p in lost_prob] #not sure if this works
        #fail_list,_=Parallel(n_jobs=-1)(
            #delayed(compute_teleportation_rate)(arg) for arg in args
        #) #can we make them communicate somehow?

        fail_list = Parallel(n_jobs=-1)(
            delayed(run_p)(arg) for arg in args
        ) #can we make them communicate somehow?

        fail_list = np.asarray(fail_list)
        yerr = np.sqrt(fail_list * (1 - fail_list) / num_shots)

        # plt.title(f"Threshold plot for {cha} channel")
        plt.title(f"Threshold plot for surface code")
        plt.errorbar(
            lost_prob,
            fail_list,
            yerr=yerr,
            fmt="o-",
            label=f"Code is of distance {i}"
        )

    plt.xlabel("Loss Rate")
    plt.ylabel("Rate of teleportation")
    plt.grid()
    plt.legend()

    if save:
        plt.savefig(f"plots/surfacecode_threshold_g{gg}_{num_shots}"+".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
    plt.show()

    return


def compute_teleportation_rate(T, p, target_qubit = None, g = 1,
                    max_cache_size = 2000,
                    method = "gspf",
                    num_shots = 2000,
                    minimise_support = True):

    cache = []
    num_qubits=T.shape[1]//2
    teleportation_rate = []
    support_list = []
    hits = 0
    solves = 0
    success = 0
    avg_rt = 0
    num_runs = 0
    lost_mask = sample_lost_masks(num_qubits, p, num_shots, exclude=target_qubit)

    for ele in range(num_shots):

        lq = np.flatnonzero(lost_mask[ele])
        lost_set = set(int(c) for c in lq)


        hit = False
        for supp in cache:
            if lost_set.isdisjoint(supp):
                hit = True
                break

        if hit:
            hits += 1
            success += 1                      # success (a valid pattern survives)
            continue


        solves += 1
        if method == "Heuristic":
            t1 = time.time()
            res = generalised_spf_logical_heuristic(T, lq, target_qubit=target_qubit) # this assumes no meas have happened
            rt = (time.time()-t1)

        elif method == "g-SPF":
            meas=[]
            t2 = time.time()
            res = generalised_spf_logical(T, meas, lq, g, target_qubit=target_qubit, minimise_support=minimise_support)
            rt = (time.time()-t2)

        elif method == "ILP":
            meas = []
            t3 = time.time()
            T = GF2(T)
            _,_,logicals = sc.find_logical_op_basis(T, num_qubits)
            xlogi = ta.to_gf2_tableau(logicals[0])
            zlogi = ta.to_gf2_tableau(logicals[1])
            res = gspf_ilp(T, xlogi, zlogi, meas, lq, g, target_qubit=target_qubit, minimise_support=minimise_support)
            rt = (time.time()-t3)
        else:
            raise AssertionError ("wrong method")

        avg_rt += rt
        if res["success"]:
            success += 1

            # --- cache the found pattern (full-width x, z) ---

            supp = sc.pair_support(res["x"], res["z"], target_qubit)
            support_list.append(len(supp))

            if supp is not None and supp not in cache:
                cache.append(supp)
                if len(cache) > max_cache_size:
                    cache.pop(0)         # FIFO eviction
            # ------------------------------------------------

    teleportation_rate=success/num_shots
    avg_rt /= solves

    if len(support_list) == 0:
        avg_supp = 0
    else:
        avg_supp = np.mean(support_list)
    # print(f"cache hits: {hits}, solves: {solves}, hit rate: {hits/(hits+solves):.3f}")

    return teleportation_rate, cache, avg_rt, avg_supp


if __name__ == "__main__":
    print()
#     from pathlib import Path

#     plt.figure()
#     base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
#     data_directory = base_dir / "er_results" / f"{date_str}_er"
#     data_directory.mkdir(parents=True, exist_ok=True)
#     # tele_rate_plot(gg=1, num_shots=2000, save=True)

#     # # Example: [[34,4,10]] over GF(3^2), so q=9 and p=3
#     # # html = cs.fetch_codetables_qecc(n=9, k=1)

#     # # Hx, Hz, H = cs.extract_stabilizer_matrix(html, n=9)

#     # # print("Hx shape:", Hx.shape)
#     # # print("Hz shape:", Hz.shape)
#     # # print("H shape :", H.shape)
#     # # print("commutes:", cs.check_stabilizer_commutes(Hx, Hz))

#     # # H = GF2(H)

#     # # a = sc.d_logical_op_basis(H, 9, CSS=True)
#     # # print(a)

#     num_shots = 2
#     lost_prob = np.linspace(0,1,9)
#     for i in [3]:

#         fail_list = []
#         _,_,H,xlogi, zlogi = surface_code(i)
#         stabi = GF2(H)
#         _,_,logicals = sc.find_logical_op_basis(stabi, i)
#         xlogi=ta.to_gf2_tableau(logicals[0])
#         zlogi=ta.to_gf2_tableau(logicals[1])

#         for p in lost_prob:
#             out = compute_teleportation_rate(stabi, p, g=1, max_cache_size=num_shots, num_shots=num_shots)
#             print(out)

#             output_path = data_directory / f"{time_str}_{n}_output.csv"
#             with output_path.open("w", newline="", encoding="utf-8") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(out_dict.keys())
#                 writer.writerows(zip(*out_dict.values()))
#     #         fail_list.append(t)


#     #     fail_list = np.asarray(fail_list)
#     #     yerr = np.sqrt(fail_list * (1 - fail_list) / num_shots)

#     #     # plt.title(f"Threshold plot for {cha} channel")
#     #     plt.title(f"Threshold plot for rotated-surface code")
#     #     plt.errorbar(
#     #         lost_prob,
#     #         fail_list,
#     #         yerr=yerr,
#     #         fmt="o-",
#     #         label=f"Code is {i}x{i}"
#     #     )

#     # plt.xlabel("Loss Rate")
#     # plt.ylabel("Rate of teleportation")

#     # plt.legend()
#     # plt.grid()
#     # # plt.savefig(f"plots/rotated_surfacecode_threshold_5_heu_g{gg}_{num_shots}"+".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
#     # plt.show()
