#importlib.reload(test_spf)

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from galois import GF2
import tableau as ta

from gspf_ilp import generalised_spf_logical,generalised_spf_logical_heuristic
from spf_graphs import crazy_graph, sample_lost_nodes
from decoder_methods import create_graph_code


from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed


def run_p(args):
    p, g, stabi, xlogi, zlogi, previous_meas, numq, gg, num_shots = args

    success = 0

    for _ in range(num_shots):
        lq = sample_lost_nodes(g, p)
        lost_cols = [node - 1 for node in lq]

        lost_qubits = [0] * (2 * numq)
        for c in lost_cols:
            lost_qubits[c] = 1
            lost_qubits[c + numq] = 1

        res = generalised_spf_logical(
            stabi,
            xlogi,
            zlogi,
            previous_meas,
            lost_qubits,
            gg,
            target_qubit=numq - 1
        )

        success += res["success"]

    return success / num_shots


plt.figure()


lost_prob = np.linspace(0,1,21)
for i in [2, 3, 4, 5]:
    # cha = "hexagonal"
    cha = "crazy graph"
    # g = hexagonal_lattice(i,i)
    g = crazy_graph(i,i)
    g = nx.to_numpy_array(g, dtype = np.uint16)
    #numq = g.shape[0]-1
    xlogi, zlogi, stabi = create_graph_code(g)
    T = GF2(stabi)
    numq=T.shape[1]//2
    gg = 1
    previous_meas = []
    previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]


    fail_list = []
    fit = []
    num_shots = 2000
    args = [
        (p, g, stabi, xlogi, zlogi,
            previous_meas, numq, gg, num_shots)
        for p in lost_prob
    ]

    fail_list = Parallel(n_jobs=-1)(
        delayed(run_p)(arg) for arg in args
    ) #can we make them communicate somehow?

    fail_list = np.asarray(fail_list)
    yerr = np.sqrt(fail_list * (1 - fail_list) / num_shots)

    plt.title(f"Threshold plot for {cha} channel")
    plt.errorbar(
        lost_prob,
        fail_list,
        yerr=yerr,
        fmt="o-",
        label=f"Channel is {i}x{i}"
    )

    plt.xlabel("Loss Rate")
    plt.ylabel("Rate of teleportation")

plt.legend()
plt.grid()
# plt.savefig(f"{cha}_threshold_5"+".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
plt.show()


g = crazy_graph(4,4)
g = nx.to_numpy_array(g, dtype = np.uint16)
xlogi, zlogi, stabi = create_graph_code(g)
T = GF2(stabi)
numq = T.shape[1]//2
gg = 1
previous_meas = []
previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]

target = numq - 1          # output qubit O in tableau columns

# --- cache setup ---

def sample_lost_qubits(num_qubits, p, exclude=None, rng=None):

    rng = np.random.default_rng(rng)

    # independent Bernoulli(p) loss per qubit
    lost_mask = rng.random(num_qubits) < p

    # qubit exclude(o) is forced not-lost
    if exclude is not None:
        lost_mask[exclude] = False

    lost_qubits = np.flatnonzero(lost_mask)

    return lost_qubits, lost_mask

def compute_threshold(T,lost_prob:np.ndarray=np.linspace(0.1, 0.95, 10),\
                      target_qubit:int=None,max_cache_size:int=None,method:str='heuristic',num_shots:int=500):
    cache = []                  
    
    num_qubits=T.shape[1]//2
 
    teleportation_rate = []
   
    hits = 0
    solves = 0


    for p in lost_prob:
        
        success = 0

        for ele in range(num_shots):
            lq,_ = sample_lost_qubits(num_qubits, p,exclude=target_qubit)
             
            lost_set = set(int(c) for c in lq)

            # --- cache lookup: any stored pattern that avoids all lost qubits? ---
            hit = False

            for supp in cache:

                if lost_set.isdisjoint(supp):
                    hit = True
                    break

            if hit:
                hits += 1
                success += 1                      # success (a valid pattern survives)
                continue
        # -------------------------------------------------------------------

            solves += 1
            if method=='heuristic':
                res = generalised_spf_logical_heuristic(stabi,  lq, target_qubit=target)
            elif method=='ILP':
                res=generalised_spf_logical(stabi,  lq, target_qubit=target)

            else:
                raise ValueError(f"Method {method} not recognised")

            if res["success"]:
                success += 1
                # --- cache the found pattern (full-width x, z) ---
             
                supp = ta.pair_support(res["x"], res["z"], numq, target)
                if supp is not None and supp not in cache:
                    cache.append(supp)
                    if len(cache) > max_cache_size:
                        cache.pop(0)         # FIFO eviction
            # ------------------------------------------------

        teleportation_rate.append(success/num_shots)
    

    print(f"cache hits: {hits}, solves: {solves}, hit rate: {hits/(hits+solves):.3f}")

    return teleportation_rate,cache

"""fail_list = np.asarray(fail_list)
yerr = np.sqrt(fail_list * (1 - fail_list) / num_shots)

plt.figure()
plt.title("Threshold plot for crazy graph")
plt.errorbar(lost_prob, fail_list, yerr=yerr, fmt="o", label="g-SPF")
plt.plot(lost_prob, fit, label=r"Fit $(1-p^{4})^{4}$")
plt.legend()
plt.show()"""