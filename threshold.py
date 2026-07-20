#importlib.reload(test_spf)
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from galois import GF2
import tableau as ta

from gspf_ilp import generalised_spf_logical, generalised_spf_logical_old,generalised_spf_logical_heuristic
from spf_graphs import crazy_graph, sample_lost_nodes
from decoder_methods import create_graph_code


from multiprocessing import Pool, cpu_count
from joblib import Parallel, delayed


def run_p(args):
    p, stabi, xlogi, zlogi, previous_meas, numq, gg, num_shots = args

    success = 0

    for _ in range(num_shots):
        # lq = sample_lost_nodes(g, p)
        # lost_cols = [node - 1 for node in lq]


        rng = random.Random()
        eligible_nodes = [ node for node in range(numq)]
        lq =  [node for node in eligible_nodes if rng.random() < p]


        # lost_qubits = [0] * (2 * numq)
        # for c in lost_cols:
        #     lost_qubits[c] = 1
        #     lost_qubits[c + numq] = 1

        # res = generalised_spf_logical_old(
        #     stabi,
        #     xlogi,
        #     zlogi,
        #     previous_meas,
        #     lq,
        #     gg,
        #     target_qubit=numq - 1
        # )

        res = generalised_spf_logical(
            stabi,
            previous_meas,
            lq,
            gg,
            target_qubit=numq - 1,
            minimise_support=False
        )


        # res = generalised_spf_logical_heuristic(
        #     stabi,
        #     lq,
        #     target_qubit=numq - 1,
        # )

        success += res["success"]

    return success / num_shots



def sample_lost_qubits(num_qubits, p, exclude=None, rng=None):

    rng = np.random.default_rng(rng)

    # independent Bernoulli(p) loss per qubit
    lost_mask = rng.random(num_qubits) < p

    if exclude is not None:
        lost_mask[exclude] = False

    lost_qubits = np.flatnonzero(lost_mask)

    return lost_qubits, lost_mask


def compute_teleportation_rate(T,p:float,\
                      target_qubit:int=None,g:int=None,max_cache_size:int=1000,method:str='deterministic',num_shots:int=2000):
    cache = []

    num_qubits=T.shape[1]//2

    teleportation_rate = []

    hits = 0
    solves = 0

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
            assert g is None, 'for heuristic method g cannot be specified'
            res = generalised_spf_logical_heuristic(stabi,  lq, target_qubit=target_qubit)
        elif method=='ILP':
            meas=[]
            res=generalised_spf_logical(stabi, meas, lq,g, target_qubit=target_qubit) #this assumes no meas have happend

        else:
            raise ValueError(f"Method {method} not recognised")

        if res["success"]:
            success += 1
            # --- cache the found pattern (full-width x, z) ---

            supp = ta.pair_support(res["x"], res["z"], numq, target_qubit)
            if supp is not None and supp not in cache:
                cache.append(supp)
                if len(cache) > max_cache_size:
                    cache.pop(0)         # FIFO eviction
            # ------------------------------------------------

    teleportation_rate=success/num_shots

    print(f"cache hits: {hits}, solves: {solves}, hit rate: {hits/(hits+solves):.3f}")

    return teleportation_rate,cache


if __name__ == "__main__":
    plt.figure()
    from stabiliser_code import find_logical_op_basis
    from galois import GF2
    import codetable_sampler as cs
    from surface_code import rotated_surface_code

    # Example: [[34,4,10]] over GF(3^2), so q=9 and p=3
    # html = cs.fetch_codetables_qecc(n=9, k=1)

    # Hx, Hz, H = cs.extract_stabilizer_matrix(html, n=9)

    # print("Hx shape:", Hx.shape)
    # print("Hz shape:", Hz.shape)
    # print("H shape :", H.shape)
    # print("commutes:", cs.check_stabilizer_commutes(Hx, Hz))

    # H = GF2(H)

    # a = find_logical_op_basis(H, 9, CSS=True)
    # print(a)

    num_shots = 1000
    lost_prob = np.linspace(0,1,21)
    for i in [3, 5, 7]:
        # cha = "hexagonal"
        # cha = "crazy graph"
        # g = hexagonal_lattice(i,i)
        # g = crazy_graph(i,i)
        # g = nx.to_numpy_array(g, dtype = np.uint16)
        #numq = g.shape[0]-1
        _,_,H,xlogi, zlogi = rotated_surface_code(i)
        stabi = GF2(H)
        _,_,logicals = find_logical_op_basis(stabi, i)
        xlogi=ta.to_gf2_tableau(logicals[0])
        zlogi=ta.to_gf2_tableau(logicals[1])

        # xlogi, zlogi, stabi = create_graph_code(g)

        T = ta.to_gf2_tableau(T)
        numq=T.shape[1]//2
        gg = 1
        previous_meas = []
        previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]


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
        plt.title(f"Threshold plot for rotate-surface code")
        plt.errorbar(
            lost_prob,
            fail_list,
            yerr=yerr,
            fmt="o-",
            label=f"Code is {i}x{i}"
        )

        plt.xlabel("Loss Rate")
        plt.ylabel("Rate of teleportation")

    plt.legend()
    plt.grid()
    # plt.savefig(f"{cha}_threshold_5"+".pdf", dpi=800, format="pdf", bbox_inches = 'tight')
    plt.show()


    # g = crazy_graph(4,4)
    # g = nx.to_numpy_array(g, dtype = np.uint16)
    # xlogi, zlogi, stabi = create_graph_code(g)
    # T = GF2(stabi)
    # numq = T.shape[1]//2
    # gg = 1
    # previous_meas = []
    # previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]

    # target = numq - 1          # output qubit O in tableau columns

    # fail_list = np.asarray(fail_list)
    # yerr = np.sqrt(fail_list * (1 - fail_list) / num_shots)

    # plt.figure()
    # plt.title("Threshold plot for crazy graph")
    # plt.errorbar(lost_prob, fail_list, yerr=yerr, fmt="o", label="g-SPF")
    # plt.plot(lost_prob, fit, label=r"Fit $(1-p^{4})^{4}$")
    # plt.legend()
    # plt.show()