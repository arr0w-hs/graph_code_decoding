import random
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from galois import GF2
import tableau as ta
from ortools.sat.python import cp_model

from gspf_ilp import create_graph_code, generalised_spf_logical

def crazy_graph(width, length, output_node=True):
    graph = nx.Graph()

    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]

    graph.add_edges_from((0, node) for node in layers[0])

    for left_layer, right_layer in zip(layers, layers[1:]):
        graph.add_edges_from(
            itertools.product(left_layer, right_layer)
        )

    if output_node:
        output = width * length + 1
        graph.add_edges_from(
            (node, output) for node in layers[-1]
        )

    return graph

def sample_lost_nodes(in_adj : np.ndarray , loss_probability):

    """
    Sample each node independently with probability p,
    excluding node 0 and the last-numbered node.
    """

    rng = random.Random()
    last_node = in_adj.shape[0]-1

    eligible_nodes = [
        node
        for node in range(in_adj.shape[0])
        if node != 0 and node != last_node
    ]

    return [
        node
        for node in eligible_nodes
        if rng.random() < loss_probability
    ]


if __name__ == "__main__":
    print()
    # g = crazy_graph(4,4)
    # g = nx.to_numpy_array(g, dtype = np.uint16)
    # numq = g.shape[0]-1
    # xlogi, zlogi, stabi = create_graph_code(g)
    # gg = 1
    # previous_meas = []
    # previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]
    # T = GF2(stabi)

    # fail_list = []
    # fit = []
    # num_shots = 5_00

    # lost_prob = np.linspace(0,1,10)
    # for p in lost_prob:
    #     print(p)
    #     fail = 0

    #     for ele in range(num_shots):
    #         lq = sample_lost_nodes(g, p)
    #         lost_qubits = [0]*2*numq
    #         for ele in lq:
    #             lost_qubits[ele] = 1
    #             lost_qubits[ele+numq] = 1

    #         res = generalised_spf_logical(stabi, xlogi, zlogi, previous_meas, lost_qubits, gg, target_qubit=numq)
    #         if res["success"]:
    #             fail += 1

    #     fail_list.append(fail/num_shots)
    #     fit.append((1-p**4)**4)

    # fail_list = np.asarray(fail_list)
    # yerr =  np.sqrt(fail_list * (1 - fail_list) / num_shots)

    # plt.figure()
    # # plt.plot(lost_prob, fail_list)
    # plt.errorbar(lost_prob, fail_list, yerr=yerr, fmt = "o", label = "g-SPF")
    # plt.plot(lost_prob, fit, label = r"Fit $(1-p^{4})^{4}$")
    # plt.legend()
    # plt.show()