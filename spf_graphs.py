import random
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from galois import GF2
import tableau as ta

from gspf_ilp import generalised_spf_logical

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

def square_lattice(width, length, output_node=True):
    graph = nx.Graph()

    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]

    # Input to first layer
    graph.add_edges_from((0, node) for node in layers[0])

    # Vertical edges inside each layer
    for layer in layers:
        graph.add_edges_from(zip(layer, layer[1:]))

    # Horizontal edges between layers
    for left_layer, right_layer in zip(layers, layers[1:]):
        graph.add_edges_from(zip(left_layer, right_layer))

    if output_node:
        output = width * length + 1
        graph.add_edges_from(
            (node, output) for node in layers[-1]
        )

    return graph

def hexagonal_lattice(width, length, output_node=True):
    graph = nx.Graph()

    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]

    # Input to first layer
    graph.add_edges_from((0, node) for node in layers[0])

    # Brickwork edges inside each layer
    for layer_number, layer in enumerate(layers, start=1):
        for row in range(width - 1):
            if (row + layer_number) % 2 == 1:
                graph.add_edge(layer[row], layer[row + 1])

    # Corresponding nodes between consecutive layers
    for left_layer, right_layer in zip(layers, layers[1:]):
        graph.add_edges_from(zip(left_layer, right_layer))

    if output_node:
        output = width * length + 1
        graph.add_edges_from(
            (node, output) for node in layers[-1]
        )

    return graph

def triangular_lattice(
    width,
    length,
    alternate_diagonals=True,
    output_node=True,
):
    graph = nx.Graph()

    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]

    # Input to first layer
    graph.add_edges_from((0, node) for node in layers[0])

    # Edges inside each layer
    for layer in layers:
        graph.add_edges_from(zip(layer, layer[1:]))

    for layer_number, (left_layer, right_layer) in enumerate(
        zip(layers, layers[1:]),
        start=2,
    ):
        # Straight edges
        graph.add_edges_from(zip(left_layer, right_layer))

        # Diagonal edges
        if alternate_diagonals and layer_number % 2 == 1:
            graph.add_edges_from(
                zip(left_layer[:-1], right_layer[1:])
            )
        else:
            graph.add_edges_from(
                zip(left_layer[1:], right_layer[:-1])
            )

    if output_node:
        output = width * length + 1
        graph.add_edges_from(
            (node, output) for node in layers[-1]
        )

    return graph

def tree_to_tree_graph(branches, depth):
    graph = nx.Graph()

    next_node = 1
    current_layer = [0]

    # First tree: expand outward
    for _ in range(depth):
        new_layer = []

        for parent in current_layer:
            children = list(
                range(next_node, next_node + branches)
            )
            next_node += branches

            graph.add_edges_from(
                (parent, child) for child in children
            )

            new_layer.extend(children)

        current_layer = new_layer

    # Second tree: merge inward
    for _ in range(depth):
        new_layer = []

        for start in range(0, len(current_layer), branches):
            children = current_layer[start:start + branches]
            parent = next_node
            next_node += 1

            graph.add_edges_from(
                (child, parent) for child in children
            )

            new_layer.append(parent)

        current_layer = new_layer

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

    return [node for node in eligible_nodes if rng.random() < loss_probability]



def create_graph_code_based_on_spf(in_adj : np.array, code_node : int = 0):
    """create a graph code from an input graph
    using the 0th node as the input node"""

    num_nodes = in_adj.shape[0]
    identity = np.identity(num_nodes, dtype = np.uint16)
    gen = np.hstack([identity, in_adj])
    zlogi = gen[code_node].copy()

    neigh = [i for i, ele in enumerate(in_adj[code_node]) if ele ==1]
    assert(len(neigh)>0)

    xlogi = np.zeros(2*num_nodes, dtype = np.uint16)
    xlogi[num_nodes+code_node] = 1


    rows_to_remove = [code_node]
    gen = np.delete(gen, rows_to_remove, axis=0)
    # gen = np.delete(gen, [code_node, code_node+num_nodes], axis=1)
    # xlogi = np.delete(xlogi, [code_node, code_node+num_nodes])#, axis=1)
    # zlogi = np.delete(zlogi, [code_node, code_node+num_nodes])#, axis=1)


    return xlogi, zlogi, gen

if __name__ == "__main__":
    from stabiliser_code import create_graph_code


    plt.figure()

    for i in [2,3,4]:
        g = triangular_lattice(i,i)
        g = nx.to_numpy_array(g, dtype = np.uint16)
        numq = g.shape[0]-1
        xlogi, zlogi, stabi = create_graph_code(g)
        gg = 1
        previous_meas = []
        previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]
        T = GF2(stabi)

        fail_list = []
        fit = []
        num_shots = 5_000

        lost_prob = np.linspace(0,1,21)
        for p in lost_prob:
            # print(p)
            fail = 0

            for ele in range(num_shots):
                lq = sample_lost_nodes(g, p)
                lost_qubits = [0]*2*numq
                for ele in lq:
                    lost_qubits[ele] = 1
                    lost_qubits[ele+numq] = 1

                res = generalised_spf_logical(stabi, xlogi, zlogi, previous_meas, lost_qubits, gg, target_qubit=numq)
                if res["success"]:
                    fail += 1

            fail_list.append(fail/num_shots)
            # fit.append((1-p**i)**i)

        fail_list = np.asarray(fail_list)
        yerr =  np.sqrt(fail_list * (1 - fail_list) / num_shots)


        plt.title("Threshold plot for traingular lattice")
        # plt.plot(lost_prob, fail_list)
        plt.errorbar(lost_prob, fail_list, yerr=yerr, fmt = "o-", label = f"channel is {i}x{i}")
        # plt.plot(lost_prob, fail_list, label = r"Fit $(1-p^{4})^{4}$")
        # plt.plot(lost_prob, fail_list, label = f"channel is {i}x{i}")
        plt.legend()
    plt.show()