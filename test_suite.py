
from galois import GF2
import numpy as np
import networkx as nx
import tableau as ta
import stabiliser_code as sc


def stabiliser_logical_commutation_check(stab_tableau : list[list], logical):

    if isinstance(stab_tableau, list):
        print("Number of stabilisers:", len(stab_tableau))
        if len(stab_tableau) == 0:
            print("Error: no stabilisers found. Test failed.")
            return
        print("Number of qubits:", len(stab_tableau[0])//2)
    elif isinstance(stab_tableau, GF2):
        print("Number of stabilisers:", stab_tableau.shape[0])
        if stab_tableau.shape[0] == 0:
            print("Error: no stabilisers found. Test failed.")
            return
        print("Number of qubits:", stab_tableau.shape[1]//2)
    elif isinstance(stab_tableau, np.ndarray):
        print("Number of stabilisers:", stab_tableau.shape[0])
        if stab_tableau.shape[0] == 0:
            print("Error: no stabilisers found. Test failed.")
            return
        print("Number of qubits:", stab_tableau.shape[1]//2)



    for stab in stab_tableau:

        if not ta.commutation_check(stab, logical):
            print("Error: Logical does not commute with the following stabiliser:x")
            print(ta.paulistring2tableau(stab))
    print("Stabiliser logical commutation test complete.", "\n")
    return

def x_and_z_logical_anticommutation(x_logi, z_logi):

    print("X logical =", ta.tableau2paulistring(x_logi))
    print("Z logical =", ta.tableau2paulistring(z_logi))

    res = ta.commutation_check(x_logi, z_logi)
    if res:
        print("Error: X and Z logical commute", "\n")
    else:

        res1 = ta.qubit_wise_commutation(x_logi, z_logi)
        print("X and Z logical anticommute on the following qubits", res1)
        print("The number of anticommutations =", len(res1), "\n")

    return

def lost_qubit_logical_overlap(logical, lost_qubits):

    num_qubits = len(logical)//2
    error = []
    for qubit in lost_qubits:

        xpart = bool(logical[qubit])
        zpart = bool(logical[num_qubits+qubit])
        total = xpart ^ zpart

        if total:
            error.append(qubit)

    if len(error)>0:
        print("Logical and lost qubits have overlaps at:", error, "\n")
    else:
        print("Lost-qubit logical test passed", "\n")

    return

def logical_commute_with_measurement(logical, measurement):

    if ta.commutation_check(logical, measurement):
        print("Logical commutes with measurement")
    else:
        print("Logical does not commute with the measurements")



if __name__ == "__main__":
    from gspf_ilp import create_graph_code


    numq=13
    g = nx.erdos_renyi_graph(numq, 0.7)
    #g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)
    print(ta.tableau2paulistring(xlogi))
    # print(len(zlogi))
    #for i in range(stabi.shape[0]):
        #print(tableau2paulistring(stabi[i,:]))

    T=sc.tableau_list_to_matrix(stabi)
    print(T.shape)
    #X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)
    lq=[2]

    T_new,destroyed_logicals=sc.remove_lost_qubits_from_tableau(T,lq)
    print(T_new.shape)
    zlogi_new=sc.find_clean_logical(T,zlogi,lq)

    if zlogi_new is not None:
        # print(type(zlogi_new))
        # print(ta.tableau2paulistring(zlogi_new))

        x_and_z_logical_anticommutation(xlogi, zlogi_new)
        #x_and_z_logical_anticommutation(xlogi, zlogi)

        # stabiliser_logical_commutation_check(np.asarray(T_new), zlogi)

        lost_qubit_logical_overlap(zlogi_new, lq)
        #lost_qubit_logical_overlap(zlogi, lq)