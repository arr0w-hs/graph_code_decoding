import numpy as np
import networkx as nx
import tableau as ta
import stabiliser_code as sc


def stabiliser_logical_commutation_check(stab_tableau : list[list], logical):

    for stab in stab_tableau:

        if not ta.commutation_check(stab, logical):
            print("Logical does not commute with the following stabiliser")
            print(ta.paulistring2tableau(stab))
    print("Stabiliser logical commutation test complete.", "\n")
    return

def x_and_z_logical_anticommutation(x_logi, z_logi):


    print("X logical = ", ta.tableau2paulistring(x_logi))
    print("Z logical = ", ta.tableau2paulistring(z_logi))

    res = ta.commutation_check(x_logi, z_logi)
    if res:
        print("Error: X and Z logical commute", "\n")
    else:

        res1 = ta.qubit_wise_commutation(x_logi, z_logi)
        print("X and Z logical anitcommute on the following qubits", res1)
        print("The number of anticommutations =", len(res1), "\n")

    return

if __name__ == "__main__":
    from gspf_ilp import create_graph_code

    nodes=7
    numq=nodes-1
    g = nx.erdos_renyi_graph(numq, 0.7)
    #g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)
    print(ta.tableau2paulistring(xlogi))
    # print(len(zlogi))
    #for i in range(stabi.shape[0]):
        #print(tableau2paulistring(stabi[i,:]))

    T=sc.tableau_list_to_matrix(stabi)
    #X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)
    lost_qubits=[0,1]

    T_new=sc.remove_lost_qubits_from_tableau(T,lost_qubits)
    # print(T_new)
    zlogi_new=sc.find_clean_logical(T,xlogi,lost_qubits)
    if zlogi_new is not None:
        print(ta.tableau2paulistring(zlogi_new))

        x_and_z_logical_anticommutation(xlogi, zlogi_new)
        x_and_z_logical_anticommutation(xlogi, zlogi)
