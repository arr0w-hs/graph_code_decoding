
from galois import GF2
import numpy as np
import networkx as nx
import tableau as ta
import stabiliser_code as sc


def stabiliser_logical_commutation_check(stab_tableau : list[list], x_logi, z_logi, fail):

    print("Number of stabilisers:", len(stab_tableau))
    print("Number of qubits:", len(stab_tableau[0])//2)
    assert len(stab_tableau) > 0, "No stabilisers found"


    for stab in stab_tableau:

        if not ta.commutation_check(stab, x_logi):
            fail.append(f"Error: Logical {ta.tableau2paulistring(x_logi)} does not commute with stabiliser {ta.tableau2paulistring(stab)}")


        if not ta.commutation_check(stab, z_logi):
            fail.append(f"Error: Logical {ta.tableau2paulistring(z_logi)} does not commute with the following stabiliser {ta.tableau2paulistring(stab)}")

    print("Stabiliser logical commutation test complete.")

    return fail


def x_and_z_logical_anticommutation(x_logi, z_logi, g, fail):

    res = ta.commutation_check(x_logi, z_logi)
    if res:
        fail.append("Error: X and Z logical commute")
    else:

        res1 = ta.qubit_wise_commutation(x_logi, z_logi)
        # print("X and Z logical anticommute on the following qubits", res1)
        print("The number of logical anticommutations =", len(res1))
        if len(res1) > g:
            print("Anti-commutation of logicals exceeds g")

    print("Logical anticommutation test complete.")
    return fail

def lost_qubit_logical_overlap(x_logi, z_logi, lost_qubits, fail):

    for logical in (x_logi, z_logi):
        num_qubits = len(logical)//2
        error = []
        for qubit in lost_qubits:

            xpart = bool(logical[qubit])
            zpart = bool(logical[num_qubits+qubit])
            total = xpart or zpart

            if total:
                error.append(qubit)

        if len(error)>0:
            fail.append(f"Logical {ta.tableau2paulistring(logical)} and lost qubits have overlaps at: {error}")


    print("Lost-qubit logical test complete.")

    return fail


def logical_commute_with_measurements(x_logi, z_logi, measurement, fail):

    for meas in measurement:
        if not ta.commutation_check(x_logi, meas):
            fail.append("Logical does not commute with the measurements")
            fail.append(f"Logical: {ta.tableau2paulistring(x_logi)}, Measurement: {ta.tableau2paulistring(meas)}")

        if not ta.commutation_check(z_logi, meas):
            fail.append("Logical does not commute with the measurements")
            fail.append(f"Logical: {ta.tableau2paulistring(z_logi)}, Measurement: {ta.tableau2paulistring(meas)}")

    print("Measurement commutation test complete.")
    return fail


def test_gspf(tableau, x_logi, z_logi, measurements, lost_qubits, g=0):
    failure = []
    print("\n========================================")
    print("             g-SPF test")
    print("======================================== \n")

    failure = stabiliser_logical_commutation_check(tableau, x_logi, z_logi, failure)
    failure = x_and_z_logical_anticommutation(x_logi, z_logi, g, failure)
    failure = logical_commute_with_measurements(x_logi, z_logi, measurements, failure)
    failure = lost_qubit_logical_overlap(x_logi, z_logi, lost_qubits, failure)

    if len(failure) != 0:
        print("*************************")
        print("Following tests failed")
        print("*************************\n")
        for ele in failure:
            print(ele)

    return

if __name__ == "__main__":
    from gspf_ilp import create_graph_code


    numq=13
    g = nx.erdos_renyi_graph(numq, 0.7)
    #g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)
    # print(ta.tableau2paulistring(xlogi))
    # print(len(zlogi))
    #for i in range(stabi.shape[0]):
        #print(tableau2paulistring(stabi[i,:]))

    T=sc.tableau_list_to_matrix(stabi)
    # print(T.shape)
    #X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)
    lq=[2]

    T_new=sc.remove_lost_qubits_from_tableau(T,lq)
    print(T_new.shape)
    zlogi_new=sc.find_clean_logical(T,xlogi,lq)

    if zlogi_new is not None:
        # print(type(zlogi_new))
        # print(ta.tableau2paulistring(zlogi_new))

        x_and_z_logical_anticommutation(xlogi, zlogi_new)
        x_and_z_logical_anticommutation(xlogi, zlogi)

        # stabiliser_logical_commutation_check(np.asarray(T_new), zlogi)

        # lost_qubit_logical_overlap(zlogi_new, lq)
        # lost_qubit_logical_overlap(zlogi, lq)