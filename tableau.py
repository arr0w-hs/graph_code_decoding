"""for finding fidelity of qubit encoding/decoding"""
import random
from collections import defaultdict
import numpy as np



def tableau2paulistring(in_stab):
    """
    converts a stabilizer in tableau form
    into a pauli string in stim form
    """

    ps = str()
    n = len(in_stab)//2

    x1 = [ele for i, ele in enumerate(in_stab) if i<n]
    z1 = [ele for i, ele in enumerate(in_stab) if i>=n]

    for i, ele in enumerate(x1):
        if ele and not z1[i]:
            ps += "X"+str(i)+"*"
        elif ele and z1[i]:
            ps += "Y"+str(i)+"*"
        elif not ele and z1[i]:
            ps += "Z"+str(i)+"*"
        else:
            continue

    out_ps = str()
    ps.replace("*", "", 2)
    for i, ele in enumerate(ps):
        if i != len(ps)-1:
            out_ps += ele

    return out_ps


def qubit_wise_commutation(a1, a2):
    """
    a1 is the measurements already done
    a2 is a stabilizer

    it returns the locations of qubits that dont commute qubit-wise
    returns 0 if they commute, otherwise
    """
    # score = 0
    anticom_loc = []

    x1, z1 = np.hsplit(np.array(a1), 2)
    x2, z2 = np.hsplit(np.array(a2), 2)
    for i, ele in enumerate(x1):
        a = (ele*z2[i]+z1[i]*x2[i]+1)%2
        if not a:
            # score += 1
            anticom_loc.append(i)

        # else:
        #     anticom_loc = True

    return anticom_loc

def paulistring2tableau(pauli_str, num_qubits):

    tab = [0]*2*num_qubits
    pauli_str = pauli_str.split(sep = "*")

    for stab, *a in pauli_str:
        loc = str()
        for ele in a:
            loc += ele

        if stab == 'Z':
            tab[int(loc)] = 0
            tab[int(loc)+num_qubits] = 1
        elif stab == 'X':
            tab[int(loc)] = 1
            tab[int(loc)+num_qubits] = 0
        else:
            tab[int(loc)] = 1
            tab[int(loc)+num_qubits] = 1

    return tab


def commutation_check(a1, a2):
    """commutation between two pauli strings
    in tableau form.

    returns 1 if they commute, 0 otherwise"""

    mid = len(a1) // 2
    s = 0
    for i in range(mid):
        s ^= (a1[i] & a2[mid+i]) ^ (a2[i] & a1[mid+i])

    return s == 0


def er_depol_channel(error_prob, log):
    thres = random.random()
    if thres >= error_prob:
        return ((0,0), "I") if log else ((0,0), None)

    if thres < error_prob / 3:
        return ((1,0), "X") if log else ((1,0), None)
    if thres < 2*error_prob / 3:
        return ((0,1), "Z") if log else ((0,1), None)
    return ((1,1), "Y") if log else ((1,1), None)


def depol_qubit(er_tableau, dim, target, error_prob, log):
    """depolarise a qubit"""
    if dim != len(er_tableau)//2:
        print(er_tableau, dim)
        print("wrong dimensions in depol qubit")

    error, er1 = er_depol_channel(error_prob, log)

    # er_tableau[target] ^= error[0]
    # er_tableau[dim+target] ^= error[1]

    if error[0]: er_tableau[target] ^= 1
    if error[1]: er_tableau[dim+target] ^= 1


    return er_tableau, er1


def er_cz_old(er_tableau, dim, q1, q2, er_list, log, error_prob=0):
    """
    apply CZ gate between q1 and q2

    any error caused during cz is of the
    following form: ("cz", er1, q2, q1,)

    er1 is the error (X,Y,Z,I)
    q2 is qubit the error acts on
    q1 is the other qubit of the CZ gate

    """

    er_tab1 = er_tableau.copy()

    er_tableau[dim+q1] = er_tableau[q2]^er_tableau[dim+q1]
    er_tableau[dim+q2] = er_tab1[q1]^er_tableau[dim+q2]

    er_tableau, er1 = depol_qubit(er_tableau, dim, q1, error_prob, log)
    er_tableau, er2 = depol_qubit(er_tableau, dim, q2, error_prob, log)

    if log:
        er_list.append(("cz", er1, q1, q2,))
        er_list.append(("cz", er2, q2, q1,))

    return er_tableau, er_list


def er_cz(er_tableau, dim, cz_list, er_list = None, log = False, error_prob=0):
    """apply CZ gate between q1 and q2"""
    # print(cz_list, dim, "hshhs")
    for q1, q2 in cz_list:

        # er_tab1 = er_tableau.copy()

        if er_tableau[q2]: er_tableau[dim+q1] ^= er_tableau[q2]
        if er_tableau[q1]: er_tableau[dim+q2] ^= er_tableau[q1]

        er_tableau, er1 = depol_qubit(er_tableau, dim, q1, error_prob, log)
        er_tableau, er2 = depol_qubit(er_tableau, dim, q2, error_prob, log)

        if log:
            er_list.append(("cz", er1+str(q1), q1, q2,))
            er_list.append(("cz", er2+str(q2), q1, q2,))

    return er_tableau, er_list


def er_cnot(er_tableau, dim, cnot_list, er_list = None, log = False, error_prob=0):
    """for propagation of errors thorugh cnot gate
    assuming q1 is the control and q2 is the target"""

    for q1, q2 in cnot_list:

        # er_tab1 = er_tableau.copy()
        if er_tableau[q1]: er_tableau[q2] ^= 1
        if er_tableau[dim+q2]: er_tableau[dim+q1] ^= 1

        er_tableau, er1 = depol_qubit(er_tableau, dim, q1, error_prob, log)
        er_tableau, er2 = depol_qubit(er_tableau, dim, q2, error_prob, log)
        # print(tableau2paulistring(er_tab1), q1, er1, q2, er2, tableau2paulistring(er_tableau))

        if log:
            er_list.append(("cnot", er1+str(q1), q1, q2,))
            er_list.append(("cnot", er2+str(q2), q1, q2,))

    return er_tableau, er_list


def er_h(er_tableau, dim, qubit_list):

    for qubit in qubit_list:
        xval = er_tableau[qubit]
        er_tableau[qubit] = er_tableau[qubit+dim]
        er_tableau[qubit+dim] = xval

    return er_tableau


def er_s(er_tableau, dim, qubit_list):

    for qubit in qubit_list:
        er_tableau[qubit+dim] ^= er_tableau[qubit]

    return er_tableau


def depolarise_graphcode(er_tableau, num_qubits, boundary, error_prob, er_list, log):
    """apply depolarising channel on the state"""

    for ele in range(num_qubits-len(boundary)):
        er_tableau, er1 = depol_qubit(er_tableau, num_qubits, ele, error_prob, log)
        if log:
            er_list.append(("dep", er1, ele))

    return er_tableau, er_list


def find_req_measurements(stab_list, zlogi_list):
    """combine the list of all the measurements needed for
    state encoding and decoding"""

    stab_list = stab_list+zlogi_list
    stab_list = [ele for ele in stab_list]
    stab_list = [ele.split(sep = "*") for ele in stab_list]
    stab_list = [elem for ele in stab_list for elem in ele]

    return list(set(stab_list))






if __name__ =="__main__":

    import networkx as nx

    gs = nx.star_graph(4)
    bound = [[0,1,1,0,0],[0,0,1,1,0], [0,1,0,0,1]]
    zlogical = ["X4*X1", "Z0*X3", "Z0*X4"]
    stabi = ["X1*X2*X3*X4"]

    for i in range(len(bound)):
        gs.add_node(gs.number_of_nodes()+i)
    gs = nx.convert_node_labels_to_integers(gs)
