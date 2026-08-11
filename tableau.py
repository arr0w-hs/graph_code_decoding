"""for finding fidelity of qubit encoding/decoding"""
import random
from collections import defaultdict
import numpy as np
from galois import GF2
import galois
import stabiliser_code as sc



def _row2paulistring(row,indices:None):
    n = len(row) // 2

    x, z = row[:n], row[n:]

    if indices is None:
        indices=np.arange(n)

    terms = []
    for i in range(n):
        if x[i] and z[i]:
            terms.append(f"Y{indices[i]}")
        elif x[i]:
            terms.append(f"X{indices[i]}")
        elif z[i]:
            terms.append(f"Z{indices[i]}")

    return "*".join(terms) if terms else "I"


def tableau2paulistring(in_stab, indices:list=None):
    """
    Convert an X|Z stabilizer tableau to Stim-style Pauli strings.

    in_stab : (2n,) array   -> returns a single string
              (k, 2n) array -> returns a list of k strings
    Column layout assumed: [x_0 ... x_{n-1} | z_0 ... z_{n-1}].
    """
    if in_stab is not None:

        arr = np.asarray(in_stab, dtype=bool)

        if arr.ndim == 1:
            return _row2paulistring(arr,indices=indices)

        if arr.ndim == 2:
            return [_row2paulistring(row,indices=indices) for row in arr]
        raise ValueError(f"expected 1D or 2D array, got ndim={arr.ndim}")
    else:
        return None


def tableau_list_to_matrix(tableau:list[list]):

    if isinstance(tableau,np.ndarray):
        return tableau

    length = max(map(len, tableau))
    tableau=[ti+[None]*(length-len(ti)) for ti in tableau]

    if np.isnan(tableau).any():
        raise ValueError("Tableau does not contain lists of the same lengths.")

    return np.array(tableau)

def to_gf2_tableau(T): #this preserves shape

    if isinstance(T, galois.FieldArray):
        return T

    if isinstance(T, list):
        arr = np.asarray(T)
        # flat list of scalars -> 1-D int array; rectangular list of lists -> 2-D int array;
        # both wrap into GF2 directly. Only a ragged nested list yields dtype=object,
        # which needs the padding logic in tableau_list_to_matrix.
        if arr.dtype == object:
            arr = tableau_list_to_matrix(T)
        return GF2(np.asarray(arr).astype(np.int64))

    if isinstance(T, np.ndarray):
        return GF2(T.astype(np.int64))

    raise TypeError(f"unsupported type for T: {type(T)}")



def construct_Omega_Matrix(n_qubits):

    Omega=np.zeros((2*n_qubits,2*n_qubits))
    Omega[n_qubits:,:n_qubits]=np.eye(n_qubits,n_qubits)
    Omega[:n_qubits,n_qubits:]=np.eye(n_qubits,n_qubits)

    return Omega


def qubit_wise_commutation(a1, a2):
    """
    a1 is the measurements already done
    a2 is a stabilizer

    it returns the locations of qubits that dont commute qubit-wise
    returns 0 if they commute, otherwise
    """
    # score = 0
    if a1 is None or a2 is None:
        return None
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
    if len(pauli_str) >0:
        pauli_str = pauli_str.split(sep = "*")
    else:
        pauli_str = []

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


def sp(a, b): #computes the symplectic product between a and b

    a=to_gf2_tableau(a).ravel() #make 1-D TODO: check generally how shapes of arrays are changing
    b=to_gf2_tableau(b).ravel()

    num_qubits=len(a)//2

    assert len(b)//2==num_qubits,'a and b shape inconsistent'

    Omega=to_gf2_tableau(construct_Omega_Matrix(num_qubits))

    return int(a @ Omega @ b)

def turn_into_symplectic_basis(xz_pair:list,logicals:list): #takes in a list of logicals and turns them into a symplectic basis
    """xz_pair: tuple of the form (x,z) where x is a representative of the logical x and z is the representative of logical z
    they must anti-commute
    logicals: the FULL new normaliser basis, i.e. the normaliser outside of the stabiliser of the whole code

    Returns: list of pairs of (x_i,z_i) for each logical qubit i. The first entry will be (x,z) with the original xz_pair from the input.
    The later entries will fulfil: {x_i,z_i}=0 and [x_i,x_j]=[x_i,z_j]=[z_i,z_j]=0"""


    num_logical_qubits_remain=(len(logicals))//2
    X_logical_qubit=xz_pair[0]
    Z_logical_qubit=xz_pair[1]

    assert sp(Z_logical_qubit,X_logical_qubit)==1; 'Z and X must anti-commute'
    assert sc.rank_F2(to_gf2_tableau(logicals))==num_logical_qubits_remain*2;'Logicals must be linearly independent'
    additional_logicals=[]
    if num_logical_qubits_remain>1:
        
        #making the additional logicals commute with X and Z
        for lg in logicals:

            c = to_gf2_tableau(lg).ravel()


            if sp(c, Z_logical_qubit) == 1:

                c = c + X_logical_qubit

            if sp(c, X_logical_qubit) == 1:

                c = c + Z_logical_qubit

            if np.asarray(c).any(): #this makes sure to only add c to the additional logicals if it was not
                #equal to X_logical_qubit or Z_logical_qubit
                additional_logicals.append(c)

        additional_logicals=to_gf2_tableau(additional_logicals)
        additional_logicals=additional_logicals.row_reduce()

        additional_logicals = additional_logicals[np.any(additional_logicals, axis=1)]

        additional_logical_pairs = symplectic_basis(additional_logicals)


    return [(X_logical_qubit,Z_logical_qubit)]+additional_logical_pairs

def turn_tableau_into_TXZY(T): #robust version
    T = to_gf2_tableau(T)

    if T.ndim == 1:
        num_qubits = T.shape[0] // 2
        x_part = T[:num_qubits]
        z_part = T[num_qubits:]
        yblock = x_part ^ z_part
        return to_gf2_tableau(np.concatenate([T, yblock]))

    if T.ndim == 2:
        num_qubits = T.shape[1] // 2
        yblock = T[:, :num_qubits] ^ T[:, num_qubits:]
        return to_gf2_tableau(np.hstack([T, yblock]))

    raise ValueError(f"expected 1D or 2D tableau, got ndim={T.ndim}")

def turn_TXZY_into_tableau(T):

    T=to_gf2_tableau(T)

    num_qubits=T.shape[1]//3

    Zblock=T[:,:num_qubits]
    Xblock=T[:,num_qubits:2*num_qubits]

    T_new=np.hstack((Xblock,Zblock))


    return T_new

def turn_TXZYerror_into_tableau(T):

    T = to_gf2_tableau(T)
    num_qubits = T.shape[1] // 3

    a = T[:, :num_qubits]                 # pairs with the sx columns -> Z errors
    b = T[:, num_qubits:2*num_qubits]     # pairs with the sz columns -> X errors
    c = T[:, 2*num_qubits:]               # Y errors

    Xblock = b + c    #x+x+z, check 3n notation
    Zblock = a + c

    return np.hstack((Xblock, Zblock))

def turn_tableau_into_TXZYerror(T): #robust version, inverse of turn_TXZYerror_into_tableau
    T = to_gf2_tableau(T)

    if T.ndim == 1:
        num_qubits = T.shape[0] // 2
        x_part = np.asarray(T[:num_qubits]).astype(bool)
        z_part = np.asarray(T[num_qubits:]).astype(bool)
        a = z_part & ~x_part   # pure Z -> Z-error block
        b = x_part & ~z_part   # pure X -> X-error block
        c = x_part & z_part    # Y      -> Y-error block
        return to_gf2_tableau(np.concatenate([a, b, c]).astype(np.int64))

    if T.ndim == 2:
        num_qubits = T.shape[1] // 2
        x_part = np.asarray(T[:, :num_qubits]).astype(bool)
        z_part = np.asarray(T[:, num_qubits:]).astype(bool)
        a = z_part & ~x_part
        b = x_part & ~z_part
        c = x_part & z_part
        return to_gf2_tableau(np.hstack([a, b, c]).astype(np.int64))

    raise ValueError(f"expected 1D or 2D tableau, got ndim={T.ndim}")


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

def symplectic_basis(logicals):
    """
    Given a linear basis of logical operators (each (x|z), length 2n),
    return a symplectic basis: a list of conjugate pairs [(X0,Z0),(X1,Z1),...]
    where Xi anticommutes with Zi and commutes with all other basis elements.

    Vectors with no anticommuting partner in the span (the symplectic radical)
    are discarded rather than raising. This handles a linear basis of a
    possibly-degenerate space, e.g. the logical space of a code after
    lost-qubit removal.
    """

    # working list of GF2 row vectors
    if isinstance(logicals, list):
        ops = [to_gf2_tableau(l).ravel() for l in logicals]
    elif isinstance(logicals, np.ndarray):
        ops = [to_gf2_tableau(logicals[i, :]).ravel() for i in range(logicals.shape[0])]

    xz_pairs_per_logical = []  # pairs of X,Z logical operators per qubit

    while ops:
        a = ops.pop(0)  # take the next operator, find a partner that anticommutes with it
        partner_idx = None
        for i, b in enumerate(ops):
            if sp(a, b) == 1:
                partner_idx = i
                break

        if partner_idx is None:
            # a lies in the radical: it commutes with everything remaining.
            # It is not a logical degree of freedom — discard and continue.
            continue

        b = ops.pop(partner_idx)

        new_ops = []
        # make all other operators commute with a and b
        for c in ops:
            if sp(c, b) == 1:      # c anticommutes with b -> add a to c
                c = c + a
            if sp(c, a) == 1:      # c anticommutes with a -> add b to c
                c = c + b
            new_ops.append(c)
        ops = new_ops
        xz_pairs_per_logical.append((a, b))

    return xz_pairs_per_logical



if __name__ =="__main__":

    import networkx as nx

    gs = nx.star_graph(4)
    bound = [[0,1,1,0,0],[0,0,1,1,0], [0,1,0,0,1]]
    zlogical = ["X4*X1", "Z0*X3", "Z0*X4"]
    stabi = ["X1*X2*X3*X4"]

    for i in range(len(bound)):
        gs.add_node(gs.number_of_nodes()+i)
    gs = nx.convert_node_labels_to_integers(gs)
