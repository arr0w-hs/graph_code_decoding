import numpy as np
import galois
from galois import GF2
import tableau as ta
import networkx as nx
from itertools import product
from decoder_methods import create_graph_code

from ortools.sat.python import cp_model

def lost_indices_to_mask(lost, n_qubits):
    """
    Convert a list of lost qubit indices into a (L|L).

    lost     : iterable of qubit indices that are lost
    n_qubits : total number of qubits, n

    """
    lost = np.atleast_1d(np.asarray(lost, dtype=int))

    if lost.size and (lost.min() < 0 or lost.max() >= n_qubits):
        raise ValueError(f"lost qubit indices must lie in 0..{n_qubits - 1}")

    L = np.zeros(n_qubits, dtype=int)
    L[lost] = 1

    return np.concatenate([L, L])


def update_tableau_after_measurements(tableau : GF2, measurements : list):

    n = len(measurements[0])//2
    tableau = GF2(tableau)

    tab_copy = tableau.copy()
    for meas in measurements:
        # location = [i for i in range(n) if meas[i]+meas[n+i] > 0][0]
        # meas = [meas[location], meas[n+location]]

        anticommuting_stab = []
        commuting_stab = []

        for stab in tab_copy:
            # short_stab = [ele for i, ele in enumerate(stab) if i == location or i ==location+n]

            if not ta.commutation_check(stab, meas):
                anticommuting_stab.append(stab)
            elif ta.commutation_check(stab, meas):
                commuting_stab.append(stab)

        if len(anticommuting_stab) > 0:
            pivot = anticommuting_stab[0].copy()

            anticommuting_stab = [ stab + pivot for stab in anticommuting_stab[1:]]

        tab_copy = commuting_stab + anticommuting_stab + [meas.copy()]

    return tab_copy

def append_logical_to_tableau(tableau,logical):
    num_qubits=tableau.shape[1]//2 #note that this does not assume CSS form
    if type(logical)==str:
        logical=ta.paulistring2tableau(logical,num_qubits)

    logical = np.asarray(logical).reshape(1, -1)   # force a single row (1, n)
    logical=GF2(np.array(logical, dtype=int))
    tableau=GF2(np.vstack([tableau,logical]))

    return tableau

def append_rows_to_tableau(tableau,row):

    row=GF2(np.array(row, dtype=int))
    tableau=GF2(np.vstack([tableau,row]))

    return tableau


def tableau_list_to_matrix(tableau:list[list]):

    if isinstance(tableau,np.ndarray):
        return tableau

    length = max(map(len, tableau))
    tableau=[ti+[None]*(length-len(ti)) for ti in tableau]

    if np.isnan(tableau).any():
        raise ValueError("Tableau does not contain lists of the same lengths.")

    return np.array(tableau)

def construct_Omega_Matrix(n_qubits):

    Omega=np.zeros((2*n_qubits,2*n_qubits))
    Omega[n_qubits:,:n_qubits]=np.eye(n_qubits,n_qubits)
    Omega[:n_qubits,n_qubits:]=np.eye(n_qubits,n_qubits)

    return Omega

def compute_Pauli_weight(v):


    if isinstance(v,str):
        num_qubits=len(v)
        v=ta.paulistring2tableau(v,num_qubits)

    v = np.asarray(v).ravel()

    num_qubits=len(v)//2
    x_part, z_part = v[:num_qubits], v[num_qubits:]
    touched = (x_part | z_part)
    pauli_weight = int(touched.sum())

    return pauli_weight

def y_positions(v):

    if isinstance(v,str):
        num_qubits=len(v)
        v=ta.paulistring2tableau(v,num_qubits)

    v = np.asarray(v).ravel()

    num_qubits=len(v)//2
    x_part, z_part = v[:num_qubits], v[num_qubits:]

    y_mask = (x_part == 1) & (z_part == 1)

    return y_mask


def rank_F2(M):

    if not isinstance(M, galois.GF2):
        M=GF2(M)

    row_space=M.row_space()
    rref = row_space.row_reduce()
    row_space_rank = int(np.any(rref, axis=1).sum())

    return row_space_rank


def find_ker_minus_rowspace(M,CSS:bool=False): #M must be GF2 matrix

    M=to_gf2_tableau(M)

    if not isinstance(M, galois.GF2):
        raise TypeError("M must be binary matrix of type galois.GF(2)")

    if CSS:
        ker=M.null_space()
    else:
        num_qubits=M.shape[1]//2
        ker=(M@construct_Omega_Matrix(num_qubits)).null_space()

    row_space=M.row_space()
    row_space_rank=rank_F2(M)
    ker_minus_rowspace=[]

    for row in ker:
        test = GF2(np.vstack([row_space, row.reshape(1, -1)]))
        rref_test = test.row_reduce()
        rank = int(np.any(rref_test, axis=1).sum())

        if rank > row_space_rank:
            ker_minus_rowspace.append(row)

    return ker_minus_rowspace


def to_gf2_tableau(T):

    if isinstance(T, galois.FieldArray):
        return T


    if isinstance(T, list):
        T = tableau_list_to_matrix(T)
        return GF2(np.asarray(T).astype(np.int64))


    if isinstance(T, np.ndarray):
        return GF2(T.astype(np.int64))

    raise TypeError(f"unsupported type for T: {type(T)}")


def overlap_with_lost_qubits(T,lost_qubits):

    T=to_gf2_tableau(T)
    num_qubits=T.shape[1]//2
    x_touch = T[:, lost_qubits].any(axis=1)               # X part of lost qubits
    z_touch = T[:, np.asarray(lost_qubits) + num_qubits].any(axis=1)    # Z part of lost qubits
    touches_lost = x_touch | z_touch
    keep_rows=~touches_lost

    return touches_lost,keep_rows

def reduced_row_echelon_form(T,column_from_where_to_start_RREF): # returns partial reduced echelon form, where the last rows
    #have only ones for the columns in list columns
    #logicals are returned in the same order
    T=to_gf2_tableau(T)

    logical_columns=np.arange(column_from_where_to_start_RREF,T.shape[1]) #to list when the logical columns start

    pivot_row = 0

    destroyed_logicals=[]
    for c in logical_columns:

        rows_with_1_in_c = [r for r in range(pivot_row, T.shape[0]) if T[r, c]] #rows with 1 on logical column

        if not rows_with_1_in_c:
            destroyed_logicals.append(c-column_from_where_to_start_RREF) #logical was destroyed by loss
            continue #go to next logical

        r = rows_with_1_in_c[0]

        T[[pivot_row, r]] = T[[r, pivot_row]]

        for rr in range(T.shape[0]):
            if rr != pivot_row and T[rr, c]:
                T[rr] = T[rr] + T[pivot_row]           # GF(2)

        pivot_row += 1


    s = pivot_row                       # number of survivors actually pivoted
    num_rows = T.shape[0]
    order = np.concatenate([np.arange(s, num_rows),   # non-logical rows first
                        np.arange(s)])             # logical rows last
    T = T[order]

    return T, destroyed_logicals


def split_stabilizer_except(stab, o):
    """
    Decompose a stabilizer into its single-qubit Pauli factors, omitting qubit o.

    stab : (2n,) array in [x_0..x_{n-1} | z_0..z_{n-1}] form
    o    : qubit index to exclude

    Returns a list of (2n,) GF2 rows, one per qubit i != o on which stab acts
    non-trivially. Each row is zero everywhere except at columns i and i+n,
    where it copies stab's entries. Identity qubits produce no entry.
    """
    v = GF2(np.asarray(stab).astype(np.int64)).ravel()
    n = v.shape[0] // 2

    if not 0 <= o < n:
        raise ValueError(f"o={o} outside qubit range 0..{n-1}")

    factors = []
    for i in range(n):
        if i == o:
            continue
        if not (v[i] or v[i + n]):
            continue
        row = GF2(np.zeros(2 * n, dtype=int))
        row[i] = v[i]
        row[i + n] = v[i + n]
        factors.append(row)

    return factors

def make_T_solve_anti_commuting_logi_at_O(T,first_logi,o:int):

    num_qubits=T.shape[1]//2
    first_logi=GF2(np.asarray(first_logi).astype(int)).ravel()
    if first_logi.shape[0]//2 != num_qubits:
        raise ValueError("Logical does not contain as many qubits as tableau")

    anti_commute_qubits = np.asarray([o])
    anti_commute_cols = np.concatenate([anti_commute_qubits, anti_commute_qubits + num_qubits])   # X and Z halves

    first_logi_anti_commute=np.zeros(T.shape[1],dtype=int)
    first_logi_anti_commute=GF2(first_logi_anti_commute)
    first_logi_anti_commute[anti_commute_cols] = first_logi[anti_commute_cols]



    first_logi_commute=first_logi.copy()
    first_logi_commute_list=split_stabilizer_except(first_logi_commute,o)



    for l in first_logi_commute_list:
        T=append_logical_to_tableau(T,l)

    T=append_logical_to_tableau(T,first_logi_anti_commute)


    syndrome=GF2(np.zeros(T.shape[0], dtype=int))
    syndrome[-1]=1

    return T,syndrome

def find_anti_commuting_logi_at_O(T,first_logi,o): #TODO: currently only works for one logical qubit

    num_qubits=T.shape[1]//2
    T_solve,syndrome=make_T_solve_anti_commuting_logi_at_O(T,first_logi,o)
    T_solve = T_solve@GF2(construct_Omega_Matrix(num_qubits).astype(np.int64))
    v = solve_gf2(T_solve, syndrome)
    if v is None:
        return None                          # no such logical exists
    return v

def index_array(n:int, lost:list):
    """
    Build np.concatenate([arange(n), arange(n)]) with every entry whose
    value appears in `lost` removed from both copies.

    """
    base = np.arange(n)
    keep = base[~np.isin(base, np.asarray(lost, dtype=int))]

    return keep

def remove_lost_qubits_from_tableau(T,lost_qubits:list,row_in_T_where_logical_begins:int=None,CSS:bool=False,\
                                    collapse:bool=True):
    #TODO: implement for CSS
    #can also remove lost qubits from tableau when the tableau includes the logical operators of the code
    #they first rows up to row row_in_T_where_logical_begins must be stabilisers. Returns the same order



    T=to_gf2_tableau(T)

    num_qubits=T.shape[1]//2

    if not lost_qubits:
        return T,[],index_array(num_qubits,lost_qubits)

    lost = np.asarray(lost_qubits)
    lost_cols = np.concatenate([lost, lost + num_qubits])   # X and Z halves
    T_columns_lost_qubits=T[:,lost_cols]
    coeffs=T_columns_lost_qubits.left_null_space() #note the left null space!
    coeffs=coeffs.row_reduce()
    destroyed_logicals=[]

    if row_in_T_where_logical_begins is not None:
        coeffs,destroyed_logicals=reduced_row_echelon_form(coeffs,row_in_T_where_logical_begins)

    T_update=coeffs@T

    if collapse:
        indices=index_array(num_qubits,lost_qubits)
        T_update=kick_out_qubits(T_update,lost_qubits)
    else:
        indices=index_array(num_qubits,[])

    return T_update,destroyed_logicals,indices


def find_clean_logical(T, logi, lost_qubits,collapse:bool=True):

    """Return logi multiplied by stabilizers so it has no support on lost_qubits.
    T:    (num_gen, 2n) GF2 stabilizer generators
    logi: (2n,) GF2 logical operator
    Returns cleaned logical (2n,) GF2, or None if it can't be cleaned."""
    T = to_gf2_tableau(T)
    logi = GF2(logi) if not isinstance(logi, galois.FieldArray) else logi

    n_qubits = T.shape[1] // 2
    lost = np.asarray(lost_qubits)
    lost_cols = np.concatenate([lost, lost + n_qubits])   # X and Z halves

    A = T[:, lost_cols]        # (num_gen, |lost_cols|) — stabilizers on lost qubits
    b = logi[lost_cols]        # (|lost_cols|,)          — logical on lost qubits


    c = solve_gf2(A.T, b)

    if c is None:
        print('oh no')
        indices=index_array(n_qubits,[])
        return None,indices          # not correctable: no clean representative exists

    logi_clean = logi + c @ T

    if collapse:
        indices=index_array(n_qubits,lost_qubits)
        print('indices in find_clean_logica',indices)
        logi_clean=kick_out_qubits(logi_clean,lost_qubits)
    else:
        indices=index_array(n_qubits,[])

    return logi_clean, indices


def solve_gf2(M, b):
    """returns x with M x = b over GF(2), or None if inconsistent."""
    M = GF2(M); b = GF2(b).reshape(-1, 1)
    R = GF2(np.hstack([M, b])).row_reduce()
    k = M.shape[1]
    x = GF2(np.zeros(k, dtype=int))
    for row in R:
        rowM, rhs = row[:k], row[k]
        if not rowM.any():
            if rhs:
                return None            # 0 = 1, inconsistent
            continue
        pivot = int(np.argmax(rowM.view(np.ndarray)))
        x[pivot] = rhs
    return x


def kick_out_qubits(T, qubits):
    """
    Remove the X and Z columns of the given qubits from a tableau.

    T      : (2n,) single stabilizer, or (k, 2n) tableau
    qubits : iterable of qubit indices to drop

    Returns the same dimensionality as the input.
    """
    T = to_gf2_tableau(T)

    if T.ndim not in (1, 2):
        raise ValueError(f"expected 1D or 2D tableau, got ndim={T.ndim}")

    n_cols = T.shape[-1]
    n_qubits = n_cols // 2

    lost = np.atleast_1d(np.asarray(qubits, dtype=int))
    lost_cols = np.concatenate([lost, lost + n_qubits])

    keep_cols = np.setdiff1d(np.arange(n_cols), lost_cols)

    return T[..., keep_cols] #... is short for take the whole array if 1d array, take all rows if 2d array

def restore_lost_qubits(T_reduced, indices, n_qubits):
    """
    Inverse of kick_out_qubits: re-embed a reduced tableau into the full
    2*n_qubits column layout, writing 0 into the X and Z columns of every
    qubit that is not in `indices`.

    T_reduced : (2m,) or (k, 2m) array, m = len(indices)
    indices   : surviving qubit labels, as returned by index_array
    n_qubits  : number of qubits in the original (pre-loss) tableau

    Returns the same dimensionality as the input, with 2*n_qubits columns.
    """
    T_reduced = to_gf2_tableau(T_reduced)

    if T_reduced.ndim not in (1, 2):
        raise ValueError(f"expected 1D or 2D tableau, got ndim={T_reduced.ndim}")

    indices = np.atleast_1d(np.asarray(indices, dtype=int))
    m = indices.size

    if T_reduced.shape[-1] != 2 * m:
        raise ValueError(
            f"reduced tableau has {T_reduced.shape[-1]} columns but "
            f"{m} surviving qubits implies {2 * m}"
        )
    if m and (indices.min() < 0 or indices.max() >= n_qubits):
        raise ValueError(f"indices must lie in 0..{n_qubits - 1}")

    full_shape = T_reduced.shape[:-1] + (2 * n_qubits,)
    T_full = GF2(np.zeros(full_shape, dtype=int))

    keep_cols = np.concatenate([indices, indices + n_qubits])
    T_full[..., keep_cols] = T_reduced

    return T_full

def add_measurements_to_tableau(T,measurements):
    T=to_gf2_tableau(T)
    pass


def find_logical_op_basis(tableau_matrix,n_qubits): #idk if this could be super slow

    T = GF2(tableau_matrix)      # numpy array of 0/1

    is_CSS = np.all(T[:n_qubits, n_qubits:] == 0) and np.all(T[n_qubits:, :n_qubits] == 0)
    #TODO: put this in other places?

    if is_CSS: #convention: T=((H_x,0),(0,H_z))
        T_x=T[:n_qubits,:n_qubits]
        T_z= T[n_qubits:,n_qubits:]
        Z_logicals=find_ker_minus_rowspace(T_x,CSS=is_CSS)
        X_logicals=find_ker_minus_rowspace(T_z,CSS=is_CSS)
        n_zeros= GF2(np.zeros(n_qubits, dtype=int))

        for i in range(len(Z_logicals)):
            Z_logicals[i]=np.hstack([n_zeros, Z_logicals[i]])

        for i in range(len(X_logicals)):
            X_logicals[i]=np.hstack([X_logicals[i], n_zeros])  # X logicals get zeros on the Z side

        return X_logicals,Z_logicals,None


    else:


        logicals=find_ker_minus_rowspace(T,CSS=is_CSS) #TODO check if working correctly

        return None,None,logicals


def turn_tableau_into_TXZY(T): #turns into (x|z|x+z) string
    T=to_gf2_tableau(T)
    num_qubits=T.shape[1]//2
    Yblock=np.zeros((T.shape[0],num_qubits),dtype=np.int64)

    Yblock=T[:,:num_qubits]^T[:,num_qubits:]

    Yblock=to_gf2_tableau(Yblock)
    T=np.hstack((T,Yblock))

    return T

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


def main():


    nodes=15
    numq=nodes-1
    g = nx.erdos_renyi_graph(numq, 0.7)
    #g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)

    print(ta.tableau2paulistring(xlogi))
    # print(len(zlogi))
    #for i in range(stabi.shape[0]):
        #print(tableau2paulistring(stabi[i,:]))

    T=tableau_list_to_matrix(stabi)
    numq=T.shape[1]//2
    print('number of qubits: ',numq)
    #X_logicals,Z_logicals,logicals=find_logical_op_basis(T,n_qubits)
    lost_qubits=[0,1]

    T_new,destroyed_logicals,indices=remove_lost_qubits_from_tableau(T,lost_qubits) # good
    print("new T: {}".format(ta.tableau2paulistring(restore_lost_qubits(T_new,n_qubits=numq,indices=indices))))
    print('new indices', indices)
    zlogi_new,z_indices=find_clean_logical(T,xlogi,lost_qubits) # nice
    if zlogi_new is not None:
        print('zlogi_new ',ta.tableau2paulistring(restore_lost_qubits(zlogi_new,n_qubits=numq,indices=indices)))
    print('z indices',z_indices)

if __name__ == "__main__":

    main()
