import numpy as np
import galois
from galois import GF2
import tableau as ta
import networkx as nx

from ortools.sat.python import cp_model


def update_tableau(tableau : GF2, measurements : list):

    n = len(measurements[0])//2
    # tableau = GF2(tableau)

    for meas in measurements:
        location = [i for i in range(n) if meas[i]+meas[n+i] > 0][0]
        meas = [meas[location], meas[n+location]]

        anticommuting_stab = []
        commuting_stab = []

        for stab in tableau:
            short_stab = [ele for i, ele in enumerate(stab) if i == location or i ==location+n]

            if not ta.commutation_check(short_stab, meas):
                anticommuting_stab.append(stab)
            elif ta.commutation_check(short_stab, meas):
                commuting_stab.append(stab)

        if len(anticommuting_stab) > 0:
            pivot = anticommuting_stab[0].copy()

            anticommuting_stab = [ stab + pivot for stab in anticommuting_stab[1:]]

        tableau = commuting_stab + anticommuting_stab

    return tableau

def append_logical_to_tableau(tableau,logical,num_qubits):

    if type(logical)==str:
        logical=ta.paulistring2tableau(logical,num_qubits)
    
    logical=GF2(np.array(logical, dtype=int))
    tableau=GF2(np.vstack([tableau,logical]))

    return tableau

def append_rows_to_tableau(tableau,row):

    row=GF2(np.array(row, dtype=int))
    tableau=GF2(np.vstack([tableau,row]))

    return tableau

    
def tableau_list_to_matrix(tableau:list[list]):

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

def find_ker_minus_rowspace(M): #M must be GF2 matrix
    
    if not isinstance(M, galois.GF2):
        raise TypeError("M must be binary matrix of type galois.GF(2)")
    
    ker=M.null_space()
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
    


def find_logical_op_basis(tableau_matrix,n_qubits): #idk if this could be super slow

    T = GF2(tableau_matrix)      # numpy array of 0/1

    is_CSS = np.all(tableau_matrix[:n_qubits, n_qubits:] == 0) and np.all(tableau_matrix[n_qubits:, :n_qubits] == 0)

    if is_CSS: #convention: T=((H_x,0),(0,H_z))
        T_x=T[:n_qubits,:n_qubits]
        T_z= T[n_qubits:,n_qubits:]
        Z_logicals=find_ker_minus_rowspace(T_x)
        X_logicals=find_ker_minus_rowspace(T_z)
        n_zeros= GF2(np.zeros(n_qubits, dtype=int))

        for i in range(len(Z_logicals)):
            Z_logicals[i]=np.hstack([n_zeros, Z_logicals[i]])

        for i in range(len(X_logicals)):
            X_logicals[i]=np.hstack([X_logicals[i], n_zeros])  # X logicals get zeros on the Z side

        return X_logicals,Z_logicals,None
          
            
    else:

        T=T.dot(GF2(construct_Omega_Matrix(n_qubits)))  
        logicals=find_ker_minus_rowspace(T)

        return None,None,logicals


    
 

def main():
    return

if __name__ == "__main__":

    main()