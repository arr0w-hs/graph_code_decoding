import numpy as np
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

def tableau_list_to_matrix(tableau:list[list]):

    length = max(map(len, tableau))
    tableau=[ti+[None]*(length-len(ti)) for ti in tableau]

    if np.isnan(tableau).any():
        raise ValueError("Tableau does not contain lists of the same lenghts.")

    return np.array(tableau)

def construct_Omega_Matrix(n_qubits): #TODO
    pass


def find_logical_op_basis(tableau_matrix,n_qubits): #idk if this could be super slow

    T = GF2(tableau_matrix)      # numpy array of 0/1

    T=T.dot(construct_Omega_Matrix(n_qubits)) #TODO idk if this dot does what I want

    return T.null_space()

def append_logical_X_or_Z(tableau_matrix,ker): #appends ONE of the logical operators per logical qubit to tableau
    pass

def main():
    return

if __name__ == "__main__":

    main()