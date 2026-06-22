import numpy as np
from galois import GF2
import tableau as ta
import networkx as nx

from ortools.sat.python import cp_model


def update_tableau(tableau : list[list], measurements : list):

    n = len(measurements[0])//2
    tableau = GF2(tableau)

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
 

def find_gspf_logical(tableau : list[list], logical: list,
                    other_logical : list,
                    measurements : list[list], lost_qubits : list,
                    g : int):
    """
    Find a logical satisfying g-SPF algebra of GF2.

    Parameters
    ----------
    tableau : list of list
        Tableau of the stabiliser code with stabilisers.
        The representation is X part then Z part ( X | Z ).

    logical : list
        The original logical that sets the afine space
        Can be X logical or Z logical.

    other_logical : list
        'other_logical' is the starting point of the other logical.
        If logical is X (Z) then other_logical has to be Z (X)

    measurements : list of list
        List of previously completed measurements.
        Each measuremnet in the list is a single qubit measurement
        represented in the tableau form ( X | Z ).

    lost_qubits : list
        A list of length '2*number_of_qubits' all the known lost qubits.
        Represented in the tableau form as (L|L).
        L is of length 'n' and is 1 if a qubit is lost.

    g : int
        The g in 'g-SPF'.

    Returns
    -------
    A loss-tolerant representation of the 'logical' satisfying g-SPF.

    """

    T = GF2(tableau)

    n, m = T.shape

    # Model
    model = cp_model.CpModel()

    # binary variable B_x
    b = [model.NewBoolVar(f"b_{i}") for i in range(n)]
    bt = [sum(b[i] * T[i, j] for i in range(n)) for j in range(m)]
    mod2_terms = []
    for j in range(m):
        raw = logical[j] + bt[j]
        max_raw = 1 + sum(int(T[i, j]) for i in range(n))
        k = model.NewIntVar(0, max_raw // 2, f"k_{j}")
        mod2_expr = raw - 2 * k
        model.Add(mod2_expr >= 0)
        model.Add(mod2_expr <= 1)
        mod2_terms.append(mod2_expr)


    num_qubits = m // 2
    support = []
    logical_x_part = mod2_terms[:num_qubits]
    logical_z_part = mod2_terms[num_qubits:]
    for q in range(num_qubits):
        sq = model.NewBoolVar(f"support_{q}")
        model.Add(sq >= logical_x_part[q])
        model.Add(sq >= logical_z_part[q])
        model.Add(sq <= logical_x_part[q] + logical_z_part[q])
        support.append(sq)


    # lost qubits constraints
    raw_L = sum(int(lost_qubits[j]) * mod2_terms[j] for j in range(m))
    max_raw_L = sum(int(lost_qubits[j]) for j in range(m))
    # r = model.NewIntVar(0, max_raw_L // 2, "logical_constraint_k")
    model.Add(raw_L == 0)


    # measurement constraints
    for i, meas in enumerate(measurements):
        meas_x = meas[:num_qubits]
        meas_z = meas[num_qubits:]

        for q in range(num_qubits):
            raw = int(meas_z[q]) * logical_x_part[q] + int(meas_x[q]) * logical_z_part[q]

            k_comm = model.NewIntVar(0, 1, f"comm_{q}")
            model.Add(raw == 2 * k_comm)


    # constraint for g-SPF
    ologi_x = other_logical[:num_qubits]
    ologi_z = other_logical[num_qubits:]
    anti_terms = []
    for q in range(num_qubits):
        anit_comm = int(ologi_z[q]) * logical_x_part[q] + int(ologi_x[q]) * logical_z_part[q]


        anti_j = model.NewIntVar(0, 1, f"anti_{j}")
        k_anti = model.NewIntVar(0, 1, f"k_anti_{j}")


        model.Add(anti_j == anit_comm - 2 * k_anti)
        anti_terms.append(anti_j)

    model.Add(sum(anti_terms) <= g)

    # objective function
    model.Minimize(sum(support))


    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 60
    solver.parameters.num_search_workers = 8

    status = solver.Solve(model)

    print("Status:", solver.StatusName(status))

    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print("Minimum support:", solver.ObjectiveValue())
        print("mod2 vector:", [solver.Value(v) for v in mod2_terms])
        print("support:", [solver.Value(v) for v in support])
        a = [solver.Value(v) for v in mod2_terms]
        print(ta.tableau2paulistring(a))
        bx_sol = np.array([solver.Value(b[i]) for i in range(n)], dtype=int)
        print(bx_sol)
        print(ta.tableau2paulistring(other_logical))

        return a

    else:
        return status


if __name__ == "__main__":



    gs = nx.cycle_graph(5)
    n = 4
    k = 1
    gg = 3

    xlogical = ["X0*Z1"]
    zlogical = ["Z0*Z3"]
    stabi = ["Z0*X1*Z2", "Z1*X2*Z3", "X0*Z1*Z2*X3"]

    previous_meas = ["Z2", "X3"]
    previous_meas = ["Z0"]
    xlogi = [ta.paulistring2tableau(ele, 4) for ele in xlogical][0]
    zlogi = [ta.paulistring2tableau(ele, 4) for ele in zlogical][0]
    stabi = [ta.paulistring2tableau(ele, 4) for ele in stabi]
    previous_meas = [ta.paulistring2tableau(ele, 4) for ele in previous_meas]

    # update_tableau(stabi, previous_meas)

    # print(x)
    # previous_meas = GF2(previous_meas)
    lost_qubits = ta.paulistring2tableau("Y1", 4)

    T = GF2(stabi)
    # T = T.row_space()
    # print((stabi, xlogi, zlogi, previous_meas, lost_qubits))

    find_gspf_logical(stabi, xlogi, zlogi, previous_meas, lost_qubits, gg)
    # find_gspf_logical()

    print()