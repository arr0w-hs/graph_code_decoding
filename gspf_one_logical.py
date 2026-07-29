import os
import numpy as np
from galois import GF2
import tableau as ta
import stabiliser_code as sc
import networkx as nx
import decoder_methods as dc

from ortools.sat.python import cp_model

def find_gspf_one_logical(tableau : list[list], logical: list,
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
    if lost_qubits: #making sure there is no error if lost_qubits is empty #jelena: this only works if lost_qubits is list
        raw_L = sum(int(lost_qubits[j]) * mod2_terms[j] for j in range(m))
        # max_raw_L = sum(int(lost_qubits[j]) for j in range(m))
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
