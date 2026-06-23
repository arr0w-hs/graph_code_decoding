import numpy as np
from galois import GF2
import tableau as ta
import networkx as nx

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


def find_gspf_logical(tableau : list[list], xlogical: list,
                    zlogical : list,
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

    num_stab, m = T.shape
    num_qubits = m // 2

    # Model
    model = cp_model.CpModel()

    # binary variable vector [B_x|B_z]
    bx = [model.NewBoolVar(f"bx_{i}") for i in range(num_stab)]
    bz = [model.NewBoolVar(f"bz_{i}") for i in range(num_stab)]

    bxt = [sum(bx[i] * T[i, j] for i in range(num_stab)) for j in range(m)]
    bzt = [sum(bz[i] * T[i, j] for i in range(num_stab)) for j in range(m)]

    # minimise the x logical
    xmod2_terms = []
    for j in range(m):
        raw = xlogical[j] + bxt[j]
        max_raw = 1 + sum(int(T[i, j]) for i in range(num_stab))
        k = model.NewIntVar(0, max_raw // 2, f"kx_{j}")
        mod2_expr = raw - 2 * k
        model.Add(mod2_expr >= 0)
        model.Add(mod2_expr <= 1)
        xmod2_terms.append(mod2_expr)

    support = []
    xlogical_x_part = xmod2_terms[:num_qubits]
    xlogical_z_part = xmod2_terms[num_qubits:]
    for q in range(num_qubits):
        sq = model.NewBoolVar(f"supportx_{q}")
        model.Add(sq >= xlogical_x_part[q])
        model.Add(sq >= xlogical_z_part[q])
        model.Add(sq <= xlogical_x_part[q] + xlogical_z_part[q])
        support.append(sq)

    # minimise the z logical
    zmod2_terms = []
    for j in range(m):
        raw = zlogical[j] + bzt[j]
        max_raw = 1 + sum(int(T[i, j]) for i in range(num_stab))
        k = model.NewIntVar(0, max_raw // 2, f"kz_{j}")
        mod2_expr = raw - 2 * k
        model.Add(mod2_expr >= 0)
        model.Add(mod2_expr <= 1)
        zmod2_terms.append(mod2_expr)

    # support = []
    zlogical_x_part = zmod2_terms[:num_qubits]
    zlogical_z_part = zmod2_terms[num_qubits:]
    for q in range(num_qubits):
        sq = model.NewBoolVar(f"supportz_{q}")
        model.Add(sq >= zlogical_x_part[q])
        model.Add(sq >= zlogical_z_part[q])
        model.Add(sq <= zlogical_x_part[q] + zlogical_z_part[q])
        support.append(sq)


    # lost qubits constraints
    xraw_L = sum(int(lost_qubits[j]) * xmod2_terms[j] for j in range(m))
    zraw_L = sum(int(lost_qubits[j]) * zmod2_terms[j] for j in range(m))
    # max_raw_L = sum(int(lost_qubits[j]) for j in range(m))
    # r = model.NewIntVar(0, max_raw_L // 2, "logical_constraint_k")
    model.Add(xraw_L == 0)
    model.Add(zraw_L == 0)


    # measurement constraints X
    for i, meas in enumerate(measurements):
        meas_x = meas[:num_qubits]
        meas_z = meas[num_qubits:]

        for q in range(num_qubits):
            raw = int(meas_z[q]) * xlogical_x_part[q] + int(meas_x[q]) * xlogical_z_part[q]

            k_comm = model.NewIntVar(0, 1, f"commx_{q}")
            model.Add(raw == 2 * k_comm)


    # measurement constraints Z
    for i, meas in enumerate(measurements):
        meas_x = meas[:num_qubits]
        meas_z = meas[num_qubits:]

        for q in range(num_qubits):
            raw = int(meas_z[q]) * zlogical_x_part[q] + int(meas_x[q]) * zlogical_z_part[q]

            k_comm = model.NewIntVar(0, 1, f"commz_{q}")
            model.Add(raw == 2 * k_comm)


    # constraint for g-SPF
    ologi_x = zlogical[:num_qubits]
    ologi_z = zlogical[num_qubits:]
    anti_terms = []
    for q in range(num_qubits):
        z = model.NewBoolVar(f"z_{q}")  # z = x AND y
        model.Add(z <= zlogical_z_part[q])
        model.Add(z <= xlogical_x_part[q])
        model.Add(z >= xlogical_x_part[q] + zlogical_z_part[q] - 1)


        z2 = model.NewBoolVar(f"z2_{q}")  # z = x AND y
        model.Add(z2 <= xlogical_z_part[q])
        model.Add(z2 <= zlogical_x_part[q])
        model.Add(z2 >= zlogical_x_part[q] + xlogical_z_part[q] - 1)

        # anit_comm = zlogical_z_part[q] * xlogical_x_part[q] + zlogical_x_part[q] * xlogical_z_part[q]
        anti_comm = z + z2


        anti_j = model.NewIntVar(0, 1, f"anti_{j}")
        k_anti = model.NewIntVar(0, 1, f"k_anti_{j}")


        model.Add(anti_j == anti_comm - 2 * k_anti)
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
        print("mod2 vector:", [solver.Value(v) for v in zmod2_terms])
        print("support:", [solver.Value(v) for v in support])
        z = [solver.Value(v) for v in zmod2_terms]
        print(ta.tableau2paulistring(z))
        x = [solver.Value(v) for v in xmod2_terms]
        print(ta.tableau2paulistring(x))
        bx_sol = np.array([solver.Value(bx[i]) for i in range(num_stab)], dtype=int)
        bz_sol = np.array([solver.Value(bz[i]) for i in range(num_stab)], dtype=int)
        # print(bx_sol)
        # print(bz_sol)
        # print(ta.tableau2paulistring(zlogical))

        return x,z

    else:
        return status


def create_graph_code(in_adj : np.array, code_node : int = 0):

    num_nodes = in_adj.shape[0]
    identity = np.identity(num_nodes, dtype = np.uint16)
    gen = np.hstack([identity, in_adj])
    zlogi = gen[code_node].copy()
    zlogi[code_node] = 0
    neigh = [i for i, ele in enumerate(in_adj[code_node]) if ele ==1]

    assert(len(neigh)>0)
    xlogi = gen[neigh[0]].copy()

    for ele in neigh[1:]:
        gen[ele] ^= gen[neigh[0]]

    rows_to_remove = [code_node, neigh[0]]
    gen = np.delete(gen, rows_to_remove, axis=0)

    gen = np.delete(gen, [code_node, code_node+num_nodes], axis=1)
    xlogi = np.delete(xlogi, [code_node, code_node+num_nodes])#, axis=1)
    zlogi = np.delete(zlogi, [code_node, code_node+num_nodes])#, axis=1)


    return xlogi, zlogi, gen


if __name__ == "__main__":

    numq = 30
    # g = nx.erdos_renyi_graph(numq, 0.7)
    g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = create_graph_code(g)

    numq -= 1
    gg = 3

    previous_meas = ["Z2", "X1"]
    previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]
    lost_qubits = ta.paulistring2tableau("Y0", numq)

    T = GF2(stabi)
    find_gspf_logical(stabi, xlogi, zlogi, previous_meas, lost_qubits, gg)

    print()