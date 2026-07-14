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


def generalised_spf_logical(tableau : np.ndarray, xlogical: list,
                    zlogical : list,
                    measurements : list[list], lost_qubits : list,
                    g : int,
                    target_qubit = None,
                    max_time = 60_0):
    """
    Find a logical satisfying g-SPF algebra of GF2.

    Parameters
    ----------
    tableau : numpy array
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

    target_qubit : int
        The required output qubit. Currently only of length 1, could be made a list.

    max_time : int
    The maximum amount of time in seconds the Solver runs for

    Returns
    -------
    A loss-tolerant representation of the 'logical' satisfying g-SPF.

    """

    T = GF2(tableau)

    num_stab, m = T.shape
    num_qubits = m // 2
    assert m % 2 == 0, "Tableau length must be even"
    assert len(xlogical) == m, f"xlogical must have length {m}"
    assert len(zlogical) == m, f"zlogical must have length {m}"
    assert len(lost_qubits) == m, f"lost_qubits must have length {m}"
    assert g >= 1, "g must be at least 1"

    for meas in measurements:
        assert len(meas) == m, f"measurement must have length {m}"
    # Model
    model = cp_model.CpModel()

    # binary variable vector [B_x|B_z]
    bx = [model.NewBoolVar(f"bx_{i}") for i in range(num_stab)]
    bz = [model.NewBoolVar(f"bz_{i}") for i in range(num_stab)]

    bxt = [sum(bx[i] * int(T[i, j]) for i in range(num_stab)) for j in range(m)]
    bzt = [sum(bz[i] * int(T[i, j]) for i in range(num_stab)) for j in range(m)]

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

        raw = sum(
            int(meas_z[q]) * xlogical_x_part[q] + int(meas_x[q]) * xlogical_z_part[q]
            for q in range(num_qubits)
        )

        k_comm = model.NewIntVar(0, num_qubits, f"commx_{i}")
        model.Add(raw == 2 * k_comm)

    # measurement constraints Z
    for i, meas in enumerate(measurements):
        meas_x = meas[:num_qubits]
        meas_z = meas[num_qubits:]

        raw = sum(
            int(meas_z[q]) * zlogical_x_part[q]
            + int(meas_x[q]) * zlogical_z_part[q]
            for q in range(num_qubits)
        )

        k_comm = model.NewIntVar(0, num_qubits, f"commz_{i}")
        model.Add(raw == 2 * k_comm)

    # constraint for g-SPF
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


        anti_q = model.NewIntVar(0, 1, f"anti_{q}")
        k_anti = model.NewIntVar(0, 1, f"k_anti_{q}")

        if q == target_qubit:# and target_qubit is not None:
            model.Add(anti_q == 1)

        model.Add(anti_q == anti_comm - 2 * k_anti)
        anti_terms.append(anti_q)

    # model.Add(sum(anti_terms) <= g)
    anti_sum = sum(anti_terms)
    model.Add(anti_sum <= g)

    k_total = model.NewIntVar(0, num_qubits, "k_total_anti")
    model.Add(anti_sum == 2 * k_total + 1)

    # objective function
    model.Minimize(sum(support))


    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.num_search_workers = min(8, os.cpu_count() or 1)
    status = solver.Solve(model)

    # print("Status:", solver.StatusName(status))
    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        x = [solver.Value(v) for v in xmod2_terms]
        z = [solver.Value(v) for v in zmod2_terms]

        result = {
            "success": True,
            "status": status,
            "status_name": status_name,
            "objective": solver.ObjectiveValue(),
            "x": x,
            "z": z,
            "support_size": [solver.Value(v) for v in support],
            "bx": np.array([solver.Value(bx[i]) for i in range(num_stab)], dtype=int),
            "bz": np.array([solver.Value(bz[i]) for i in range(num_stab)], dtype=int),
        }

        # print("Minimum support:", result["objective"])
        # print("mod2 vector:", z)
        # print("support:", result["support_size"])
        # print(ta.tableau2paulistring(z))
        # print(ta.tableau2paulistring(x))

    else:
        result = {
            "success": False,
            "status": status,
            "status_name": status_name,
            "objective": None,
            "x": xlogical,
            "z": zlogical,
            "support": None,
            "bx": None,
            "bz": None,
        }

    return result


def generalised_spf_logical_heuristic(tableau, lost_qubits : list,
                    target_qubit:int= None):
    """
    Find a logical satisfying g-SPF algebra of GF2.

    Parameters
    ----------
    tableau :
        Tableau of the stabiliser code with stabilisers.
        The representation is X part then Z part ( X | Z ).


    lost_qubits : list
        A list of length all the known lost qubits. [0,1,2] means qubit 0,1 and 2 are lost

    target_qubit : int
        The required output qubit. Currently only of length 1, could be made a list.

    max_time : int
    The maximum amount of time in seconds the decoder runs for

    Returns
    -------
    A loss-tolerant representation of the 'logical' minimising g heuristically

    """

    T = sc.to_gf2_tableau(tableau)

    num_stab, m = T.shape
    num_qubits = m // 2


    #find logical op basis:

    _,_,logical_ops=sc.find_logical_op_basis(T,num_qubits)
    logical_repr=logical_ops[0]

    #remove lost qubits

    T_clean,indices=sc.remove_lost_qubits_from_tableau(T,lost_qubits)
    logi_clean,indices=sc.find_clean_logical(T,logical_repr,lost_qubits)

    result = {
            "success": False,
            "x":None,
            "z":None
        }

    if logi_clean is None:
        print("logical information destroyed")
        return result


    # find short first logical

    short_first=dc.find_short_first_logical(T_clean,logi_clean)

    if short_first is None: #BP decoder failed
        return result

    #find second logical
    if target_qubit is not None:
        short_second=sc.find_anti_commuting_logi_at_O(T_clean,short_first,target_qubit) #note that this does
        #not need to be short


    short_second=dc.find_short_second_logical(T_clean,short_first)

    if short_second is None:
        return result


    result = {
            "success": True,

            "x": ta.tableau2paulistring(short_first,indices=indices),
            "z": ta.tableau2paulistring(short_second,indices=indices)
        }

    #TODO: output overlap

    return result


if __name__ == "__main__":

    numq = 10
    g = nx.erdos_renyi_graph(numq, 0.7)
    # g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = dc.create_graph_code(g)

    numq -= 1
    gg = 1

    previous_meas = ["Z1*Z2", "X1*X2"]
    # previous_meas = ["Z1", "X2"]
    previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]
    # print(previous_meas)

    lost_qubits = np.unique(np.random.randint(0, numq, numq//4))
    # lost_qubits = []
    s = str()
    for ele in lost_qubits:
        s += "Y"+str(ele)+"*"
    s = s[:-1]
    print(s)
    # lost_qubits = ta.paulistring2tableau("Y0", numq)
    lost_qubits = ta.paulistring2tableau(s, numq)


    T = GF2(stabi)
    generalised_spf_logical(stabi, xlogi, zlogi, previous_meas, lost_qubits, gg, target_qubit=None)

    print()