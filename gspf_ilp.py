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


def generalised_spf_logical(tableau : np.ndarray, 
                    measurements : list[list], lost_qubits : list,
                    g : int,
                    target_qubit:int = None,
                    return_reduced_form:bool=False,
                    max_time:float = 60_0,
                    minimize_support:bool=False):
    """
    Find a logical satisfying g-SPF algebra of GF2.

    Parameters
    ----------
    tableau : numpy array
        Tableau of the stabiliser code with stabilisers.
        The representation is X part then Z part ( X | Z ).

    measurements : list of list
        List of previously completed measurements.
        Each measuremnet in the list is a single qubit measurement
        represented in the tableau form ( X | Z ).

    lost_qubits : list
        A list indices for the lost qubits.

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
    result = {
            "success": False,
            "status": None,
            "status_name": None,
            "objective": None,
            "x": None,
            "z": None,
            "support_size": None,
            "bx":None,
            "bz": None,
        }
    T = sc.to_gf2_tableau(tableau) #this also catches if T is a string

    #remove lost qubits, keep indices
    num_stab, m = T.shape
    num_qubits = m // 2
    _,_,logicals=sc.find_logical_op_basis(T,num_qubits)
    xlogi=sc.to_gf2_tableau(logicals[0])
    zlogi=sc.to_gf2_tableau(logicals[1]) #arbitrary designation

    assert m % 2 == 0, "Tableau length must be even"
    assert all(0 <= x <= num_qubits-1 for x in lost_qubits), f"Lost qubits can only contain qubits indices from 0 to {num_qubits-1}"
    assert g >= 1, "g must be at least 1"


    assert len(logicals)==2, f"There must only be one logical qubits, here there are {len(logicals)//2}."

    xlogi,x_indices=sc.find_clean_logical(T,xlogi,lost_qubits) #this already removes the lost qubits entirely
    zlogi,z_indices=sc.find_clean_logical(T,zlogi,lost_qubits)

    if xlogi is None or zlogi is None: #no clean logi exists, information destroyed
        return result
    
    T,_,indices=sc.remove_lost_qubits_from_tableau(T,lost_qubits) #indices to remember which qubits removed

    assert np.array_equal(x_indices, indices) and np.array_equal(z_indices, indices), "clean/reduce index mismatch"
    num_stab_remain, m_remain = T.shape
    num_qubits_remain = m_remain // 2

    xlogi = sc.to_gf2_tableau(np.asarray(xlogi).ravel())
    zlogi = sc.to_gf2_tableau(np.asarray(zlogi).ravel())
    assert xlogi.shape == (m_remain,), f"xlogi width {xlogi.shape} != ({m_remain},)"
    assert zlogi.shape == (m_remain,), f"zlogi width {zlogi.shape} != ({m_remain},)"

    assert sc.rank_F2(T) == num_stab_remain, \
    f"reduced tableau not full rank: rank {sc.rank_F2(T)} != {num_stab_remain} rows"


    if target_qubit is not None:
        assert target_qubit < num_qubits, f"Target qubit must be in range 0 - {num_qubits-1}"
           
        if target_qubit not in indices: #target qubit lost
            return result
        
        else: 
            target_reduced = int(np.where(indices == target_qubit)[0][0])

    for meas in measurements:
        assert len(meas) == m, f"measurement must have length {m}"
    if target_qubit is not None:
        assert target_qubit < num_qubits, "Target qubit not in the code"

    # Model
    model = cp_model.CpModel()

    # binary variable vector [B_x|B_z]
    bx = [model.NewBoolVar(f"bx_{i}") for i in range(num_stab_remain)]
    bz = [model.NewBoolVar(f"bz_{i}") for i in range(num_stab_remain)]

    bxt = [sum(bx[i] * int(T[i, j]) for i in range(num_stab_remain)) for j in range(m_remain)]
    bzt = [sum(bz[i] * int(T[i, j]) for i in range(num_stab_remain)) for j in range(m_remain)]

    # minimise the x logical
    xmod2_terms = []
    for j in range(m_remain):
        raw = int(xlogi[j]) + bxt[j]
        max_raw = 1 + sum(int(T[i, j]) for i in range(num_stab_remain))
        k = model.NewIntVar(0, max_raw // 2, f"kx_{j}")
        mod2_expr = raw - 2 * k
        model.Add(mod2_expr >= 0)
        model.Add(mod2_expr <= 1)
        xmod2_terms.append(mod2_expr)

    # minimise the z logical
    zmod2_terms = []
    for j in range(m_remain):
        raw = int(zlogi[j]) + bzt[j]
        max_raw = 1 + sum(int(T[i, j]) for i in range(num_stab_remain))
        k = model.NewIntVar(0, max_raw // 2, f"kz_{j}")
        mod2_expr = raw - 2 * k
        model.Add(mod2_expr >= 0)
        model.Add(mod2_expr <= 1)
        zmod2_terms.append(mod2_expr)

    
    xlogical_x_part = xmod2_terms[:num_qubits_remain]
    xlogical_z_part = xmod2_terms[num_qubits_remain:]
    zlogical_x_part=zmod2_terms[:num_qubits_remain]
    zlogical_z_part=zmod2_terms[num_qubits_remain:]
    
    if minimize_support:
        support = []
        for q in range(num_qubits_remain):
            sq = model.NewBoolVar(f"support_{q}")
            parts = [xlogical_x_part[q], xlogical_z_part[q],zlogical_x_part[q], zlogical_z_part[q]]
            for p in parts:
                model.Add(sq >= p)          # any part set -> sq = 1
            model.Add(sq <= sum(parts))     # all parts 0 -> sq = 0
            support.append(sq)

    #support was doubly counted before if qubit is supported in X AND Z logical. So solutions with large overlap 
    #were not found


    # measurement constraints X
    for i, meas in enumerate(measurements):
        meas_x = meas[:num_qubits_remain]
        meas_z = meas[num_qubits_remain:]

        raw = sum(
            int(meas_z[q]) * xlogical_x_part[q] + int(meas_x[q]) * xlogical_z_part[q]
            for q in range(num_qubits_remain)
        )

        k_comm = model.NewIntVar(0, num_qubits_remain, f"commx_{i}")
        model.Add(raw == 2 * k_comm)

    # measurement constraints Z
    for i, meas in enumerate(measurements):
        meas_x = meas[:num_qubits_remain]
        meas_z = meas[num_qubits_remain:]

        raw = sum(
            int(meas_z[q]) * zlogical_x_part[q]
            + int(meas_x[q]) * zlogical_z_part[q]
            for q in range(num_qubits_remain)
        )

        k_comm = model.NewIntVar(0, num_qubits_remain, f"commz_{i}")
        model.Add(raw == 2 * k_comm)


    # constraint for g-SPF

    anti_terms = []
    for q in range(num_qubits_remain):
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

        if target_qubit is not None and q==target_reduced:
           
            model.Add(anti_q == 1)

        model.Add(anti_q == anti_comm - 2 * k_anti)
        anti_terms.append(anti_q)

    # model.Add(sum(anti_terms) <= g)
    anti_sum = sum(anti_terms)
    model.Add(anti_sum <= g)

    k_total = model.NewIntVar(0, num_qubits_remain, "k_total_anti")
    model.Add(anti_sum == 2 * k_total + 1)

    # objective function #jelena: support does not need to be minimized if one only wants to find a pair for threshold
    if minimize_support:
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
        x=sc.to_gf2_tableau(x)
        z=sc.to_gf2_tableau(z)

        if not return_reduced_form:
            x=sc.restore_lost_qubits(x,x_indices,num_qubits)
            z=sc.restore_lost_qubits(z,z_indices,num_qubits)

        result = {
            "success": True,
            "status": status,
            "status_name": status_name,
            "objective": solver.ObjectiveValue() if minimize_support else None,
            "x": x,
            "z": z,
            "support_size": [solver.Value(v) for v in support] if minimize_support else None,
            "bx": np.array([solver.Value(bx[i]) for i in range(num_stab_remain)], dtype=int),
            "bz": np.array([solver.Value(bz[i]) for i in range(num_stab_remain)], dtype=int),
        }

     

    else:
        result = {
            "success": False,
            "status": status,
            "status_name": status_name,
            "objective": None,
            "x": xlogi,
            "z": zlogi,
            "support_size": None,
            "bx": None,
            "bz": None,
        }

    return result




def generalised_spf_logical_heuristic(tableau, lost_qubits: list,
                    target_qubit: int = None, anti_commut_iter: int = 100,
                    return_reduced_form: bool = False,
                    max_iter: int = 100):

    result = {
        "success": False,
        "status": None,
        "status_name": None,
        "objective": None,
        "x": None,
        "z": None,
        "support_size": None,
        "bx": None,
        "bz": None,
    }

    T = sc.to_gf2_tableau(tableau)
    num_stab, m = T.shape
    num_qubits = m // 2

    assert m % 2 == 0, "Tableau length must be even"
    assert all(0 <= x <= num_qubits - 1 for x in lost_qubits), \
        f"Lost qubits must be indices in 0..{num_qubits-1}"
    if target_qubit is not None:
        assert target_qubit < num_qubits, \
            f"Target qubit must be in range 0..{num_qubits-1}"

    _, _, logical_ops = sc.find_logical_op_basis(T, num_qubits)
    assert len(logical_ops) == 2, \
        f"Expected one logical qubit, got {len(logical_ops)//2}"
    logical_repr_commute = sc.to_gf2_tableau(logical_ops[0])
    logical_repr_anticommute=sc.to_gf2_tableau(logical_ops[1])

    logi_commute_clean, logi_indices = sc.find_clean_logical(T, logical_repr_commute, lost_qubits)
    logi_anticommute_clean,logi_indices=sc.find_clean_logical(T, logical_repr_anticommute, lost_qubits)
   

    if logi_commute_clean is None or logi_anticommute_clean is None:
        return result
    

    T_clean, _, indices = sc.remove_lost_qubits_from_tableau(T, lost_qubits)

    assert np.array_equal(logi_indices, indices), "clean/reduce index mismatch"

    num_stab_remain, m_remain = T_clean.shape
    num_qubits_remain = m_remain // 2

    """Omega = GF2(sc.construct_Omega_Matrix(num_qubits_remain).astype(np.int64))
    lc = GF2(np.asarray(logi_commute_clean).reshape(1,-1).astype(int))
    la = GF2(np.asarray(logi_anticommute_clean).reshape(-1,1).astype(int))
    print("logi_commute vs logi_anticommute symplectic:", int((lc @ Omega @ la).ravel()[0]))

    assert sc.rank_F2(T_clean) == num_stab_remain, \
        f"reduced tableau not full rank: {sc.rank_F2(T_clean)} != {num_stab_remain}"""

    target_reduced = None
    if target_qubit is not None:
        if target_qubit not in indices:
            return result
        target_reduced = int(np.where(indices == target_qubit)[0][0])

    _, _, reduced_logicals = sc.find_logical_op_basis(T_clean, num_qubits_remain)

    # ---- build accidental symplectic basis (REPLACES the old rank-filter loop) ----
    Omega = GF2(sc.construct_Omega_Matrix(num_qubits_remain).astype(np.int64))

    def sp(a, b):
        a = GF2(np.asarray(a).reshape(1, -1).astype(int))
        b = GF2(np.asarray(b).reshape(-1, 1).astype(int))
        return int((a @ Omega @ b).ravel()[0])

    Xy = GF2(np.asarray(logi_commute_clean).ravel().astype(int))
    Zy = GF2(np.asarray(logi_anticommute_clean).ravel().astype(int))

    acc_raw = []
    for lg in reduced_logicals:
        c = GF2(np.asarray(lg).ravel().astype(int))
        if sp(c, Zy) == 1:
            c = c + Xy
        if sp(c, Xy) == 1:
            c = c + Zy
        if np.asarray(c).any():
            acc_raw.append(c)

    acc_pairs = sc.symplectic_basis(acc_raw, num_qubits_remain)

    accidentals = []
    for (Xa, Za) in acc_pairs:
        accidentals.append(sc.to_gf2_tableau(np.asarray(Xa).ravel()))
        accidentals.append(sc.to_gf2_tableau(np.asarray(Za).ravel()))
    # ---- end accidental construction ----

    """# (optional) verification asserts
    full_set = GF2(np.vstack([
        np.asarray(T_clean).astype(int),
        np.asarray(logi_commute_clean).reshape(1, -1).astype(int),
        np.asarray(logi_anticommute_clean).reshape(1, -1).astype(int),
    ] + [np.asarray(a).reshape(1, -1).astype(int) for a in accidentals]))
    n_log = num_qubits_remain - sc.rank_F2(T_clean)
    assert sc.rank_F2(full_set) == num_stab_remain + 2*n_log, \
        f"appended set does not span full logical space"""

    short_first, channel_probs = dc.find_first_short_logical_in_coset(
        T_clean, logi_anticommute=logi_anticommute_clean, logi_commute=logi_commute_clean,
        accidental_logicals=accidentals, o=target_reduced, max_iter=max_iter)
    
    if short_first is None:
        return result
 

    """sf = np.asarray(short_first).ravel().astype(int)
    # the logical short_first should represent (the one it's pinned to):
    intended = np.asarray(logi_commute_clean).ravel().astype(int)   # or whichever it's meant to be
    diff = (sf + intended) % 2
    basis = GF2(np.vstack([np.asarray(T_clean).astype(int), diff.reshape(1, -1).astype(int)]))
    in_coset = sc.rank_F2(basis) == sc.rank_F2(T_clean)
    print("short_first in intended coset:", in_coset, " weight:", int(sf.sum()))"""
    
    """print('short first',ta.tableau2paulistring(short_first))
    print('logi_anti_commute_clean',ta.tableau2paulistring(logi_anticommute_clean))"""

    if target_reduced is not None:
        short_second = sc.find_anti_commuting_logi_at_O(T_clean, short_first,accidentals, target_reduced)
        if short_second is None:
            channel_probs = dc.make_given_logical_unlikely(channel_probs, num_qubits_remain, short_first)
            for i in range(anti_commut_iter):
                short_first, channel_probs = dc.find_first_short_logical_in_coset(T_clean, logi_anticommute=logi_anticommute_clean,\
                    logi_commute=logi_commute_clean,accidental_logicals=accidentals, o=target_reduced, max_iter=max_iter)

                if short_first is None:
                    short_second = None
                    break

                short_second = sc.find_anti_commuting_logi_at_O(T_clean, short_first,accidentals, target_reduced)
                if short_second is not None:
                    break
                else:
                    channel_probs = dc.make_given_logical_unlikely(channel_probs, num_qubits_remain, short_first)
    else:
        short_second = dc.find_short_second_logical(T_clean, short_first, max_iter=max_iter)

    if short_second is None:
        return result
    
    """# ---- commutation validation ----
    Omega_r = GF2(sc.construct_Omega_Matrix(num_qubits_remain).astype(np.int64))

    def _symp(a, b):
        a = GF2(np.asarray(a).reshape(1, -1).astype(int))
        b = GF2(np.asarray(b).reshape(-1, 1).astype(int))
        return int((a @ Omega_r @ b).ravel()[0])

    # short_first should anticommute with logi_anticommute_clean, commute with logi_commute_clean
    assert _symp(short_first, logi_anticommute_clean) == 1, \
        "short_first does NOT anticommute with the anticommute logical"
    assert _symp(short_first, logi_commute_clean) == 0, \
        "short_first does NOT commute with the commute logical"

    # short_second should commute with logi_commute_clean (it's the partner conjugate side)
    # and both should commute with every accidental logical
    for acc in accidentals:
        assert _symp(short_first, acc) == 0, \
            "short_first anticommutes with an accidental logical (coset not pinned)"
        assert _symp(short_second, acc) == 0, \
            "short_second anticommutes with an accidental logical (coset not pinned)"

    # the pair must anticommute at O and commute qubit-wise elsewhere (Box 1)
    if target_reduced is not None:
        sf_ = np.asarray(short_first).ravel().astype(int)
        ss_ = np.asarray(short_second).ravel().astype(int)
        for q in range(num_qubits_remain):
            anti_q = (sf_[q] & ss_[q+num_qubits_remain]) ^ (sf_[q+num_qubits_remain] & ss_[q])
            if q == target_reduced:
                assert anti_q == 1, f"pair does NOT anticommute at O ({target_reduced})"
            else:
                assert anti_q == 0, f"pair anticommutes at non-O qubit {q}"
    
    sf = np.asarray(short_first).ravel().astype(int)
    ss = np.asarray(short_second).ravel().astype(int)
    sf_supp = [q for q in range(num_qubits_remain) if sf[q] or sf[q+num_qubits_remain]]
    ss_supp = [q for q in range(num_qubits_remain) if ss[q] or ss[q+num_qubits_remain]]
    print("short_first support:", sf_supp, "short_second support:", ss_supp, "O:", target_reduced)"""

    x = sc.to_gf2_tableau(np.asarray(short_first).ravel())
    z = sc.to_gf2_tableau(np.asarray(short_second).ravel())


    if not return_reduced_form:
        x = sc.restore_lost_qubits(x, indices, num_qubits)
        z = sc.restore_lost_qubits(z, indices, num_qubits)

    result = {
        "success": True,
        "status": None,
        "status_name": "heuristic",
        "objective": None,
        "x": x,
        "z": z,
        "support_size": None,
        "bx": None,
        "bz": None,
    }
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
    
    # lost_qubits = ta.paulistring2tableau("Y0", numq)
    lost_qubits = ta.paulistring2tableau(s, numq)


    T = GF2(stabi)
    generalised_spf_logical(stabi, xlogi, zlogi, previous_meas, lost_qubits, gg, target_qubit=None)

   