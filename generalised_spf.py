import os
import numpy as np
import time
from galois import GF2
import tableau as ta
import stabiliser_code as sc
import networkx as nx
import decoder_methods as dc

from ortools.sat.python import cp_model


def gspf_ilp(tableau : np.ndarray, xlogical: list,
                    zlogical : list,
                    measurements : list[list], lost_qubits : list,
                    g : int,
                    target_qubit = None,
                    max_time = 600,
                    minimise_support = True):
    """
    Find a logical satisfying g-SPF algebra of GF2.

    Parameters
    ----------
    tableau : numpy array
        Tableau of the stabiliser code with stabilisers.
        The representation is X part then Z part ( X | Z ).

    xlogical : list
        The original X-logical that sets the afine space
        Can be X logical or Z logical.

    zlogical : list
        The original Z-logical that sets the afine space

    other_logical : list
        'other_logical' is the starting point of the other logical.
        If logical is X (Z) then other_logical has to be Z (X)

    measurements : list of list
        List of previously completed measurements.
        Each measuremnet in the list is a single qubit measurement
        represented in the tableau form ( X | Z ).

    lost_qubits : list
        List containing the index of lost qubits

    g : int
        The g in 'g-SPF'.

    target_qubit : int
        The required output qubit. Currently only of length 1, could be made a list.

    Returns
    -------
    A loss-tolerant representation of the 'logical' satisfying g-SPF.

    """

    T = ta.to_gf2_tableau(tableau)

    num_stab, m = T.shape
    num_qubits = m // 2
    assert m % 2 == 0, "Tableau length must be even"
    assert len(xlogical) == m, f"xlogical must have length {m}"
    assert len(zlogical) == m, f"zlogical must have length {m}"
    assert len(lost_qubits) <= num_qubits, f"lost_qubits must have length {m}"
    assert g >= 1, "g must be at least 1"

    for meas in measurements:
        assert len(meas) == m, f"measurement must have length {m}"
    lq = lost_qubits.copy()
    lost_qubits = [0]*2*num_qubits
    for ele in lq:
        lost_qubits[ele] = 1
        lost_qubits[ele+num_qubits] = 1

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
        raw = int(xlogical[j]) + bxt[j]
        max_raw = 1 + sum(int(T[i, j]) for i in range(num_stab))
        k = model.NewIntVar(0, max_raw // 2, f"kx_{j}")
        mod2_expr = raw - 2 * k
        model.Add(mod2_expr >= 0)
        model.Add(mod2_expr <= 1)
        xmod2_terms.append(mod2_expr)

    # minimise the z logical
    zmod2_terms = []
    for j in range(m):
        raw = int(zlogical[j]) + bzt[j]
        max_raw = 1 + sum(int(T[i, j]) for i in range(num_stab))
        k = model.NewIntVar(0, max_raw // 2, f"kz_{j}")
        mod2_expr = raw - 2 * k
        model.Add(mod2_expr >= 0)
        model.Add(mod2_expr <= 1)
        zmod2_terms.append(mod2_expr)


    xlogical_x_part = xmod2_terms[:num_qubits]
    xlogical_z_part = xmod2_terms[num_qubits:]
    zlogical_x_part = zmod2_terms[:num_qubits]
    zlogical_z_part = zmod2_terms[num_qubits:]
    support = []
    if minimise_support:
        for q in range(num_qubits):
            sq = model.NewBoolVar(f"support_{q}")
            parts = [xlogical_x_part[q], xlogical_z_part[q], zlogical_x_part[q], zlogical_z_part[q]]
            for p in parts:
                model.Add(sq >= p)          # any part set -> sq = 1
            model.Add(sq <= sum(parts))     # all parts 0 -> sq = 0
            support.append(sq)

    # support is counted doubly in this commented section
    # support = []
    # xlogical_x_part = xmod2_terms[:num_qubits]
    # xlogical_z_part = xmod2_terms[num_qubits:]
    # for q in range(num_qubits):
    #     sq = model.NewBoolVar(f"supportx_{q}")
    #     model.Add(sq >= xlogical_x_part[q])
    #     model.Add(sq >= xlogical_z_part[q])
    #     model.Add(sq <= xlogical_x_part[q] + xlogical_z_part[q])
    #     support.append(sq)

    # # support = []
    # zlogical_x_part = zmod2_terms[:num_qubits]
    # zlogical_z_part = zmod2_terms[num_qubits:]
    # for q in range(num_qubits):
    #     sq = model.NewBoolVar(f"supportz_{q}")
    #     model.Add(sq >= zlogical_x_part[q])
    #     model.Add(sq >= zlogical_z_part[q])
    #     model.Add(sq <= zlogical_x_part[q] + zlogical_z_part[q])
    #     support.append(sq)


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
    if minimise_support:
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


def initialise_logical_basis(tableau: np.ndarray):

    T = ta.to_gf2_tableau(tableau) # this also catches if T is a string
    num_qubits = T.shape[1]//2
    _, _, logicals = sc.find_logical_op_basis(T,num_qubits)

    #turn into symplectic basis
    symplectic_basis = ta.symplectic_basis(logicals)

    assert len(symplectic_basis) == 1, f"There must only be one logical qubits, \
          here there are {len(symplectic_basis[0])//2}."

    xlogical, zlogical = symplectic_basis[0] # initialise_logical_basis gives back a tuple
    xlogical = ta.to_gf2_tableau(xlogical)
    zlogical = ta.to_gf2_tableau(zlogical) # arbitrary designations

    return T, xlogical, zlogical


def update_T_and_logi_after_loss(T : np.ndarray, logicals : list, lost_qubits : list):

    op_indices=[]
    # removing lost qubits from logicals
    for i, l in enumerate(logicals):

        l, l_indices = sc.find_clean_logical(T, l, lost_qubits) #this already removes the lost qubits entirely
        logicals[i] = l
        op_indices.append(l_indices)

        if l is None:
            return T,logicals[0], logicals[1],None,False


    T = ta.to_gf2_tableau(T)
    T, _, indices = sc.remove_lost_qubits_from_tableau(T, lost_qubits, reduce = False) #indices to remember which qubits removed
    num_stab_remain, m_remain = T.shape
    xlogical_reduced = ta.to_gf2_tableau(np.asarray(logicals[0]).ravel())
    zlogical_reduced = ta.to_gf2_tableau(np.asarray(logicals[1]).ravel())

    # removing lost qubits from tableau

    assert all(np.array_equal(op_indices[j], indices) for j in range(len(logicals))), "clean/reduce index mismatch"
    assert xlogical_reduced.shape == (m_remain,), f"xlogical width {xlogical_reduced.shape} != ({m_remain},)"
    assert zlogical_reduced.shape == (m_remain,), f"zlogical width {zlogical_reduced.shape} != ({m_remain},)"
    assert sc.rank_F2(T) == num_stab_remain, f"reduced tableau not full rank: rank {sc.rank_F2(T)} != {num_stab_remain} rows"

    return T, xlogical_reduced, zlogical_reduced, indices, True


def update_measurements_after_losses(measurements, lost_qubits, len_meas):

    new_meas=[]
    lost_qubits_set = set(lost_qubits)
    for meas in measurements:
        assert len(meas) == len_meas, f"measurement must have length {len_meas}"
        assert not any(meas[qub] == 1 for qub in lost_qubits_set), (
            "measurement contains a lost qubit")
        meas = ta.to_gf2_tableau(meas)
        meas = sc.kick_out_qubits(meas, lost_qubits)
        if meas.any():
            new_meas.append(list(meas.ravel())) # making meas 1D-array, the shapeshifting is a bit of a mess

    return new_meas


def update_target_after_losses(target_qubit, indices, num_qubits):

    target_reduced = None
    if target_qubit is not None:
        assert target_qubit < num_qubits, f"Target qubit must be in range 0 - {num_qubits-1}"
        if target_qubit not in indices: # target qubit lost
            print("target qubit lost")
            return False
        else:
            target_reduced = int(np.where(indices == target_qubit)[0][0])
            return target_reduced
    else:
        return target_reduced




def generalised_spf_logical(tableau : np.ndarray,
                    measurements : list[list], lost_qubits : list,
                    g : int,
                    target_qubit:int = None,
                    correct_for_lost_indices:bool=True,
                    max_time:float = 600,
                    minimise_support:bool=True):
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
        Indices start from 0 up to n-1, where n is the total number of qubits.

    g : int
        The g in 'g-SPF'.

    target_qubit : int
        The required output qubit. Currently only of length 1, could be made a list.

    max_time : int
        The maximum amount of time in seconds the Solver runs for

    minimise_support : bool
        Default is True, where we minimise the support of X and Z logicals.
        Can be set to False so that we only find any two logicals that
        satify the constraints.

    Returns
    -------
    A loss-tolerant representation of the 'logical' satisfying g-SPF.

    """
    t1 = time.time()
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
    T = ta.to_gf2_tableau(tableau) # this also catches if T is a string

    if not isinstance(lost_qubits,list):
        if not isinstance(lost_qubits,np.ndarray):
            raise ValueError("lost_qubits must be list or numpy nd.array")
        else:
            lost_qubits=list(lost_qubits)

    #remove lost qubits, keep indices
    _, m = T.shape
    num_qubits = m // 2
    early_flag = True
    assert m % 2 == 0, "Tableau length must be even"
    assert all(0 <= x <= num_qubits-1 for x in lost_qubits), f"Lost qubits can only contain qubits indices from 0 to {num_qubits-1}"
    assert g >= 1, "g must be at least 1"

    T, xlogical, zlogical = initialise_logical_basis(T)

    res_update = update_T_and_logi_after_loss(T, [xlogical, zlogical], lost_qubits)
    T_reduced, xlogical_reduced, zlogical_reduced, indices, success = res_update
    if not success: # no clean logi exists, information destroyed
        return result, early_flag, time.time() - t1

    target_reduced = update_target_after_losses(target_qubit, indices, num_qubits)
    if target_reduced is False:
        print("target qubit lost")
        return result, early_flag, time.time() - t1

    meas_reduced = update_measurements_after_losses(measurements, lost_qubits, m)

    result = gspf_ilp(T_reduced, xlogical_reduced, zlogical_reduced,
                      meas_reduced,
                      lost_qubits = [],
                      g=g,
                      target_qubit=target_reduced,
                      max_time = max_time,
                      minimise_support = minimise_support)

    if correct_for_lost_indices:
        x = result['x']
        z = result['z']
        x=sc.restore_lost_qubits(x, indices, num_qubits)
        z=sc.restore_lost_qubits(z, indices, num_qubits)
        result['x'] = x
        result['z'] = z

    return result, False, None


def generalised_spf_logical_heuristic(tableau, lost_qubits: list,
                    target_qubit: int = None, anti_commut_iter: int = 100,
                    correct_for_lost_indices: bool = True,
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

    T = ta.to_gf2_tableau(tableau)
    _, m = T.shape
    num_qubits = m // 2

    assert m % 2 == 0, "Tableau length must be even"

    assert all(0 <= x <= num_qubits - 1 for x in lost_qubits), \
        f"Lost qubits must be indices in 0..{num_qubits-1}"

    if target_qubit is not None:
        assert target_qubit < num_qubits, \
            f"Target qubit must be in range 0..{num_qubits-1}"

    _, _, logical_ops = sc.find_logical_op_basis(T, num_qubits) #find a basis of logical operators

    assert len(logical_ops) == 2, \
        f"Expected one logical qubit, got {len(logical_ops)//2}"

    logical_repr_commute = ta.to_gf2_tableau(logical_ops[0])
    logical_repr_anticommute=ta.to_gf2_tableau(logical_ops[1])

    logi_commute_clean, logi_indices = sc.find_clean_logical(T, logical_repr_commute, lost_qubits)
    logi_anticommute_clean,logi_indices=sc.find_clean_logical(T, logical_repr_anticommute, lost_qubits)


    if logi_commute_clean is None or logi_anticommute_clean is None: #logical information destroyed
        return result


    T_clean, _, indices = sc.remove_lost_qubits_from_tableau(T, lost_qubits) #indices tells us which
    #indices of T_clean (which is a reduced tableau) corresponds to tthe old indices of T

    assert np.array_equal(logi_indices, indices), "clean/reduce index mismatch"

    _, m_remain = T_clean.shape #number of remaining stabilisers changes
    num_qubits_remain = m_remain // 2

    target_reduced = None
    if target_qubit is not None:
        if target_qubit not in indices:
            return result
        target_reduced = int(np.where(indices == target_qubit)[0][0])

    _, _, reduced_logicals = sc.find_logical_op_basis(T_clean, num_qubits_remain)
    #There is now a new basis of reduced logicals, which may be larger than the old one.
    #in order for the decoder to work this basis must be a symplectic basis

    num_logical_qubits_remain=len(reduced_logicals)//2
    # print('number of logical qubits', num_logical_qubits_remain)

    X_logical_qubit = ta.to_gf2_tableau(logi_commute_clean).ravel()
    Z_logical_qubit = ta.to_gf2_tableau(logi_anticommute_clean).ravel()

    additional_logicals=[]

    if num_logical_qubits_remain>1:

        #making the additional logicals commute with X and Z

        logical_pairs=ta.turn_into_symplectic_basis([X_logical_qubit,Z_logical_qubit],reduced_logicals)

        for (Xa, Za) in logical_pairs[1:]:
            additional_logicals.append(ta.to_gf2_tableau(Xa).ravel())
            additional_logicals.append(ta.to_gf2_tableau(Za).ravel())

    #-------------------------------------------------------------------------------------------------------#
    #find short logical that commutes with logi_commute and anti-commutes with logi_anti_commute_clean
    #passing a symplectic basis to decoder, otherwise there cannot be a solution!

    short_first, channel_probs = dc.find_first_short_logical_in_coset(
        T_clean, logi_anticommute=logi_anticommute_clean, logi_commute=logi_commute_clean,
        remaining_logicals=additional_logicals, o=target_reduced, max_iter=max_iter)

    if short_first is None: #decoder failed
        return result


    """The loop does the following: It tries to find an anti-commuting second logical with anti-commutation at O.
    That is a pure linear algebra method. Additional logicals is passed so that the anti-commuting logical is not
    one of the new additional logicals. If no solution is find, a new first logical is found. The BP decoder is intiialised
    with probabiltiies that make the previously found logical unlikely."""

    if target_reduced is not None:
        short_second = sc.find_anti_commuting_logi_at_O(T_clean, short_first,additional_logicals+[logi_commute_clean], target_reduced)
        if short_second is None:
            channel_probs = dc.make_given_logical_unlikely(channel_probs, num_qubits_remain, short_first)
            for i in range(anti_commut_iter):
                short_first, channel_probs = dc.find_first_short_logical_in_coset(T_clean, \
                logi_anticommute=logi_anticommute_clean,\
                logi_commute=logi_commute_clean,accidental_logicals=additional_logicals, o=target_reduced, max_iter=max_iter)

                if short_first is None:
                    short_second = None
                    break

                short_second = sc.find_anti_commuting_logi_at_O(T_clean, short_first,additional_logicals+[logi_commute_clean], target_reduced)
                if short_second is not None:
                    break
                else:
                    channel_probs = dc.make_given_logical_unlikely(channel_probs, num_qubits_remain, short_first)
    else:
        # print('finding short_second')
        short_second = dc.find_short_second_logical_in_coset(T_clean, short_first,Z_logical_qubit,additional_logicals, max_iter=max_iter)

    if short_second is None:
        return result


    short_first = ta.to_gf2_tableau(short_first).ravel()
    short_second = ta.to_gf2_tableau(short_second).ravel()


    if correct_for_lost_indices:
        x = sc.restore_lost_qubits(short_first, indices, num_qubits)
        z = sc.restore_lost_qubits(short_second, indices, num_qubits)

    result = {
        "success": True,
        "status": None,
        "status_name": "heuristic",
        "objective": None,
        "x": x,
        "z": z, #this designation is arbitrary
        "support_size": None,
        "bx": None,
        "bz": None,
    }
    return result



if __name__ == "__main__":
    from gspf_tests import test_gspf
    import time
    from code_importer import rotated_surface_code, surface_code
    # numq = 20
    # g = nx.erdos_renyi_graph(numq, 0.57)
    # # g = nx.cycle_graph(numq)
    # g = nx.to_numpy_array(g, dtype = np.uint16)
    # xlogiii, zlogiii, stabi = dc.create_graph_code(g)
    # print("xlogical input", ta.tableau2paulistring(xlogiii))
    # print("zlogical input", ta.tableau2paulistring(zlogiii))

    gg = 1


    _,_,H,xlogi, zlogi = surface_code(5)
    # stabi = GF2(H)

    stabi=ta.to_gf2_tableau(H)
    numq=stabi.shape[1]//2

    previous_meas = []
    # previous_meas = []
    previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]
    T = ta.to_gf2_tableau(stabi)
    numq=T.shape[1]//2
    lq=[4,3,1]
    # print('lq',lq)

    t1 = time.time()
    res =  generalised_spf_logical(T,previous_meas, lq, gg, target_qubit=None, minimise_support=True)
    print(time.time()-t1)

    # res = generalised_spf_logical(stabi, xlogiii, zlogi, previous_meas, lq, gg, target_qubit=None)
    # print(res)
    print("success new: ",res['success'])
    if res['success']:
        xlo = res["x"]
        zlo = res["z"]
        print("support_size", sum(res["support_size"]))
        print("xlogical output from gspf", ta.tableau2paulistring(xlo))
        print("zlogical output from gspf", ta.tableau2paulistring(zlo))
        # test_gspf(T, xlo, zlo, previous_meas, lq, g=gg)
        supp = sc.pair_support(res["x"], res["z"], None)
        print(len(supp))

    print("\n ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
    # t1 = time.time()
    # res =  gspf_ilp(T,xlogi, zlogi, previous_meas, lq, gg, target_qubit=7, minimise_support=False)
    # print(time.time()-t1)
    # print("success old: ",res['success'])
    # if res['success']:
    #     xlo = res["x"]
    #     zlo = res["z"]
    #     print("support_size", sum(res["support_size"]))
    #     print("xlogical output from gspf", ta.tableau2paulistring(xlo))
    #     print("zlogical output from gspf", ta.tableau2paulistring(zlo))
    #     test_gspf(T, xlo, zlo, previous_meas, lq, g=gg)


    t1 = time.time()
    res = generalised_spf_logical_heuristic(T, lq)
    print(time.time()-t1)
    if res['success']:
        xlo = res["x"]
        zlo = res["z"]
        print("xlogical output from Heuristic", ta.tableau2paulistring(xlo))
        print("zlogical output from Heuristic", ta.tableau2paulistring(zlo))
        test_gspf(T, xlo, zlo, previous_meas, lq, g =gg)
        supp = sc.pair_support(res["x"], res["z"], None)
        print(f"support = {len(supp)}")
    else:
        print('failure')
