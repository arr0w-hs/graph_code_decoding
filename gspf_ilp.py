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
    T = ta.to_gf2_tableau(tableau) #this also catches if T is a string

    if not isinstance(lost_qubits,list):
        if not isinstance(lost_qubits,np.ndarray):
            raise ValueError("lost_qubits must be list or numpy nd.array")
        else:
            lost_qubits=list(lost_qubits)

    #remove lost qubits, keep indices
    num_stab, m = T.shape
    num_qubits = m // 2
    _,_,logicals=sc.find_logical_op_basis(T,num_qubits)
    xlogi=ta.to_gf2_tableau(logicals[0])
    zlogi=ta.to_gf2_tableau(logicals[1]) #arbitrary designation

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

    xlogi = ta.to_gf2_tableau(np.asarray(xlogi).ravel())
    zlogi = ta.to_gf2_tableau(np.asarray(zlogi).ravel())
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

    if minimise_support:
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

    #removing measuements with support on lost qubits.
    #Note: This is only algebraically correct for single qubits measurements. Only this way the meas.
    #are still non-redundant, i.e. a full generating set

    new_meas=[]
    for m in measurements:
        m=ta.to_gf2_tableau(m)
        m= sc.kick_out_qubits(m,lost_qubits)
        if m.any():
            new_meas.append(m.ravel())#making meas 1D-array, the shapeshifting is a bit of a mess
        
        
    # measurement constraints X
    for i, meas in enumerate(new_meas):
       
     
        meas_x = meas[:num_qubits_remain]
        meas_z = meas[num_qubits_remain:]
         
        raw = sum(
            int(meas_z[q]) * xlogical_x_part[q] + int(meas_x[q]) * xlogical_z_part[q]
            for q in range(num_qubits_remain)
        )

        k_comm = model.NewIntVar(0, num_qubits_remain, f"commx_{i}")
        model.Add(raw == 2 * k_comm)

    # measurement constraints Z
    for i, meas in enumerate(new_meas):
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

        # target qubit constraints
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
    if minimise_support:
        model.Minimize(sum(support))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time
    solver.parameters.num_search_workers = min(8, os.cpu_count() or 1)
    status = solver.Solve(model)

    # print("Status:", solver.StatusName(status))
    status_name = solver.StatusName(status)
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print('solved')
        x = [solver.Value(v) for v in xmod2_terms]
        z = [solver.Value(v) for v in zmod2_terms]
        x=ta.to_gf2_tableau(x)
        z=ta.to_gf2_tableau(z)

        if not return_reduced_form:
            x=sc.restore_lost_qubits(x,x_indices,num_qubits)
            z=sc.restore_lost_qubits(z,z_indices,num_qubits)

        result = {
            "success": True,
            "status": status,
            "status_name": status_name,
            "objective": solver.ObjectiveValue() if minimise_support else None,
            "x": x,
            "z": z,
            "support_size": [solver.Value(v) for v in support] if minimise_support else None,
            "bx": np.array([solver.Value(bx[i]) for i in range(num_stab_remain)], dtype=int),
            "bz": np.array([solver.Value(bz[i]) for i in range(num_stab_remain)], dtype=int),
        }



    else:
        result = {
            "success": False,
            "status": status,
            "status_name": status_name,
            "objective": None,
            "x": None, #giving back None for failure
            "z": None,
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

    num_stab_remain, m_remain = T_clean.shape #number of remaining stabilisers changes
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

    Omega = ta.to_gf2_tableau(ta.construct_Omega_Matrix(num_qubits_remain))


    X_logical_qubit = ta.to_gf2_tableau(logi_commute_clean).ravel()
    Z_logical_qubit = ta.to_gf2_tableau(logi_anticommute_clean).ravel()

    if num_logical_qubits_remain>1:
        additional_logicals = []
        #making the additional logicals commute with X and Z
        for lg in reduced_logicals:

            c = ta.to_gf2_tableau(lg).ravel()

            if ta.sp(c, Z_logical_qubit) == 1:
                c = c + X_logical_qubit
            if ta.sp(c, X_logical_qubit) == 1:
                c = c + Z_logical_qubit

            if np.asarray(c).any(): #this makes sure to only add c to the additional logicals if it was not
                #equal to X_logical_qubit or Z_logical_qubit
                additional_logicals.append(c)

    additional_logical_pairs = sc.symplectic_basis(additional_logicals, num_qubits_remain)

    additional_logicals=[]
    for (Xa, Za) in additional_logical_pairs:
        additional_logicals.append(ta.to_gf2_tableau(Xa).ravel())
        additional_logicals.append(ta.to_gf2_tableau(Za).ravel())

    #-------------------------------------------------------------------------------------------------------#
    #find short logical that commutes with logi_commute and anti-commutes with logi_anti_commute_clean
    #passing a symplectic basis to decoder, otherwise there cannot be a solution!

    short_first, channel_probs = dc.find_first_short_logical_in_coset(
        T_clean, logi_anticommute=logi_anticommute_clean, logi_commute=logi_commute_clean,
        accidental_logicals=additional_logicals, o=target_reduced, max_iter=max_iter)

    if short_first is None: #decoder failed
        return result


    """The loop does the following: It tries to find an anti-commuting second logical with anti-commutation at O.
    That is a pure linear algebra method. Additional logicals is passed so that the anti-commuting logical is not
    one of the new additional logicals. If no solution is find, a new first logical is found. The BP decoder is intiialised
    with probabiltiies that make the previously found logical unlikely."""

    if target_reduced is not None:
        short_second = sc.find_anti_commuting_logi_at_O(T_clean, short_first,additional_logicals, target_reduced)
        if short_second is None:
            channel_probs = dc.make_given_logical_unlikely(channel_probs, num_qubits_remain, short_first)
            for i in range(anti_commut_iter):
                short_first, channel_probs = dc.find_first_short_logical_in_coset(T_clean, \
                logi_anticommute=logi_anticommute_clean,\
                logi_commute=logi_commute_clean,accidental_logicals=additional_logicals, o=target_reduced, max_iter=max_iter)

                if short_first is None:
                    short_second = None
                    break

                short_second = sc.find_anti_commuting_logi_at_O(T_clean, short_first,additional_logicals, target_reduced)
                if short_second is not None:
                    break
                else:
                    channel_probs = dc.make_given_logical_unlikely(channel_probs, num_qubits_remain, short_first)
    else:
        short_second = dc.find_short_second_logical(T_clean, short_first, max_iter=max_iter)

    if short_second is None:
        return result


    short_first = ta.to_gf2_tableau(short_first).ravel()
    short_second = ta.to_gf2_tableau(short_second).ravel()


    if not return_reduced_form:
        x = sc.restore_lost_qubits(x, indices, num_qubits)
        z = sc.restore_lost_qubits(z, indices, num_qubits)

    result = {
        "success": True,
        "status": None,
        "status_name": "heuristic",
        "objective": None,
        "x": short_first,
        "z": short_second, #this designation is arbitrary
        "support_size": None,
        "bx": None,
        "bz": None,
    }
    return result



if __name__ == "__main__":
    from test_suite import test_gspf
    # print()
    numq = 20
    g = nx.erdos_renyi_graph(numq, 0.7)
    # g = nx.cycle_graph(numq)
    g = nx.to_numpy_array(g, dtype = np.uint16)

    xlogi, zlogi, stabi = dc.create_graph_code(g)

    #numq -= 1 #jelena: idk if this number of qubits correspondence is correct?
    gg = 1
    
    stabi=ta.to_gf2_tableau(stabi)
    numq=stabi.shape[1]//2

    previous_meas = ["Z2", "X2","X6"]

    print(previous_meas)
    # previous_meas = ["Z1", "X2"]
    previous_meas = [ta.paulistring2tableau(ele, numq) for ele in previous_meas]
    

    """lost_qubits = np.unique(np.random.randint(0, numq, numq//3))
    # lost_qubits = []
    s = str()
    for ele in lost_qubits:
        s += "Y"+str(ele)+"*"
    s = s[:-1]
    print("lost_qubits: ", lost_qubits, s)"""
    # lost_qubits = ta.paulistring2tableau("Y0", numq)
    #lost_qubits = ta.paulistring2tableau(s, numq)


    T = ta.to_gf2_tableau(stabi)
    numq=T.shape[1]//2
    print('number of qubits')
    lost_qubits=[10,4,5]
    t=0
    res =  generalised_spf_logical(T,previous_meas, lost_qubits, gg,target_qubit = t, minimise_support=True)
    print('lost_qubits',lost_qubits)
    # res = generalised_spf_logical(stabi, xlogi, zlogi, previous_meas, lost_qubits, gg, target_qubit=None)
    # print(res)
    print("success: ",res['success'])
    xlo = res["x"]
    zlo = res["z"]
    if res['success']:
        print("xlogical output from gspf", ta.tableau2paulistring(xlo))
        print("zlogical output from gspf", ta.tableau2paulistring(zlo))
