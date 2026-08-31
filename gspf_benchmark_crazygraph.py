"""
Benchmark g-SPF (ILP) against the g-SPF heuristic on graph codes, swept over
code size and loss probability. See FAMILY / SIZES / PROBS in the config block.

Edit the imports if module names differ:
  - solvers come from `generalised_spf`
  - create_graph_code comes from `stabiliser_code`
"""

import os
import time
import random
import itertools
import contextlib

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import tableau as ta
from stabiliser_code import create_graph_code
from generalised_spf import (
    generalised_spf_logical,
    generalised_spf_logical_heuristic,
)
# sub-modules whose routines the heuristic calls -- imported so they can be
# wrapped for phase timing / exhaustion counting (see instrumentation below).
import stabiliser_code as _sc
import decoder_methods as _dc
import generalised_spf as _gspf_mod


# ============================ configuration =================================
# FAMILY selects the graph code family (see GRAPH_FAMILIES below). Options:
#   "crazy_graph", "square_lattice", "hexagonal_lattice",
#   "triangular_lattice", "tree_to_tree_graph", "random_graph",
#   "surface_code" (special-cased: not a graph code).
FAMILY = "triangular_lattice"
# Each SIZES entry is the pair of size parameters for the chosen family:
# (width, length) for the lattice families, (branches, depth) for
# tree_to_tree_graph, (num_nodes, edge_prob) for random_graph, (L, _) for
# surface_code (second element ignored). Qubit count grows along this list.
#   e.g. for random_graph, sweep size at fixed density:
#   SIZES = [(8, 0.5), (11, 0.5), (14, 0.5), (18, 0.5), (22, 0.5), (26, 0.5)]
#   e.g. for surface_code:  SIZES = [(3, 0), (5, 0), (7, 0), (9, 0)]
#SIZES = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
#SIZES=[(3, 0), (5, 0), (7, 0)]
#SIZES = [(3,3),(4,4),(5,5),(6,6),(7,7)]
SIZES = [(2, 2),  (3, 3), (4, 4),  (5, 5),(6,6)]
SIZES = [(2, 2),  (3, 3), (4, 4),]
N_TRIALS       = 50     # random loss patterns per (size, p) cell
# Loss probability grid. The sweep runs every size at every p, so cost scales as
# len(SIZES) * len(PROBS) * N_TRIALS. Trim any of the three to shorten it.
PROBS          = [round(x, 3) for x in np.linspace(0.3, 0.5, 5)]
# Which slice each plot takes from the 2-D (size, p) grid:
PROB_FOR_SIZE_PLOT = 0.1  # size-axis plot uses the p in PROBS nearest this
SIZE_FOR_PROB_PLOT = None    # prob-axis plot uses this (w, ell); None -> largest
# TARGET_QUBIT: None -> no target, endpoints eligible for loss (only the input
#   node is structurally excluded, see below). "output" -> target the output
#   (readout) qubit of each size; that qubit's node is protected from loss.
#   int -> a fixed target qubit index; that qubit's node is protected.
TARGET_QUBIT   = None
G              = 1# the g in g-SPF
MINIMISE_SUPPORT = True # keep True: this is where the ILP pays its cost
CACHE_LOGICAL_BASIS = True  # compute the full-T logical basis once per size
                            # and reuse it across trials (both solvers)
# FAMILY = "bivariate_bicycle"
# #SIZES  = [(6, 6), (6, 12),(12,12)]    # n = 72, 144, 288
# #SIZES  = [(6, 6), (9, 6),(6,12)]  #(9,6) strictly speaking not the same code family
# SIZES = [(3,3), (3,6),(6,6),(9, 6),(6,12)]
# #   TARGET_QUBIT = None      # REQUIRED: BB is k=12; target path is k=1-only

PROBS = [0.05, 0.10]     #  keep p low so logicals survive (high p -> mass failure, meaningless timing)
PROBS = np.linspace(0,1,5)
N_TRIALS = 10
MAX_TIME = 100        # BB ILP is slow; give it room, but it censors
CACHE_LOGICAL_BASIS = True
COMPARE_ON_BOTH_SUCCEEDED = True

# NOTE: PROB_FOR_SIZE_PLOT should be one of PROBS (e.g. 0.10) so the size plot
# has data. With n=288 and MAX_TIME=120 the largest cell can take a long time;
# start with SIZES=[(6,6),(6,12)] to smoke-test before adding (12,12).
# ---------------------------------------------------------------------------
MAX_TIME       = 120    # ILP solver time cap in seconds (censors long solves)
SEED           = 0       # RNG seed for reproducible loss patterns
QUIET          = True  # suppress the solvers' own print output
FORCE_SINGLE_THREAD_ILP = False  # pin CP-SAT to 1 worker for fair timing
LOG_Y          = True    # log-scale the runtime axis to show scaling
# Compare ILP vs heuristic only on trials where BOTH returned a solution, so the
# timing curves are averaged over identical instances (recommended when g>1,
# where the ILP and the g=1-style heuristic have different feasible sets).
COMPARE_ON_BOTH_SUCCEEDED = True
OUT_PNG_SIZE   = f"gspf_timing_vs_size_{FAMILY}_{G}.png"
OUT_PNG_SIZE_ALLP = f"gspf_timing_vs_size_allp_{FAMILY}_{G}.png"
OUT_PNG_PROB   = f"gspf_timing_vs_prob_{FAMILY}_{G}.png"
OUT_PNG_SUCC   = f"gspf_success_rate_{FAMILY}_{G}.png"
# Cost note: worst-case wall time ~ sum over sizes of
# N_TRIALS * (MAX_TIME + heuristic_time). Large sizes that hit MAX_TIME
# dominate; lower N_TRIALS or MAX_TIME to shorten the sweep.
# ============================================================================


# Preserve the real os.cpu_count so the single-thread pin can be toggled.
_REAL_CPU_COUNT = os.cpu_count


def set_single_thread_ilp(enabled):
    """Pin CP-SAT to one worker (enabled=True) or restore normal core use.

    gspf_ilp computes num_search_workers = min(8, os.cpu_count() or 1) at solve
    time. Reporting os.cpu_count() as 1 forces a single worker; restoring the
    real value lets CP-SAT use up to 8 workers, as originally written. Only the
    worker count is affected; solver results are unchanged.
    """
    os.cpu_count = (lambda: 1) if enabled else _REAL_CPU_COUNT


# apply the configured default at import time (CLI can override in __main__)
set_single_thread_ilp(FORCE_SINGLE_THREAD_ILP)


# ---- heuristic phase instrumentation ---------------------------------------
# The heuristic lives in another module and is timed as a black box. To split
# its wall time into setup (logical-basis / clean / symplectic) vs coset search
# (BP + anti-commuting), and to count decoder failures, we wrap the sub-routines
# it calls. A None return from a decoder means it failed its validity check (no
# support on the target / syndrome not reproduced / all-zero), NOT that max_iter
# was exhausted -- max_iter caps BP's internal rounds, after which OSD still
# returns a result. Accumulation only happens while _HEUR_STATE["active"] is
# True, so the ILP's use of the same routines is not counted.
_HEUR_STATE = {"active": False}
_HEUR_ACCUM = {}          # label -> {"time", "calls", "none"}

_SETUP_LABELS  = ("basis", "clean", "symplectic")
_SEARCH_LABELS = ("bp_first", "bp_second", "anti_second")


def _install_probe(module, name, label, track_none=False):
    try:
        orig = getattr(module, name)
    except AttributeError:
        print(f"[probe] skipped {getattr(module, '__name__', module)}.{name} "
              f"(not found)")
        return

    def probe(*args, **kwargs):
        if not _HEUR_STATE["active"]:
            return orig(*args, **kwargs)
        t0 = time.perf_counter()
        out = orig(*args, **kwargs)
        dt = time.perf_counter() - t0
        rec = _HEUR_ACCUM.setdefault(label, {"time": 0.0, "calls": 0, "none": 0})
        rec["time"] += dt
        rec["calls"] += 1
        if track_none and out is None:
            rec["none"] += 1
        return out

    setattr(module, name, probe)


# setup phase
# -- full-T logical-basis cache: the basis of the unclean tableau T is identical
# across all trials at a size, so compute it once and serve copies. Installed
# BEFORE the probe so the probe times the cached (fast) call. Returns copies so
# callers that mutate the logicals list (update_T_and_logi_after_loss) can't
# corrupt the cached value. T_clean bases don't match the key and recompute.
_REAL_FIND_BASIS = _sc.find_logical_op_basis
_BASIS_CACHE = {"key": None, "value": None}


def _basis_key(tableau_matrix):
    arr = np.asarray(ta.to_gf2_tableau(tableau_matrix)).astype(np.uint8)
    return (arr.shape, arr.tobytes())


def _copy_basis(value):
    def cp(item):
        return None if item is None else [a.copy() for a in item]
    x, z, logi = value
    return (cp(x), cp(z), cp(logi))


def _cached_find_logical_op_basis(tableau_matrix, n_qubits, CSS=False):
    if (CACHE_LOGICAL_BASIS and not CSS
            and _BASIS_CACHE["key"] is not None
            and _basis_key(tableau_matrix) == _BASIS_CACHE["key"]):
        return _copy_basis(_BASIS_CACHE["value"])
    return _REAL_FIND_BASIS(tableau_matrix, n_qubits, CSS=CSS)


def _prime_basis_cache(tableau_matrix, n_qubits):
    """Compute the full-T basis once and store it (used both solvers)."""
    if not CACHE_LOGICAL_BASIS:
        return
    _BASIS_CACHE["value"] = _REAL_FIND_BASIS(tableau_matrix, n_qubits)
    _BASIS_CACHE["key"] = _basis_key(tableau_matrix)


_sc.find_logical_op_basis = _cached_find_logical_op_basis

_install_probe(_sc, "find_logical_op_basis", "basis")
_install_probe(_gspf_mod, "update_T_and_logi_after_loss", "clean")
_install_probe(ta, "turn_into_symplectic_basis", "symplectic")
# coset-search phase (track_none: None return = decoder failed its validity
# check -- no support on target / syndrome not reproduced / all-zero)
_install_probe(_dc, "find_first_short_logical_in_coset", "bp_first",
               track_none=True)
_install_probe(_dc, "find_short_second_logical_in_coset", "bp_second",
               track_none=True)
_install_probe(_sc, "find_anti_commuting_logi_at_O", "anti_second",
               track_none=True)


def _run_heuristic_probed(fn):
    """Run fn (the heuristic call) with phase accumulation on; return the
    result tuple from _time_call plus a snapshot of the phase breakdown."""
    _HEUR_ACCUM.clear()
    _HEUR_STATE["active"] = True
    try:
        res, elapsed, crashed = _time_call(fn)
    finally:
        _HEUR_STATE["active"] = False
    phases = {k: dict(v) for k, v in _HEUR_ACCUM.items()}
    return res, elapsed, crashed, phases
# ----------------------------------------------------------------------------


# --------- graph builders + loss sampler, copied from the builders file -------
def crazy_graph(width, length, output_node=True):
    graph = nx.Graph()
    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]
    graph.add_edges_from((0, node) for node in layers[0])
    for left_layer, right_layer in zip(layers, layers[1:]):
        graph.add_edges_from(itertools.product(left_layer, right_layer))
    if output_node:
        output = width * length + 1
        graph.add_edges_from((node, output) for node in layers[-1])
    return graph


def square_lattice(width, length, output_node=True):
    graph = nx.Graph()
    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]
    graph.add_edges_from((0, node) for node in layers[0])
    for layer in layers:                       # vertical edges within a layer
        graph.add_edges_from(zip(layer, layer[1:]))
    for left_layer, right_layer in zip(layers, layers[1:]):  # between layers
        graph.add_edges_from(zip(left_layer, right_layer))
    if output_node:
        output = width * length + 1
        graph.add_edges_from((node, output) for node in layers[-1])
    return graph


def hexagonal_lattice(width, length, output_node=True):
    graph = nx.Graph()
    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]
    graph.add_edges_from((0, node) for node in layers[0])
    for layer_number, layer in enumerate(layers, start=1):   # brickwork
        for row in range(width - 1):
            if (row + layer_number) % 2 == 1:
                graph.add_edge(layer[row], layer[row + 1])
    for left_layer, right_layer in zip(layers, layers[1:]):
        graph.add_edges_from(zip(left_layer, right_layer))
    if output_node:
        output = width * length + 1
        graph.add_edges_from((node, output) for node in layers[-1])
    return graph


def triangular_lattice(width, length, alternate_diagonals=True, output_node=True):
    graph = nx.Graph()
    layers = [
        list(range(layer * width + 1, (layer + 1) * width + 1))
        for layer in range(length)
    ]
    graph.add_edges_from((0, node) for node in layers[0])
    for layer in layers:
        graph.add_edges_from(zip(layer, layer[1:]))
    for layer_number, (left_layer, right_layer) in enumerate(
            zip(layers, layers[1:]), start=2):
        graph.add_edges_from(zip(left_layer, right_layer))       # straight
        if alternate_diagonals and layer_number % 2 == 1:        # diagonals
            graph.add_edges_from(zip(left_layer[:-1], right_layer[1:]))
        else:
            graph.add_edges_from(zip(left_layer[1:], right_layer[:-1]))
    if output_node:
        output = width * length + 1
        graph.add_edges_from((node, output) for node in layers[-1])
    return graph


def tree_to_tree_graph(branches, depth, output_node=True):
    # output_node accepted for a uniform builder signature; this construction
    # already ends in a single merged output node, so the flag is unused.
    graph = nx.Graph()
    next_node = 1
    current_layer = [0]
    for _ in range(depth):                     # first tree: expand outward
        new_layer = []
        for parent in current_layer:
            children = list(range(next_node, next_node + branches))
            next_node += branches
            graph.add_edges_from((parent, child) for child in children)
            new_layer.extend(children)
        current_layer = new_layer
    for _ in range(depth):                     # second tree: merge inward
        new_layer = []
        for start in range(0, len(current_layer), branches):
            children = current_layer[start:start + branches]
            parent = next_node
            next_node += 1
            graph.add_edges_from((child, parent) for child in children)
            new_layer.append(parent)
        current_layer = new_layer
    return graph


def random_graph(num_nodes, edge_prob, output_node=True):
    """Erdos-Renyi random graph code. SIZES entries are (num_nodes, edge_prob):
    the first grows the qubit count, the second is the edge density (the axis
    that stresses the ILP). Regenerated with a deterministic seed derived from
    SEED and the parameters until the graph is connected and node 0 has a
    neighbour -- create_graph_code requires node 0 to be linked, and the
    heuristic assumes one logical qubit (k=1), which needs independent
    stabilisers, i.e. a connected graph. output_node is unused (accepted for a
    uniform builder signature). Each size is a single random instance; there is
    no averaging over graphs within a size."""
    n = int(num_nodes)
    p_edge = float(edge_prob)
    base = (SEED * 1000003 + n * 131 + int(round(p_edge * 1000))) % (2 ** 31)
    for attempt in range(2000):
        G = nx.erdos_renyi_graph(n, p_edge, seed=base + attempt)
        if nx.is_connected(G) and G.degree(0) > 0:
            return G
    raise RuntimeError(
        f"random_graph: no connected graph with node 0 linked after retries "
        f"(n={n}, edge_prob={p_edge}); raise edge_prob or num_nodes")

BB_FAMILY_A = {
    # (l, m): (A_terms, B_terms)   -- IBM-style series, k=12, treewidth Theta(n)
    (6, 6):  ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)]),
    (9, 6):  ([('x',3),('y',1),('y',2)], [('y',3),('x',1),('x',2)]),
    (6, 12): ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)]),
    (12, 12):([('x', 3), ('y', 2), ('y', 7)], [('y', 3), ('x', 1), ('x', 2)]),
    (12, 6): ([('x', 3), ('y', 1), ('y', 2)], [('y', 3), ('x', 1), ('x', 2)]),
    (3, 3): ([('x',3),('y',1),('y',2)], [('y',3),('x',1),('x',2)]),   # n=18, k=8
    (3, 6): ([('x',3),('y',1),('y',2)], [('y',3),('x',1),('x',2)]),   # n=36, k=8
}


def _bb_shift(N):
    S = np.zeros((N, N), dtype=int)
    for i in range(N):
        S[i, (i + 1) % N] = 1
    return S


def bb_code_tableau(l, m, A_terms, B_terms):
    """Bivariate bicycle code -> (X|Z) full-rank stabiliser tableau (GF2).
    n = 2*l*m data qubits, weight-6 checks, CSS. k=12 for Family A."""
    Il, Im = np.eye(l, dtype=int), np.eye(m, dtype=int)
    Sl, Sm = _bb_shift(l), _bb_shift(m)
    xp = lambda k: np.kron(np.linalg.matrix_power(Sl, k % l) % 2, Im) % 2
    yp = lambda k: np.kron(Il, np.linalg.matrix_power(Sm, k % m) % 2) % 2
    def bld(ts):
        M = np.zeros((l * m, l * m), dtype=int)
        for v, k in ts:
            M = (M + (xp(k) if v == 'x' else yp(k))) % 2
        return M
    A, B = bld(A_terms), bld(B_terms)
    HX = np.hstack([A, B]) % 2
    HZ = np.hstack([B.T, A.T]) % 2
    n = HX.shape[1]
    zero = np.zeros_like(HX)
    T = np.vstack([np.hstack([HX, zero]), np.hstack([zero, HZ])]).astype(int)
    T = ta.to_gf2_tableau(T).row_reduce()
    T = T[np.any(np.asarray(T), axis=1)]
    return ta.to_gf2_tableau(T)

# Family selector: FAMILY picks which builder build_code uses. Each builder
# takes two size parameters. For the lattice families that is (width, length)
# and qubit count grows with width*length; for tree_to_tree_graph it is
# (branches, depth); for random_graph it is (num_nodes, edge_prob). SIZES
# entries are interpreted accordingly.
GRAPH_FAMILIES = {
    "crazy_graph":        crazy_graph,
    "square_lattice":     square_lattice,
    "hexagonal_lattice":  hexagonal_lattice,
    "triangular_lattice": triangular_lattice,
    "tree_to_tree_graph": tree_to_tree_graph,
    "random_graph":       random_graph,
}


def sample_lost_qubits(num_qubits, loss_probability, rng, protected=None):
    """Each of the num_qubits tableau qubits is lost independently with prob p.
    Returns a list of qubit indices. `protected`, if given, is a qubit index
    kept out of the sample (used to protect a target qubit when one is set)."""
    return [q for q in range(num_qubits)
            if q != protected and rng.random() < loss_probability]


def resolve_target(num_qubits):
    """Map the TARGET_QUBIT setting to a concrete tableau qubit index or None."""
    if TARGET_QUBIT is None:
        return None
    if TARGET_QUBIT == "output":
        return num_qubits - 1          # output/readout qubit of this size
    target = int(TARGET_QUBIT)
    assert 0 <= target < num_qubits, \
        f"TARGET_QUBIT {target} out of range 0..{num_qubits - 1}"
    return target
# ----------------------------------------------------------------------------


@contextlib.contextmanager
def _maybe_silence(quiet):
    if not quiet:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def _time_call(fn):
    """Run fn(), returning (result_dict, elapsed_seconds, crashed_flag)."""
    t0 = time.perf_counter()
    try:
        res = fn()
        crashed = False
    except Exception as exc:            # noqa: BLE001 - report and continue
        res = {"success": False, "error": repr(exc)}
        crashed = True
    elapsed = time.perf_counter() - t0
    return res, elapsed, crashed


def build_code(a, b):
    if FAMILY == "surface_code":
        from code_importer import surface_code
        _, _, H, _, _ = surface_code(int(a))
        T = ta.to_gf2_tableau(H)
        return T, T.shape[1] // 2
    if FAMILY == "bivariate_bicycle":
        # SIZES entries are (l, m); the code is looked up in BB_FAMILY_A.
        key = (int(a), int(b))
        if key not in BB_FAMILY_A:
            raise KeyError(f"no BB Family A entry for (l,m)={key}; "
                           f"available: {sorted(BB_FAMILY_A)}")
        A_terms, B_terms = BB_FAMILY_A[key]
        T = bb_code_tableau(int(a), int(b), A_terms, B_terms)
        return T, T.shape[1] // 2
    G = GRAPH_FAMILIES[FAMILY](a, b)
    adj = nx.to_numpy_array(G, nodelist=sorted(G.nodes()), dtype=np.uint16)
    _, _, stabi = create_graph_code(adj)
    T = ta.to_gf2_tableau(stabi)
    num_qubits = T.shape[1] // 2
    return T, num_qubits


def _phase_split(heur_phases):
    """Return (setup_time, search_time) for one heuristic call's phase dict."""
    setup = sum(heur_phases.get(l, {}).get("time", 0.0) for l in _SETUP_LABELS)
    search = sum(heur_phases.get(l, {}).get("time", 0.0) for l in _SEARCH_LABELS)
    return setup, search


def run_benchmark():
    rng = random.Random(SEED)
    records = []

    for (w, ell) in SIZES:
        T, num_qubits = build_code(w, ell)
        target = resolve_target(num_qubits)
        _prime_basis_cache(T, num_qubits)   # full-T basis computed once here

        for p in PROBS:

            for trial in range(N_TRIALS):
                # every tableau qubit is lost independently with probability p
                # (the target qubit, if one is set, is kept out of the sample)
                lost = sample_lost_qubits(num_qubits, p, rng, target)

# --- g-SPF heuristic ---
                def _heur():
                    return generalised_spf_logical_heuristic(
                        T, lost, G, target_qubit=target,
                    )

                with _maybe_silence(QUIET):
                    heur_res, heur_time, heur_crashed, heur_phases = \
                        _run_heuristic_probed(_heur)

                # dedented: outside the `with`, so not silenced
                print(f"  [{w}x{ell} p={p:.2f} trial {trial+1}/{N_TRIALS}] "
                      f"heur {heur_time:7.3f}s "
                      f"({'ok' if heur_res.get('success') else 'fail'})",
                      flush=True)

                setup_t, search_t = _phase_split(heur_phases)

                # --- g-SPF (ILP) ---
                def _ilp():
                    res, _, _ = generalised_spf_logical(
                        T, [], lost, G,
                        target_qubit=target,
                        max_time=MAX_TIME,
                        minimise_support=MINIMISE_SUPPORT,
                    )
                    return res

                with _maybe_silence(QUIET):
                    ilp_res, ilp_time, ilp_crashed = _time_call(_ilp)

                # dedented: outside the `with`
                print(f"  [{w}x{ell} p={p:.2f} trial {trial+1}/{N_TRIALS}] "
                      f"ILP {ilp_time:7.1f}s "
                      f"({'ok' if ilp_res.get('success') else 'FAIL'}"
                      f"{'/cap' if ilp_time >= MAX_TIME-1 else ''})"
                      f"  | ratio {ilp_time/heur_time:6.1f}x"
                      if heur_time > 0 else "",
                      flush=True)

                # joint support of the returned (x, z) pair, via pair_support
                def _supp(res):
                    if not res.get("success"):
                        return None
                    s = _sc.pair_support(res.get("x"), res.get("z"), target, True)
                    return None if (s is None or s == "Nan") else int(s)

                ilp_supp  = _supp(ilp_res)
                heur_supp = _supp(heur_res)

                records.append({
                    "width": w, "length": ell,
                    "size_key": (w, ell),
                    "num_qubits": num_qubits,
                    "loss_prob": float(p),
                    "num_lost": len(lost),
                    "lost_qubits": list(lost),
                    "target": target,
                    "g": G,
                    # --- ILP ---
                    "ilp_success":  bool(ilp_res.get("success", False)),
                    "ilp_optimal":  ilp_res.get("status_name") == "OPTIMAL",
                    "ilp_status":   ilp_res.get("status_name"),
                    "ilp_time":     ilp_time,
                    "ilp_capped":   ilp_time >= MAX_TIME - 1,
                    "ilp_crashed":  ilp_crashed,
                    "ilp_objective": ilp_res.get("objective"),
                    "ilp_support":  ilp_supp,
                    "ilp_support_size_vec": (list(map(int, ilp_res["support_size"]))
                                             if ilp_res.get("support_size") is not None
                                             else None),
                    "ilp_x": (list(map(int, np.asarray(ilp_res["x"]).ravel()))
                              if ilp_res.get("success") and ilp_res.get("x") is not None
                              else None),
                    "ilp_z": (list(map(int, np.asarray(ilp_res["z"]).ravel()))
                              if ilp_res.get("success") and ilp_res.get("z") is not None
                              else None),
                    # --- heuristic ---
                    "heur_success": bool(heur_res.get("success", False)),
                    "heur_time":    heur_time,
                    "heur_setup":   setup_t,
                    "heur_search":  search_t,
                    "heur_crashed": heur_crashed,
                    "heur_support": heur_supp,
                    "heur_x": (list(map(int, np.asarray(heur_res["x"]).ravel()))
                               if heur_res.get("success") and heur_res.get("x") is not None
                               else None),
                    "heur_z": (list(map(int, np.asarray(heur_res["z"]).ravel()))
                               if heur_res.get("success") and heur_res.get("z") is not None
                               else None),
                    "heur_phases":  heur_phases,
                })

            cell = [r for r in records
                    if r["width"] == w and r["length"] == ell
                    and r["loss_prob"] == float(p)]
            n_ilp  = sum(r["ilp_success"] for r in cell)
            n_opt  = sum(r["ilp_optimal"] for r in cell)
            n_heur = sum(r["heur_success"] for r in cell)
            print(f"{w}x{ell}  n={num_qubits:>3}  p={p:4.2f} | "
                  f"ILP ok {n_ilp:>3}/{N_TRIALS} (opt {n_opt:>3}) | "
                  f"heur ok {n_heur:>3}/{N_TRIALS}")

    return records


def _mean_std(values):
    if values:
        return float(np.mean(values)), float(np.std(values))
    return np.nan, np.nan


def _nearest_prob(target_p):
    return min(PROBS, key=lambda p: abs(p - target_p))


def _series_over(records, group_values, group_key, value_key, success_key=None,
                 both=False):
    """Mean/std of value_key over records grouped by group_key == each value.
    If both is True, average only over records where ILP and heuristic both
    succeeded (identical-instance comparison). Otherwise, if success_key is
    given, average only over records with that flag True."""
    means, stds = [], []
    for gv in group_values:
        sel = []
        for r in records:
            if r[group_key] != gv:
                continue
            if both:
                if not (r["ilp_success"] and r["heur_success"]):
                    continue
            elif success_key is not None and not r[success_key]:
                continue
            sel.append(r)
        m, s = _mean_std([r[value_key] for r in sel])
        means.append(m)
        stds.append(s)
    return np.array(means), np.array(stds)


def build_or_lookup_n(records, size_key):
    for r in records:
        if r["size_key"] == size_key:
            return r["num_qubits"]
    return float("nan")

import json, pickle, datetime

def save_all_data(records, tag=None):
    """Dump every trial's full data to disk: a pickle (exact, all fields incl.
    heur_phases) and a JSON (portable, human-readable). Also a flat CSV of the
    key scalar columns for quick loading into pandas/plotting."""
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = tag or f"{FAMILY}_g{G}"
    base = f"gspf_data_{tag}_{stamp}"

    # 1. pickle: exact, everything
    # with open(base + ".pkl", "wb") as f:
    #     pickle.dump({"config": {
    #         "FAMILY": FAMILY, "SIZES": SIZES, "PROBS": PROBS,
    #         "N_TRIALS": N_TRIALS, "G": G, "MAX_TIME": MAX_TIME,
    #         "MINIMISE_SUPPORT": MINIMISE_SUPPORT, "SEED": SEED,
    #         "TARGET_QUBIT": TARGET_QUBIT,
    #     }, "records": records}, f)

    # # 2. JSON: drop the nested heur_phases dict's non-JSON bits are fine (all
    # #    ints/floats already); everything here is JSON-serialisable.
    # with open(base + ".json", "w") as f:
    #     json.dump({"config": {
    #         "FAMILY": FAMILY, "SIZES": [list(s) for s in SIZES],
    #         "PROBS": PROBS, "N_TRIALS": N_TRIALS, "G": G,
    #         "MAX_TIME": MAX_TIME, "MINIMISE_SUPPORT": MINIMISE_SUPPORT,
    #         "SEED": SEED, "TARGET_QUBIT": TARGET_QUBIT,
    #     }, "records": records}, f, indent=1)

    # 3. CSV: flat scalar columns for quick analysis
    cols = ["width", "length", "num_qubits", "loss_prob", "num_lost", "g",
            "ilp_success", "ilp_optimal", "ilp_status", "ilp_time",
            "ilp_capped", "ilp_objective", "ilp_support",
            "heur_success", "heur_time", "heur_setup", "heur_search",
            "heur_support"]
    import csv
    with open(base + ".csv", "w", newline="") as f:
        wtr = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        wtr.writeheader()
        for r in records:
            wtr.writerow({c: r.get(c) for c in cols})

    print(f"\ndata saved: {base}.pkl / .json / .csv  ({len(records)} trials)")
    return base

def plot_vs_size(records):
    """ILP solve time vs heuristic setup (basis) floor vs heuristic total,
    along the qubit-count axis, at the loss probability nearest
    PROB_FOR_SIZE_PLOT. Log y."""
    p = _nearest_prob(PROB_FOR_SIZE_PLOT)
    recs = [r for r in records if r["loss_prob"] == p]
    ns = [build_or_lookup_n(records, wl) for wl in SIZES]

    both = COMPARE_ON_BOTH_SUCCEEDED
    ilp_m, ilp_s   = _series_over(recs, SIZES, "size_key", "ilp_time",
                                  "ilp_success", both=both)
    setup_m, setup_s = _series_over(recs, SIZES, "size_key", "heur_setup",
                                    "heur_success", both=both)
    tot_m, tot_s   = _series_over(recs, SIZES, "size_key", "heur_time",
                                  "heur_success", both=both)

    filt = "both-ok instances" if both else "each own successes"
    print(f"\n===== time vs size at p={p:.2f} "
          f"(ILP solve vs heuristic setup floor; {filt}) =====")
    print(f"{'size':>6} {'n':>4} | {'ILP(s)':>10} {'heurSetup(s)':>13} "
          f"{'heurTot(s)':>11}")
    for i, (w, ell) in enumerate(SIZES):
        def f(x):
            return "   nan" if np.isnan(x) else f"{x:.4f}"
        print(f"{w}x{ell:<3} {ns[i]:>4.0f} | {f(ilp_m[i]):>10} "
              f"{f(setup_m[i]):>13} {f(tot_m[i]):>11}")
    print("=" * 56 + "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ns, ilp_m, yerr=ilp_s, marker="o", capsize=4,
                color="#4c72b0", label="g-SPF ILP (solve)")
    ax.errorbar(ns, setup_m, yerr=setup_s, marker="^", capsize=4,
                color="#55a868", label="heuristic setup floor (basis)")
    ax.errorbar(ns, tot_m, yerr=tot_s, marker="s", capsize=4,
                color="#dd8452", label="heuristic total")
    if LOG_Y:
        ax.set_yscale("log")
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel("average successful run time (s)")
    ax.set_title(f"{FAMILY}  |  p_loss={p:.2f}  |  "
                 f"{N_TRIALS} trials/cell  |  g={G}  |  {filt}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG_SIZE, dpi=150)
    print(f"figure written to {OUT_PNG_SIZE}")
    plt.show()


def plot_vs_size_all_p(records):
    """One time-vs-n panel per loss probability p, arranged in a grid. Each
    panel plots ILP solve time, heuristic setup floor, and heuristic total
    against qubit count on a shared log y-axis, so you can see how the
    n-scaling changes across the whole loss-probability sweep in one figure.
    Averaging follows COMPARE_ON_BOTH_SUCCEEDED."""
    both = COMPARE_ON_BOTH_SUCCEEDED
    filt = "both-ok instances" if both else "each own successes"
    ns = [build_or_lookup_n(records, wl) for wl in SIZES]

    nP = len(PROBS)
    ncols = min(5, nP)
    nrows = (nP + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, squeeze=False, sharex=True,
                             sharey=True, figsize=(3.4 * ncols, 3.0 * nrows))

    for idx, p in enumerate(PROBS):
        ax = axes[idx // ncols][idx % ncols]
        recs = [r for r in records if r["loss_prob"] == p]
        ilp_m, ilp_s = _series_over(recs, SIZES, "size_key", "ilp_time",
                                    "ilp_success", both=both)
        setup_m, setup_s = _series_over(recs, SIZES, "size_key", "heur_setup",
                                        "heur_success", both=both)
        tot_m, tot_s = _series_over(recs, SIZES, "size_key", "heur_time",
                                    "heur_success", both=both)

        ax.errorbar(ns, ilp_m, yerr=ilp_s, marker="o", capsize=3, ms=4,
                    color="#4c72b0", label="ILP (solve)")
        ax.errorbar(ns, setup_m, yerr=setup_s, marker="^", capsize=3, ms=4,
                    color="#55a868", label="heur setup")
        ax.errorbar(ns, tot_m, yerr=tot_s, marker="s", capsize=3, ms=4,
                    color="#dd8452", label="heur total")
        if LOG_Y:
            ax.set_yscale("log")
        ax.set_title(f"p = {p:.2f}", fontsize=9)
        ax.grid(True, which="both", alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    # hide any unused panels
    for idx in range(nP, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.supxlabel("number of qubits n")
    fig.supylabel("average successful run time (s)")
    fig.suptitle(f"{FAMILY}  |  {N_TRIALS} trials/cell  |  g={G}  |  {filt}")
    fig.tight_layout()
    fig.savefig(OUT_PNG_SIZE_ALLP, dpi=150)
    print(f"figure written to {OUT_PNG_SIZE_ALLP}")
    plt.show()


def plot_vs_prob(records):
    """ILP solve time vs heuristic setup vs heuristic total, along the loss
    probability axis, at a fixed size (SIZE_FOR_PROB_PLOT or the largest).
    Log y -- this is where raising p pushes the ILP into its hard regime."""
    size = SIZE_FOR_PROB_PLOT or SIZES[-1]
    recs = [r for r in records if r["size_key"] == size]
    n = recs[0]["num_qubits"] if recs else float("nan")

    both = COMPARE_ON_BOTH_SUCCEEDED
    ilp_m, ilp_s   = _series_over(recs, PROBS, "loss_prob", "ilp_time",
                                  "ilp_success", both=both)
    setup_m, setup_s = _series_over(recs, PROBS, "loss_prob", "heur_setup",
                                    "heur_success", both=both)
    tot_m, tot_s   = _series_over(recs, PROBS, "loss_prob", "heur_time",
                                  "heur_success", both=both)

    filt = "both-ok instances" if both else "each own successes"
    print(f"\n===== time vs loss prob at size {size[0]}x{size[1]} (n={n:.0f}) "
          f"({filt}) =====")
    print(f"{'p':>6} | {'ILP(s)':>10} {'heurSetup(s)':>13} {'heurTot(s)':>11} "
          f"| {'ILP ok':>7}")
    for i, p in enumerate(PROBS):
        cell = [r for r in recs if r["loss_prob"] == p]
        n_ilp = sum(r["ilp_success"] for r in cell)

        def f(x):
            return "   nan" if np.isnan(x) else f"{x:.4f}"
        print(f"{p:6.2f} | {f(ilp_m[i]):>10} {f(setup_m[i]):>13} "
              f"{f(tot_m[i]):>11} | {n_ilp:>3}/{len(cell)}")
    print("=" * 56 + "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(PROBS, ilp_m, yerr=ilp_s, marker="o", capsize=4,
                color="#4c72b0", label="g-SPF ILP (solve)")
    ax.errorbar(PROBS, setup_m, yerr=setup_s, marker="^", capsize=4,
                color="#55a868", label="heuristic setup floor (basis)")
    ax.errorbar(PROBS, tot_m, yerr=tot_s, marker="s", capsize=4,
                color="#dd8452", label="heuristic total")
    if LOG_Y:
        ax.set_yscale("log")
    ax.set_xlabel("loss probability p")
    ax.set_ylabel("average successful run time (s)")
    ax.set_title(f"{FAMILY} {size[0]}x{size[1]} (n={n:.0f})  |  "
                 f"{N_TRIALS} trials/cell  |  g={G}  |  {filt}")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG_PROB, dpi=150)
    print(f"figure written to {OUT_PNG_PROB}")
    plt.show()


def plot_success_rate(records):
    """Fraction of trials where the ILP vs the heuristic returned a solution,
    along the qubit-count axis, at the loss probability nearest
    PROB_FOR_SIZE_PLOT. Both are g-aware (success requires the pair's
    anticommutation number <= g), so the remaining gap is the heuristic's miss
    rate: where a <= g solution exists, the ILP finds it by optimising while
    the heuristic can miss it. ILP success >= heuristic success."""
    p = _nearest_prob(PROB_FOR_SIZE_PLOT)
    recs = [r for r in records if r["loss_prob"] == p]
    ns, ilp_rate, heur_rate, both_rate = [], [], [], []
    for wl in SIZES:
        cell = [r for r in recs if r["size_key"] == wl]
        ns.append(build_or_lookup_n(records, wl))
        m = len(cell) if cell else 1
        ilp_rate.append(sum(r["ilp_success"] for r in cell) / m)
        heur_rate.append(sum(r["heur_success"] for r in cell) / m)
        both_rate.append(sum(r["ilp_success"] and r["heur_success"]
                             for r in cell) / m)

    print(f"\n===== success rate vs size at p={p:.2f}  (g={G}) =====")
    print(f"{'size':>6} {'n':>4} | {'ILP ok':>8} {'heur ok':>8} {'both':>8}")
    for i, (w, ell) in enumerate(SIZES):
        print(f"{w}x{ell:<3} {ns[i]:>4.0f} | {ilp_rate[i]:>8.2f} "
              f"{heur_rate[i]:>8.2f} {both_rate[i]:>8.2f}")
    print("=" * 46 + "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(ns, ilp_rate, marker="o", color="#4c72b0", label="ILP success")
    ax.plot(ns, heur_rate, marker="s", color="#dd8452",
            label="heuristic success")
    ax.plot(ns, both_rate, marker="^", color="#55a868", linestyle="--",
            label="both success")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel("fraction of trials with a solution")
    ax.set_title(f"{FAMILY}  |  p_loss={p:.2f}  |  "
                 f"{N_TRIALS} trials/cell  |  g={G}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG_SUCC, dpi=150)
    print(f"figure written to {OUT_PNG_SUCC}")
    plt.show()


def plot_support_vs_size(records):
    """Average joint (X,Z) support of the returned logical vs qubit count, at
    the loss probability nearest PROB_FOR_SIZE_PLOT. Heuristic support vs ILP
    support. NOTE: where the ILP is time-capped (ilp_capped), its support is the
    best found within MAX_TIME, NOT the proven minimum -- so this compares
    heuristic support against 'ILP-best-within-cap', not against the optimum."""
    p = _nearest_prob(PROB_FOR_SIZE_PLOT)
    recs = [r for r in records if r["loss_prob"] == p]
    ns = [build_or_lookup_n(records, wl) for wl in SIZES]
    both = COMPARE_ON_BOTH_SUCCEEDED

    def _supp_series(key):
        means, stds = [], []
        for wl in SIZES:
            vals = []
            for r in recs:
                if r["size_key"] != wl:
                    continue
                if both and not (r["ilp_success"] and r["heur_success"]):
                    continue
                v = r.get(key)
                if v is not None:
                    vals.append(v)
            m, s = _mean_std(vals)
            means.append(m); stds.append(s)
        return np.array(means), np.array(stds)

    ilp_m, ilp_s   = _supp_series("ilp_support")
    heur_m, heur_s = _supp_series("heur_support")

    # fraction of the ILP points at each size that were time-capped (so you can
    # see where 'ILP support' stops being the true optimum)
    cap_frac = []
    for wl in SIZES:
        cell = [r for r in recs if r["size_key"] == wl
                and (not both or (r["ilp_success"] and r["heur_success"]))]
        capped = sum(bool(r.get("ilp_capped")) for r in cell if r["ilp_success"])
        nok = sum(r["ilp_success"] for r in cell)
        cap_frac.append(capped / nok if nok else float("nan"))

    filt = "both-ok instances" if both else "each own successes"
    print(f"\n===== joint (X,Z) support vs size at p={p:.2f} ({filt}) =====")
    print(f"{'size':>6} {'n':>4} | {'heurSupp':>9} {'ilpSupp':>8} "
          f"{'ratio h/i':>9} {'ILPcap%':>8}")
    for i, (w, ell) in enumerate(SIZES):
        def f(x): return "  nan" if np.isnan(x) else f"{x:.2f}"
        ratio = (heur_m[i] / ilp_m[i]) if (ilp_m[i] and not np.isnan(ilp_m[i])
                                           and ilp_m[i] > 0) else float("nan")
        print(f"{w}x{ell:<3} {ns[i]:>4.0f} | {f(heur_m[i]):>9} {f(ilp_m[i]):>8} "
              f"{f(ratio):>9} {cap_frac[i]*100:>7.0f}%")
    print("=" * 52 + "\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(ns, ilp_m, yerr=ilp_s, marker="o", capsize=4,
                color="#4c72b0", label="ILP support (best within cap)")
    ax.errorbar(ns, heur_m, yerr=heur_s, marker="s", capsize=4,
                color="#dd8452", label="heuristic support")
    ax.set_xlabel("number of qubits n")
    ax.set_ylabel("average joint (X,Z) support size")
    ax.set_title(f"{FAMILY}  |  p_loss={p:.2f}  |  {N_TRIALS} trials/cell  |  "
                 f"g={G}  |  {filt}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"gspf_support_vs_size_{FAMILY}_{G}.png", dpi=150)
    print(f"figure written to gspf_support_vs_size_{FAMILY}_{G}.png")
    plt.show()


def summarise_heuristic_phases(records):
    """Per size: heuristic setup vs coset-search wall time (mean per trial),
    unaccounted 'other' time, and decoder failure counts (None returns -- a
    failed validity check, not max_iter exhaustion)."""
    print("\n============== heuristic phase breakdown (mean per trial) "
          "==============")
    print(f"{'size':>6} {'n':>4} | {'setup(s)':>9} {'search(s)':>10} "
          f"{'other(s)':>9} | {'bp1 calls':>9} {'None: bp1':>10} {'bp2':>4} "
          f"{'anti':>5}")
    for (w, ell) in SIZES:
        s = [r for r in records if r["width"] == w and r["length"] == ell]
        if not s:
            continue
        ntr = len(s)
        agg = {}
        for r in s:
            for label, v in r["heur_phases"].items():
                a = agg.setdefault(label, {"time": 0.0, "calls": 0, "none": 0})
                a["time"]  += v["time"]
                a["calls"] += v["calls"]
                a["none"]  += v["none"]

        def _t(label):
            return agg.get(label, {}).get("time", 0.0)

        def _c(label):
            return agg.get(label, {}).get("calls", 0)

        def _n(label):
            return agg.get(label, {}).get("none", 0)

        setup_t  = sum(_t(l) for l in _SETUP_LABELS)
        search_t = sum(_t(l) for l in _SEARCH_LABELS)
        total_t  = sum(r["heur_time"] for r in s)
        other_t  = total_t - setup_t - search_t
        n = s[0]["num_qubits"]

        print(f"{w}x{ell:<3} {n:>4} | "
              f"{setup_t / ntr:9.4f} {search_t / ntr:10.4f} "
              f"{other_t / ntr:9.4f} | "
              f"{_c('bp_first') / ntr:9.2f} "
              f"{_n('bp_first'):>10} {_n('bp_second'):>4} {_n('anti_second'):>5}")
    print("bp1 calls > 1/trial means the anti-commuting retry loop is firing "
          "(only with a target set); None = decoder validity-check failures, "
          "not max_iter. With TARGET_QUBIT=None each bp call is a single "
          "BP+OSD solve.")
    print("=================================================================="
          "==========\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="graph-code g-SPF ILP vs heuristic timing sweep")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--single-thread", dest="single_thread",
                       action="store_true",
                       help="pin CP-SAT to one worker (fair timing)")
    group.add_argument("--no-single-thread", dest="single_thread",
                       action="store_false",
                       help="let CP-SAT use multiple workers")
    parser.set_defaults(single_thread=FORCE_SINGLE_THREAD_ILP)
    args = parser.parse_args()

    # CLI value overrides the FORCE_SINGLE_THREAD_ILP constant for this run
    set_single_thread_ilp(args.single_thread)
    print(f"single-thread ILP: {args.single_thread}")

    recs = run_benchmark()

    save_all_data(recs)
    plot_vs_size(recs)

    plot_vs_size_all_p(recs)
    plot_support_vs_size(recs)
    plot_vs_prob(recs)
    plot_success_rate(recs)
    summarise_heuristic_phases(recs)