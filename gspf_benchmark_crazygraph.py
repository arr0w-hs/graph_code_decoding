"""
Benchmark g-SPF (ILP) against the g-SPF heuristic on *crazy-graph* codes,
swept over code size.

crazy_graph is the densest of the supplied graph builders: every node in a
layer connects to every node in the next layer. That produces high-weight
stabilisers and large-support logicals, which is the regime where the ILP's
minimum-support optimality proof is most expensive, while the heuristic's
find-a-short-logical cost stays polynomial. So it is the family chosen to
expose the runtime gap without scaling the qubit count very far.

For each size (a crazy_graph width x length) this builds the code via
create_graph_code, draws N_TRIALS random loss patterns at a fixed loss
probability, runs both solvers on each pattern, and times them. It keeps the
runs that reported success, averages run times per method per size, and plots
average successful runtime against the qubit count on a log y-axis so the
scaling divergence is visible.

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
FAMILY = "surface_code"
# Each SIZES entry is the pair of size parameters for the chosen family:
# (width, length) for the lattice families, (branches, depth) for
# tree_to_tree_graph, (num_nodes, edge_prob) for random_graph, (L, _) for
# surface_code (second element ignored). Qubit count grows along this list.
#   e.g. for random_graph, sweep size at fixed density:
#   SIZES = [(8, 0.5), (11, 0.5), (14, 0.5), (18, 0.5), (22, 0.5), (26, 0.5)]
#   e.g. for surface_code:  SIZES = [(3, 0), (5, 0), (7, 0), (9, 0)]
#SIZES = [(2, 2), (2, 3), (3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]
SIZES=SIZES = [(3, 0), (5, 0), (7, 0), (9, 0)]
N_TRIALS       = 15      # random loss patterns per (size, p) cell
# Loss probability grid. The sweep runs every size at every p, so cost scales as
# len(SIZES) * len(PROBS) * N_TRIALS. Trim any of the three to shorten it.
PROBS          = [round(x, 3) for x in np.linspace(0.05, 0.5, 10)]
# Which slice each plot takes from the 2-D (size, p) grid:
PROB_FOR_SIZE_PLOT = 0.30    # size-axis plot uses the p in PROBS nearest this
SIZE_FOR_PROB_PLOT = None    # prob-axis plot uses this (w, ell); None -> largest
# TARGET_QUBIT: None -> no target, endpoints eligible for loss (only the input
#   node is structurally excluded, see below). "output" -> target the output
#   (readout) qubit of each size; that qubit's node is protected from loss.
#   int -> a fixed target qubit index; that qubit's node is protected.
TARGET_QUBIT   = None
G              = 3     # the g in g-SPF
MINIMISE_SUPPORT = True  # keep True: this is where the ILP pays its cost
CACHE_LOGICAL_BASIS = True  # compute the full-T logical basis once per size
                            # and reuse it across trials (both solvers)
MAX_TIME       = 60      # ILP solver time cap in seconds (censors long solves)
SEED           = 0       # RNG seed for reproducible loss patterns
QUIET          = True  # suppress the solvers' own print output
FORCE_SINGLE_THREAD_ILP = True   # pin CP-SAT to 1 worker for fair timing
LOG_Y          = True    # log-scale the runtime axis to show scaling
# Compare ILP vs heuristic only on trials where BOTH returned a solution, so the
# timing curves are averaged over identical instances (recommended when g>1,
# where the ILP and the g=1-style heuristic have different feasible sets).
COMPARE_ON_BOTH_SUCCEEDED = True
OUT_PNG_SIZE   = f"gspf_timing_vs_size_{FAMILY}.png"
OUT_PNG_PROB   = f"gspf_timing_vs_prob_{FAMILY}.png"
OUT_PNG_SUCC   = f"gspf_success_rate_{FAMILY}.png"
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
        # not a graph code: surface_code(L) returns the tableau H directly.
        # SIZES entries are (L, _) -- the second element is ignored.
        from code_importer import surface_code
        _, _, H, _, _ = surface_code(int(a))
        T = ta.to_gf2_tableau(H)
        return T, T.shape[1] // 2
    G = GRAPH_FAMILIES[FAMILY](a, b)
    # sort node labels so tableau qubit i corresponds to graph node i
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

                # --- g-SPF heuristic ---
                def _heur():
                    return generalised_spf_logical_heuristic(
                        T, lost,G, target_qubit=target,
                    )

                with _maybe_silence(QUIET):
                    heur_res, heur_time, heur_crashed, heur_phases = \
                        _run_heuristic_probed(_heur)

                setup_t, search_t = _phase_split(heur_phases)

                records.append({
                    "width": w, "length": ell,
                    "size_key": (w, ell),
                    "num_qubits": num_qubits,
                    "loss_prob": float(p),
                    "num_lost": len(lost),
                    "ilp_success":  bool(ilp_res.get("success", False)),
                    "ilp_optimal":  ilp_res.get("status_name") == "OPTIMAL",
                    "ilp_time":     ilp_time,
                    "ilp_crashed":  ilp_crashed,
                    "heur_success": bool(heur_res.get("success", False)),
                    "heur_time":    heur_time,
                    "heur_setup":   setup_t,
                    "heur_search":  search_t,
                    "heur_crashed": heur_crashed,
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


def build_or_lookup_n(records, size_key):
    for r in records:
        if r["size_key"] == size_key:
            return r["num_qubits"]
    return float("nan")


def plot_success_rate(records):
    """Fraction of trials where the ILP vs the heuristic returned a solution,
    along the qubit-count axis, at the loss probability nearest
    PROB_FOR_SIZE_PLOT. The gap is the feasibility regime: at g>1 the ILP can
    use anti_sum in {1,3,...,g} while the heuristic only finds a g=1-style
    single-anticommutation logical, so instances feasible only at higher
    anti_sum show up as ILP success above heuristic success."""
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
        description="crazy_graph g-SPF ILP vs heuristic timing sweep")
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
    plot_vs_size(recs)
    plot_vs_prob(recs)
    plot_success_rate(recs)
    summarise_heuristic_phases(recs)
