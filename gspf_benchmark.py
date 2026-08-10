"""
Benchmark g-SPF (ILP) against the g-SPF heuristic on a surface code, swept
over loss probability.

For a surface code of size L, this sweeps the per-qubit loss probability p
across a grid from 0 to 1. At each p it generates N_TRIALS random loss
patterns, runs both solvers on each pattern, and records how long each takes.
For every p it keeps only the runs that reported success (result["success"]
is True), averages the run times per method, and plots average successful
runtime against p (one line per method).

The two solver functions are imported below from `gspf`. If the file that
contains `generalised_spf_logical` / `generalised_spf_logical_heuristic` is
saved under a different name, change that import line only.
"""

import os
import time
import contextlib

import numpy as np
import matplotlib.pyplot as plt

import tableau as ta
from code_importer import surface_code

# ---- edit this line if the solver file is not named gspf.py ----------------
from generalised_spf import generalised_spf_logical, generalised_spf_logical_heuristic
# ----------------------------------------------------------------------------


# ============================ configuration =================================
L             = 6                      # surface code linear size
N_TRIALS      = 30                      # random loss patterns per probability
PROBS         = np.linspace(0.0, 1.0, 11)  # loss probabilities to sweep (0..1)
G             = 1                       # the g in g-SPF
TARGET_QUBIT  = None                    # required output qubit index, or None
MAX_TIME      = 600                     # ILP solver time cap in seconds
SEED          = 0                       # RNG seed for reproducible patterns
QUIET         = True                    # suppress the solvers' own print output
OUT_PNG       = "gspf_timing_vs_p_L{}.png".format(L)
# Note: total ILP solves = N_TRIALS * len(PROBS). Raising either multiplies
# the wall-clock cost of the sweep.
# ============================================================================


@contextlib.contextmanager
def _maybe_silence(quiet):
    """Redirect stdout to /dev/null while a solver runs, if quiet is set."""
    if not quiet:
        yield
        return
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            yield


def random_loss_pattern(num_qubits, loss_prob, rng, exclude=None):
    """Return a list of lost qubit indices, each qubit lost with prob loss_prob.

    `exclude` (an int or None) is never placed in the loss set, so that a
    chosen target qubit is not accidentally lost.
    """
    excluded = set() if exclude is None else {exclude}
    return [q for q in range(num_qubits)
            if q not in excluded and rng.random() < loss_prob]


def _time_call(fn):
    """Run fn(), returning (result_dict, elapsed_seconds, crashed_flag).

    A crash is caught so a single bad trial does not abort the sweep;
    a crashed run is treated as not-successful.
    """
    t0 = time.perf_counter()
    try:
        res = fn()
        crashed = False
    except Exception as exc:            # noqa: BLE001 - report and continue
        res = {"success": False, "error": repr(exc)}
        crashed = True
    elapsed = time.perf_counter() - t0
    return res, elapsed, crashed


def run_benchmark():
    _, _, H, _, _ = surface_code(L)
    T = ta.to_gf2_tableau(H)
    num_qubits = T.shape[1] // 2

    if TARGET_QUBIT is not None:
        assert 0 <= TARGET_QUBIT < num_qubits, \
            f"TARGET_QUBIT must be in 0..{num_qubits - 1}"

    rng = np.random.default_rng(SEED)
    records = []

    for p in PROBS:
        for trial in range(N_TRIALS):
            lost = random_loss_pattern(num_qubits, p, rng, exclude=TARGET_QUBIT)

            # --- g-SPF (ILP) ---
            def _ilp():
                res, _, _ = generalised_spf_logical(
                    T, [], lost, G,
                    target_qubit=TARGET_QUBIT,
                    max_time=MAX_TIME,
                    minimise_support=True,
                )
                return res

            with _maybe_silence(QUIET):
                ilp_res, ilp_time, ilp_crashed = _time_call(_ilp)

            # --- g-SPF heuristic ---
            def _heur():
                return generalised_spf_logical_heuristic(
                    T, lost, target_qubit=TARGET_QUBIT,
                )

            with _maybe_silence(QUIET):
                heur_res, heur_time, heur_crashed = _time_call(_heur)

            records.append({
                "loss_prob":    float(p),
                "trial":        trial,
                "num_lost":     len(lost),
                "ilp_success":  bool(ilp_res.get("success", False)),
                "ilp_time":     ilp_time,
                "ilp_crashed":  ilp_crashed,
                "heur_success": bool(heur_res.get("success", False)),
                "heur_time":    heur_time,
                "heur_crashed": heur_crashed,
            })

        # per-probability progress line
        p_recs = [r for r in records if r["loss_prob"] == float(p)]
        n_ilp  = sum(r["ilp_success"] for r in p_recs)
        n_heur = sum(r["heur_success"] for r in p_recs)
        print(f"p={p:4.2f} | {N_TRIALS} trials | "
              f"ILP ok {n_ilp:>3}/{N_TRIALS} | heur ok {n_heur:>3}/{N_TRIALS}")

    return records


def _mean_std_by_p(records, probs, time_key, success_key):
    """Return (means, stds) arrays over probs of successful-run times.

    A probability with no successful runs yields NaN so it leaves a gap.
    """
    means, stds = [], []
    for p in probs:
        times = [r[time_key] for r in records
                 if r["loss_prob"] == float(p) and r[success_key]]
        if times:
            means.append(float(np.mean(times)))
            stds.append(float(np.std(times)))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


def summarise_and_plot(records):
    probs = list(PROBS)

    ilp_mean,  ilp_std  = _mean_std_by_p(records, probs, "ilp_time",  "ilp_success")
    heur_mean, heur_std = _mean_std_by_p(records, probs, "heur_time", "heur_success")

    print("\n=================== summary ===================")
    print(f"{'p':>6} | {'ILP ok':>7} {'ILP mean(s)':>12} | "
          f"{'heur ok':>8} {'heur mean(s)':>13}")
    for i, p in enumerate(probs):
        p_recs = [r for r in records if r["loss_prob"] == float(p)]
        n_ilp  = sum(r["ilp_success"] for r in p_recs)
        n_heur = sum(r["heur_success"] for r in p_recs)
        im = "   nan" if np.isnan(ilp_mean[i])  else f"{ilp_mean[i]:.4f}"
        hm = "   nan" if np.isnan(heur_mean[i]) else f"{heur_mean[i]:.4f}"
        print(f"{p:6.2f} | {n_ilp:>7} {im:>12} | {n_heur:>8} {hm:>13}")
    print("===============================================\n")

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.errorbar(probs, ilp_mean, yerr=ilp_std, marker="o", capsize=4,
                color="#4c72b0", label="g-SPF (ILP)")
    ax.errorbar(probs, heur_mean, yerr=heur_std, marker="s", capsize=4,
                color="#dd8452", label="g-SPF heuristic")
    ax.set_xlabel("loss probability p")
    ax.set_ylabel("average successful run time (s)")
    ax.set_title(f"Surface code L={L}  |  {N_TRIALS} trials/point  |  g={G}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"figure written to {OUT_PNG}")
    plt.show()

    return {"probs": probs, "ilp_mean": ilp_mean, "heur_mean": heur_mean}


if __name__ == "__main__":
    recs = run_benchmark()
    summarise_and_plot(recs)
