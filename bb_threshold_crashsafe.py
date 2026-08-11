import time
from pathlib import Path
import numpy as np
import csv
import matplotlib.pyplot as plt
from galois import GF2
import pandas as pd
from generalised_spf import generalised_spf_logical, gspf_ilp, generalised_spf_logical_heuristic
import stabiliser_code as sc
import tableau as ta
from code_importer import rotated_surface_code, surface_code, bb_tableau
from collections import defaultdict
from gspf_tests import test_gspf


def sample_lost_masks(num_qubits, p, num_shots=1, exclude=None, rng=None):
    rng = np.random.default_rng(rng)
    # independent Bernoulli(p) loss per qubit
    lost_masks = rng.random(size=(num_shots, num_qubits)) < p
    if exclude is not None:
        lost_masks[:, exclude] = False
    return lost_masks


# ---------------------------------------------------------------------------
# crash-safe CSV writer: write to a temp file then atomically rename over the
# real file. A kill mid-write can never leave a half-written CSV -- you get
# either the previous complete file or the new complete file. Called after
# every shot, so at most one shot's work is ever at risk.
# ---------------------------------------------------------------------------
def _atomic_write_csv(out_dict, output_path):
    lengths = {k: len(v) for k, v in out_dict.items()}
    if len(set(lengths.values())) > 1:                       # ragged-column guard
        raise RuntimeError(f"ragged columns, refusing to write: {lengths}")
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_dict.keys())
        writer.writerows(zip(*out_dict.values()))
    tmp.replace(output_path)                                 # atomic rename


def _blank_row(columns):
    return {c: None for c in columns}


def _append_row(out_dict, row):
    # every row carries every column (None where unset) so columns never desync
    for k, v in row.items():
        out_dict[k].append(v)


def _safe(fn, default=None):
    """Run fn(); return (result, None) on success, (default, repr(exc)) on error."""
    try:
        return fn(), None
    except Exception as exc:   # noqa: BLE001
        return default, repr(exc)


COLUMNS = [
    "BB-type", "Loss probability", "Shot number", "lost_qubits",
    "Heuristic success", "Heuristic verify", "Heuristic runtime",
    "Heuristic support size", "Heuristic X logical", "Heuristic Z logical",
    "Heuristic g",
    "g-SPF success", "g-SPF verify", "g-SPF runtime", "g-SPF support size",
    "g-SPF X logical", "g-SPF Z logical", "early flag", "early runtime",
    "g-SPF g",
    "ILP success", "ILP verify", "ILP runtime", "ILP support size",
    "ILP X logical", "ILP Z logical", "ILP g",
    "error",
]


# ------------------------------- PARAMETERS --------------------------------- #
ts = pd.Timestamp.now(tz="Europe/Stockholm")
date_str = ts.strftime("%Y-%m-%d")
time_str = ts.strftime("%H%M%S")
base_dir = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
data_directory = base_dir / "threshold_folder" / f"{date_str}"
data_directory.mkdir(parents=True, exist_ok=True)

bb_tuples = [(3, 3), (3, 6), (6, 6), (9, 6), (6, 12), (12, 6), (12, 12)]
code = "bb"
num_shots = 10
gg = 3
lost_prob = np.linspace(0, 0.2, 5)   # loss grid (BB: keep low; NOT linspace(0,1,1))
ms = True
target_qubit = None
# ---------------------------------------------------------------------------- #

out_dict = defaultdict(list)
output_path = data_directory / f"{time_str}_{code}_thres_min_{ms}_{gg}.csv"

for lm_tuple in bb_tuples[:1]:
    seen = set()
    cache = []
    _, _, H, _, _ = bb_tableau(lm_tuple)
    T = ta.to_gf2_tableau(H)
    num_qubits = T.shape[1] // 2
    print('number of physical qubits:', num_qubits)

    for p in lost_prob:
        print('p:', p)
        lost_mask = sample_lost_masks(num_qubits, p, int(10 * num_shots), exclude=target_qubit)

        for shot in range(num_shots):
            print("shot:", shot)
            lq = np.flatnonzero(lost_mask[shot])
            lost_set = frozenset(int(c) for c in lq)
            if lost_set in seen:
                continue
            seen.add(lost_set)

            row = _blank_row(COLUMNS)
            row["BB-type"] = lm_tuple
            row["Loss probability"] = p
            row["Shot number"] = shot
            row["lost_qubits"] = lost_set
            errors = []

            # --- heuristic ---
            def _heu():
                return generalised_spf_logical_heuristic(T, lq, gg, target_qubit=target_qubit)
            t1 = time.time()
            res_heu, err = _safe(_heu, default={"success": False, "x": None, "z": None})
            row["Heuristic runtime"] = time.time() - t1
            if err:
                errors.append("heu:" + err)
            if res_heu.get("success"):
                row["Heuristic success"] = res_heu["success"]
                row["Heuristic support size"] = sc.pair_support(res_heu["x"], res_heu["z"], target_qubit, size=True)
                v, e = _safe(lambda: test_gspf(T, res_heu["x"], res_heu["z"], [], lq, g=gg), default=None)
                row["Heuristic verify"] = v
                if e:
                    errors.append("heu_verify:" + e)
                row["Heuristic X logical"] = ta.tableau2paulistring(res_heu["x"])
                row["Heuristic Z logical"] = ta.tableau2paulistring(res_heu["z"])
                a = ta.qubit_wise_commutation(res_heu["x"], res_heu["z"])
                row["Heuristic g"] = len(a) if a is not None else None
            else:
                row["Heuristic success"] = False

            # --- g-SPF (ILP with loss handling) ---
            def _gspf():
                return generalised_spf_logical(T, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
            t2 = time.time()
            out, err = _safe(_gspf, default=({"success": False, "x": None, "z": None}, None, None))
            row["g-SPF runtime"] = time.time() - t2
            if err:
                errors.append("gspf:" + err)
            res_gspf, ea_flag, rt_early = out
            row["early flag"] = ea_flag
            row["early runtime"] = rt_early
            if res_gspf.get("success"):
                row["g-SPF success"] = res_gspf["success"]
                row["g-SPF support size"] = sc.pair_support(res_gspf["x"], res_gspf["z"], target_qubit, size=True)
                v, e = _safe(lambda: test_gspf(T, res_gspf["x"], res_gspf["z"], [], lq, g=gg), default=None)
                row["g-SPF verify"] = v
                if e:
                    errors.append("gspf_verify:" + e)
                row["g-SPF X logical"] = ta.tableau2paulistring(res_gspf["x"])
                row["g-SPF Z logical"] = ta.tableau2paulistring(res_gspf["z"])
                a = ta.qubit_wise_commutation(res_gspf["x"], res_gspf["z"])
                row["g-SPF g"] = len(a) if a is not None else None
            else:
                row["g-SPF success"] = False

            # --- raw ILP ---
            def _ilp():
                Tg = ta.to_gf2_tableau(T)
                # needs initialise_logical_basis for k>1 so xlogi,zlogi anti-commute
                _, xlogi, zlogi = sc.initialise_logical_basis(Tg)
                return gspf_ilp(Tg, xlogi, zlogi, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
            t3 = time.time()
            res_ilp, err = _safe(_ilp, default={"success": False, "x": None, "z": None})
            row["ILP runtime"] = time.time() - t3
            if err:
                errors.append("ilp:" + err)
            if res_ilp.get("success"):
                row["ILP success"] = res_ilp["success"]
                row["ILP support size"] = sc.pair_support(res_ilp["x"], res_ilp["z"], target_qubit, size=True)
                v, e = _safe(lambda: test_gspf(T, res_ilp["x"], res_ilp["z"], [], lq, g=gg), default=None)
                row["ILP verify"] = v
                if e:
                    errors.append("ilp_verify:" + e)
                row["ILP X logical"] = ta.tableau2paulistring(res_ilp["x"])
                row["ILP Z logical"] = ta.tableau2paulistring(res_ilp["z"])
                a = ta.qubit_wise_commutation(res_ilp["x"], res_ilp["z"])
                row["ILP g"] = len(a) if a is not None else None
                supp = sc.pair_support(res_ilp["x"], res_ilp["z"], target_qubit)
                if supp is not None and supp not in cache:
                    cache.append(supp)
            else:
                row["ILP success"] = False

            row["error"] = "; ".join(errors) if errors else None

            # atomic append + checkpoint write after EVERY shot
            _append_row(out_dict, row)
            _atomic_write_csv(out_dict, output_path)

print("done. rows:", len(out_dict["BB-type"]), "-> file:", output_path)
