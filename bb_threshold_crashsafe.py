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
    lost_masks = rng.random(size=(num_shots, num_qubits)) < p
    if exclude is not None:
        lost_masks[:, exclude] = False
    return lost_masks


# ---------------------------------------------------------------------------
# crash-safe CSV writer: temp file then atomic rename.
# ---------------------------------------------------------------------------
def _atomic_write_csv(out_dict, output_path):
    lengths = {k: len(v) for k, v in out_dict.items()}
    if len(set(lengths.values())) > 1:
        raise RuntimeError(f"ragged columns, refusing to write: {lengths}")
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(out_dict.keys())
        writer.writerows(zip(*out_dict.values()))
    tmp.replace(output_path)


def _atomic_write_rows(rows, columns, path):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Two separate caches of found logical pairs, each keyed on (BB-type, support)
# so a given support is stored once per cache:
#   * heuristic cache  -- pairs found by the heuristic
#   * exact cache      -- pairs found by g-SPF or ILP; carries a
#                         'guaranteed minimum' marker that is True only when the
#                         solver returned status OPTIMAL. If the ILP capped out
#                         (returned FEASIBLE but not proven optimal), the marker
#                         is False. If a later exact solve proves the same
#                         support optimal, the existing entry is upgraded.
# ---------------------------------------------------------------------------
HEUR_CACHE_COLUMNS = ["BB-type", "support", "support size", "found by",
                      "X logical", "Z logical"]
EXACT_CACHE_COLUMNS = ["BB-type", "support", "support size", "found by",
                       "status", "guaranteed minimum", "X logical", "Z logical"]

MAX_CACHE = 500          # max distinct pairs per cache; None = unbounded
CACHE_EVICT = "last"     # "best" | "fifo" | "last" | "drop"


def _evict_index(store, policy):
    if not store["rows"]:
        return None
    if policy == "fifo":
        return 0
    if policy == "last":
        return len(store["rows"]) - 1
    return None


def _reindex_exact_keys(store):
    store["keys"] = {(r["BB-type"], frozenset(r["support"])): i
                     for i, r in enumerate(store["rows"])}


def cache_add_heuristic(store, x, z, target_qubit, lm_tuple,
                        max_cache=MAX_CACHE, policy=CACHE_EVICT):
    if x is None or z is None:
        return
    supp = sc.pair_support(x, z, target_qubit)
    if supp is None:
        return
    key = (lm_tuple, supp)
    if key in store["keys"]:
        return
    new_size = len(supp)
    if max_cache is not None and len(store["rows"]) >= max_cache:
        if policy == "drop":
            return
        if policy == "best":
            worst_idx = max(range(len(store["rows"])),
                            key=lambda i: store["rows"][i]["support size"])
            if store["rows"][worst_idx]["support size"] <= new_size:
                return
            evict = worst_idx
        else:
            evict = _evict_index(store, policy)
        ev_key = (store["rows"][evict]["BB-type"],
                  frozenset(store["rows"][evict]["support"]))
        store["keys"].discard(ev_key)
        store["rows"].pop(evict)
    store["keys"].add(key)
    store["rows"].append({
        "BB-type": lm_tuple,
        "support": sorted(int(q) for q in supp),
        "support size": new_size,
        "found by": "Heuristic",
        "X logical": ta.tableau2paulistring(x),
        "Z logical": ta.tableau2paulistring(z),
    })


def cache_add_exact(store, x, z, target_qubit, lm_tuple, source, status_name,
                    max_cache=MAX_CACHE, policy=CACHE_EVICT):
    if x is None or z is None:
        return
    supp = sc.pair_support(x, z, target_qubit)
    if supp is None:
        return
    key = (lm_tuple, supp)
    guaranteed = (status_name == "OPTIMAL")
    if key in store["keys"]:
        idx = store["keys"][key]
        old = store["rows"][idx]
        if guaranteed and not old["guaranteed minimum"]:
            old["guaranteed minimum"] = True
            old["status"] = status_name
            old["found by"] = source
        return
    new_size = len(supp)
    if max_cache is not None and len(store["rows"]) >= max_cache:
        if policy == "drop":
            return
        if policy == "best":
            def badness(r):
                return (0 if r["guaranteed minimum"] else 1, r["support size"])
            worst_idx = max(range(len(store["rows"])),
                            key=lambda i: badness(store["rows"][i]))
            new_badness = (1 if not guaranteed else 0, new_size)
            if badness(store["rows"][worst_idx]) <= new_badness:
                return
            evict = worst_idx
        else:
            evict = _evict_index(store, policy)
        store["rows"].pop(evict)
        _reindex_exact_keys(store)
    store["keys"][key] = len(store["rows"])
    store["rows"].append({
        "BB-type": lm_tuple,
        "support": sorted(int(q) for q in supp),
        "support size": new_size,
        "found by": source,
        "status": status_name,
        "guaranteed minimum": guaranteed,
        "X logical": ta.tableau2paulistring(x),
        "Z logical": ta.tableau2paulistring(z),
    })
def _blank_row(columns):
    return {c: None for c in columns}


def _append_row(out_dict, row):
    for k, v in row.items():
        out_dict[k].append(v)


def _safe(fn, default=None):
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
    "g-SPF X logical", "g-SPF Z logical", "g-SPF status", "early flag",
    "early runtime", "g-SPF g",
    "ILP success", "ILP verify", "ILP runtime", "ILP support size",
    "ILP X logical", "ILP Z logical", "ILP status", "ILP g",
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
lost_prob = np.linspace(0, 0.2, 5)
ms = True
target_qubit = None
MAX_CACHE = 500     # max distinct pairs kept per cache; None = unbounded
# ---------------------------------------------------------------------------- #

out_dict = defaultdict(list)
output_path     = data_directory / f"{time_str}_{code}_thres_min_{ms}_{gg}.csv"
heur_cache_path = data_directory / f"{time_str}_{code}_cache_heuristic_{gg}.csv"
exact_cache_path= data_directory / f"{time_str}_{code}_cache_exact_{gg}.csv"

heur_cache  = {"keys": set(),  "rows": []}
exact_cache = {"keys": {},     "rows": []}

for lm_tuple in bb_tuples[:1]:
    seen = set()
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
                cache_add_heuristic(heur_cache, res_heu["x"], res_heu["z"], target_qubit, lm_tuple)
            else:
                row["Heuristic success"] = False

            # --- g-SPF (ILP with loss handling) ---
            def _gspf():
                return generalised_spf_logical(T, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
            t2 = time.time()
            out, err = _safe(_gspf, default=({"success": False, "x": None, "z": None, "status_name": None}, None, None))
            row["g-SPF runtime"] = time.time() - t2
            if err:
                errors.append("gspf:" + err)
            res_gspf, ea_flag, rt_early = out
            row["early flag"] = ea_flag
            row["early runtime"] = rt_early
            row["g-SPF status"] = res_gspf.get("status_name")
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
                cache_add_exact(exact_cache, res_gspf["x"], res_gspf["z"], target_qubit,
                                lm_tuple, "g-SPF", res_gspf.get("status_name"))
            else:
                row["g-SPF success"] = False

            # --- raw ILP ---
            def _ilp():
                Tg = ta.to_gf2_tableau(T)
                _, xlogi, zlogi = sc.initialise_logical_basis(Tg)
                return gspf_ilp(Tg, xlogi, zlogi, [], lq, gg, target_qubit=target_qubit, minimise_support=ms)
            t3 = time.time()
            res_ilp, err = _safe(_ilp, default={"success": False, "x": None, "z": None, "status_name": None})
            row["ILP runtime"] = time.time() - t3
            if err:
                errors.append("ilp:" + err)
            row["ILP status"] = res_ilp.get("status_name")
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
                cache_add_exact(exact_cache, res_ilp["x"], res_ilp["z"], target_qubit,
                                lm_tuple, "ILP", res_ilp.get("status_name"))
            else:
                row["ILP success"] = False

            row["error"] = "; ".join(errors) if errors else None

            # atomic append + checkpoint all three files after EVERY shot
            _append_row(out_dict, row)
            _atomic_write_csv(out_dict, output_path)
            _atomic_write_rows(heur_cache["rows"], HEUR_CACHE_COLUMNS, heur_cache_path)
            _atomic_write_rows(exact_cache["rows"], EXACT_CACHE_COLUMNS, exact_cache_path)

n_exact_capped = sum(1 for r in exact_cache["rows"] if not r["guaranteed minimum"])
print("done. rows:", len(out_dict["BB-type"]))
print("heuristic cache pairs:", len(heur_cache["rows"]))
print("exact cache pairs:", len(exact_cache["rows"]),
      f"({n_exact_capped} not guaranteed minimum / capped)")
print("results ->", output_path)
print("heur cache ->", heur_cache_path)
print("exact cache ->", exact_cache_path)
