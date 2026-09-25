"""Messreihe Inventory Routing: gepaarte Regelvergleiche auf demselben Verbrauchsstrom (aus messreihe_inventory_routing/sweep.py, Logik
unverändert; die Instanzen und Regeln kommen jetzt aus irp_model / irp_policy).

Ausgabe (in den Ordner `tools/_out` oder den ersten Aufruf-Parameter): raw_sweep.npz (je Konfiguration und Regel die Kennzahlen je Instanz)
und sweep_meta.json. Rechenzeit: 27 Konfigurationen x 32 Regeln x 200 Instanzen x 120 Tage etwa 126 s auf 12 Prozessen, nie in der CI.

Aufruf (im Projektordner):  python tools/sweep.py [Ausgabeordner]"""
import json
import pathlib
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import irp_model  # noqa: E402
import irp_policy  # noqa: E402

N_INST = 200
D_DAYS = 120
CHUNK = 20
WORKERS = 12

# Regeln: Name -> (H, gamma, early)
POLICIES = {"R": (0.0, 0.5, 0.0)}
for H in (1, 2, 3, 4, 6, 8, 12):
    for g in (0.10, 0.25, 0.5, 1.0):
        POLICIES["P_H%d_g%.2f" % (H, g)] = (float(H), g, 0.0)
for L in (1, 2, 3):
    POLICIES["E_L%d" % L] = (0.0, 0.5, float(L))

CONFIGS = {
    "base": {},
    "sigma0": {"sigma": 0.0},
    "sigma015": {"sigma": 0.15},
    "sigma06": {"sigma": 0.6},
    "sigma10": {"sigma": 1.0},
    "q100": {"Q": 100.0},
    "q150": {"Q": 150.0},
    "q200": {"Q": 200.0},
    "q450": {"Q": 450.0},
    "q600": {"Q": 600.0},
    "q1000": {"Q": 1000.0},
    "n10": {"N": 10},
    "n40": {"N": 40, "Q": 300.0},
    "n40_q600": {"N": 40, "Q": 600.0},
    "cluster": {"clustered": True},
    "depot_far": {"depot": "far"},
    "depot_corner": {"depot": "corner"},
    "pen001": {"pfac": 0.01},
    "pen025": {"pfac": 0.25},
    "pen100": {"pfac": 1.0},
    "flotte_F1_Q300": {"F": 1, "Q": 300.0},
    "flotte_F1_Q250": {"F": 1, "Q": 250.0},
    "flotte_F2_Q150": {"F": 2, "Q": 150.0},
    "kappa0": {"kappa": 0.0},
    "kappa2": {"kappa": 2.0},
    "tank_kurz": {"tank_lo": 4.0, "tank_hi": 7.0},
    "tank_lang": {"tank_lo": 14.0, "tank_hi": 24.0},
}
METRICS = ["routing", "short", "dI", "delivered", "served", "n_routes", "n_visits", "opt_visits", "stockout_days", "deferred",
           "route_days", "util_sum"]


def run_chunk(args):
    cname, lo, hi = args
    kw = CONFIGS[cname]
    out = {p: np.zeros((len(METRICS), hi - lo)) for p in POLICIES}
    pen = None
    for j, s in enumerate(range(lo, hi)):
        inst = irp_model.make_instance(s, D=D_DAYS, **kw)
        pen = inst.p
        for pname, (H, g, e) in POLICIES.items():
            r = irp_policy.simulate(inst, H, g, early=e)
            for m, key in enumerate(METRICS):
                out[pname][m, j] = r[key]
    return cname, lo, hi, out, pen


def main(argv):
    out_dir = pathlib.Path(argv[0]) if argv else ROOT / "tools" / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    names = list(CONFIGS)
    tasks = [(c, lo, min(lo + CHUNK, N_INST)) for c in names for lo in range(0, N_INST, CHUNK)]
    res = {c: {p: np.zeros((len(METRICS), N_INST)) for p in POLICIES} for c in names}
    pens = {}
    with Pool(WORKERS) as pool:
        done = 0
        for cname, lo, hi, out, pen in pool.imap_unordered(run_chunk, tasks):
            for p in POLICIES:
                res[cname][p][:, lo:hi] = out[p]
            pens[cname] = pen
            done += 1
            if done % 20 == 0:
                print("%d/%d %.0f s" % (done, len(tasks), time.time() - t0), flush=True)
    flat = {}
    for c in names:
        for p in POLICIES:
            flat["%s|%s" % (c, p)] = res[c][p]
    np.savez_compressed(out_dir / "raw_sweep.npz", **flat)
    (out_dir / "sweep_meta.json").write_text(json.dumps(
        {"metrics": METRICS, "policies": {k: list(v) for k, v in POLICIES.items()}, "configs": {k: {a: b for a, b in v.items()} for k, v in CONFIGS.items()},
         "n_inst": N_INST, "days": D_DAYS, "penalty": pens}, indent=1), encoding="utf-8")
    print("fertig %.0f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
