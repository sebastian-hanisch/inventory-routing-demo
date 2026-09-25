"""Erzeugt data/irp_results.json aus den Auswertungen der Messreihe (nur lesend): sweep_data.json (Aggregate der 27 Konfigurationen, von
tools/analyze.py) und oracle_data_Q150.json / oracle_data_Q250.json (Kleininstanz gegen CP-SAT, von tools/oracle_run.py). Nur das, was die App
zeigt: keine Rohdaten je Instanz, Zahlen auf 4 Stellen gerundet, feste Schlüsselreihenfolge, deterministische Ausgabe.

Aufruf (im Projektordner):  python tools/build_results.py [Quellordner] [sweep_data.json]
  Quellordner: enthält sweep_meta.json, oracle_data_Q150.json, oracle_data_Q250.json (Standard ../bestand-planung/messreihe_inventory_routing)
  sweep_data.json: Standard tools/_out/sweep_data.json (aus tools/analyze.py, mit short_cmp), sonst die Datei im Quellordner"""
import json
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "irp_results.json"
DEFAULT_SOURCES = ROOT.parent / "bestand-planung" / "messreihe_inventory_routing"
DIGITS = 4

BASE_CFG = {"N": 20, "Q": 300.0, "F": 6, "sigma": 0.3, "pfac": 0.05, "kappa": 1.0, "depot": "center", "clustered": False, "tank_lo": 8.0, "tank_hi": 14.0}
GAIN_KEYS = ("mean", "se", "median", "q1", "q3", "share_loss")
CELL_KEYS = ("tau", "JR", "Q_over_qL", "qL", "short_R", "short_P", "cv_gain", "cv_chosen", "best_insample", "routes_R", "routes_P", "visits_R", "visits_P",
             "opt_visits_P", "stockout_R", "stockout_P", "deferred_R", "util_R", "util_P", "km_R", "km_P", "short_cmp", "routes_E", "visits_E", "km_E", "util_E", "short_E", "stockout_E")
ORACLE_AGG_KEYS = ("n", "n_optimal", "tau", "R_gap_to_oracle", "P_H3_g0.50_gap_to_oracle", "P_H8_g0.25_gap_to_oracle", "E_L1_gap_to_oracle", "P_H3_g0.50_gain_over_R",
                   "P_H8_g0.25_gain_over_R", "P_H3_g0.50_share_of_R_gap_closed", "P_H8_g0.25_share_of_R_gap_closed", "oracle_gain_over_R")
ORACLE_ROW_KEYS = ("seed", "status", "time", "J_oracle", "oracle_routes", "oracle_visits", "oracle_short", "R", "R_routes", "R_short", "P_H3_g0.50",
                   "P_H3_g0.50_routes", "P_H3_g0.50_short", "P_H8_g0.25", "P_H8_g0.25_routes", "P_H8_g0.25_short", "E_L1", "E_L1_routes", "E_L1_short")


def rnd(x, digits=DIGITS):
    """Rekursiv runden; bool, None und Text bleiben, Tupel werden Listen."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, float):
        return round(x, digits)
    if isinstance(x, int):
        return x
    if isinstance(x, dict):
        return {k: rnd(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [rnd(v, digits) for v in x]
    return x


def _pick(d, keys):
    return {k: d[k] for k in keys if k in d}


def build_cell(name, overrides, penalty, cell):
    cfg = dict(BASE_CFG, **overrides)
    cfg["penalty"] = round(cfg["pfac"] * 100.0, 6)
    assert abs(cfg["penalty"] - penalty) < 1e-9, (name, cfg["penalty"], penalty)
    out = {"name": name, "cfg": cfg}
    out.update(_pick(cell, CELL_KEYS))
    out["default"] = _pick(cell["default"], GAIN_KEYS)
    out["best_insample_gain"] = _pick(cell["best_insample_gain"], GAIN_KEYS)
    out["P"] = {k: _pick(v, GAIN_KEYS) for k, v in cell["P"].items()}
    out["E"] = {k: _pick(v, GAIN_KEYS) for k, v in cell["E"].items()}
    return out


def build_oracle(data):
    return {"agg": _pick(data["agg"], ORACLE_AGG_KEYS), "rows": [_pick(r, ORACLE_ROW_KEYS) for r in data["rows"]]}


def build(meta, sweep, oracle150, oracle250):
    cells = [build_cell(name, meta["configs"][name], meta["penalty"][name], sweep["configs"][name]) for name in meta["configs"]]
    out = {"_meta": {"n_inst": meta["n_inst"], "days": meta["days"], "default": {"H": 3, "gamma": 0.5}, "policies": list(meta["policies"]),
                     "spearman": sweep["spearman_gain_vs_Q_over_qL"], "base_distribution": sweep["base_distribution"],
                     "n_negative_pairs": sweep["n_negative_pairs"], "E_negative_pairs": sweep["E_negative_pairs"]},
           "cells": cells, "oracle": {"Q150": build_oracle(oracle150), "Q250": build_oracle(oracle250)}}
    return rnd(out)


def dumps(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n"


def main(argv):
    sources = pathlib.Path(argv[0]) if argv else DEFAULT_SOURCES
    sweep_path = pathlib.Path(argv[1]) if len(argv) > 1 else (ROOT / "tools" / "_out" / "sweep_data.json")
    if not sweep_path.exists():
        sweep_path = sources / "sweep_data.json"
    read = lambda p: json.loads(pathlib.Path(p).read_text(encoding="utf-8"))
    out = build(read(sources / "sweep_meta.json"), read(sweep_path), read(sources / "oracle_data_Q150.json"), read(sources / "oracle_data_Q250.json"))
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(dumps(out), encoding="utf-8", newline="\n")
    print(f"geschrieben: {OUT} ({OUT.stat().st_size // 1024} KB, {len(out['cells'])} Zellen)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
