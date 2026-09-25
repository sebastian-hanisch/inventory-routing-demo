"""Erzeugt die eingefrorenen Referenzdaten der Tests aus den Rohdaten der Messreihe (nur lesend):

  tests/data/irp_reference.json       EINGEFRORENE INSTANZEN (Koordinaten, Distanzen, Verbrauchsrate mu, Tankgröße C, Anfangsbestand I0 und der komplette
                                      Verbrauchsstrom, verlustfrei als float64 in Base64) und je Instanz und Regel die Kennzahlen, so wie der Sweep der Messreihe
                                      sie gerechnet hat (raw_sweep.npz): mehrere Konfigurationen, Seeds und alle 32 Regeln
  tests/data/irp_oracle_frozen.json   eingefrorene Kleininstanzen (7 Kunden, 6 Tage) mit den Zielwerten des CP-SAT-Optimums und der Regeln aus den Orakel-Läufen

Warum eingefroren: die CI installiert immer das NEUESTE NumPy, und die Ströme von np.random.default_rng dürfen sich zwischen Versionen ändern. Die Tests
rechnen deshalb nur die reine Simulation (irp_policy, irp_routing) auf den eingefrorenen Instanzen und verlangen die Kennzahlen der Messreihe: der Nachweis, dass die
mechanische Aufteilung von ir.py in irp_model / irp_routing / irp_policy nichts an der Logik geändert hat. Beim Einfrieren wird zusätzlich geprüft, dass make_instance
die Instanz mit der lokalen NumPy-Version genau so erzeugt (dann gilt auch der Test der Stromtreue, sonst wird er übersprungen).

Aufruf (im Projektordner):  python tools/freeze_reference.py [Ordner_der_Messreihe]   (Standard ../bestand-planung/messreihe_inventory_routing)"""
import base64
import json
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import irp_model  # noqa: E402
import irp_oracle  # noqa: E402

DEFAULT_SOURCES = ROOT.parent / "bestand-planung" / "messreihe_inventory_routing"
# (Konfiguration der Messreihe, Seeds): 14 Instanzen, von Basis über kleine und große Wagen, dünne und dichte Instanzen, knappe Flotten bis Geometrie und Meldegrenze
CASES = [("base", [0, 1, 2]), ("sigma0", [0]), ("q100", [0]), ("q600", [0]), ("n10", [0]), ("n40", [0]), ("cluster", [0]), ("depot_far", [0]),
         ("flotte_F1_Q300", [0]), ("flotte_F2_Q150", [0]), ("pen100", [0]), ("kappa2", [0])]
ORACLE_SEEDS = [0, 1, 2, 3]
ORACLE_WAGONS = (150, 250)
INST_ATTRS = ("N", "D", "F", "Q", "size", "sigma", "kappa", "p")
INST_ARRAYS = ("xy", "dist", "mu", "C", "I0", "cons")


def enc(a):
    """Feld als Base64 seiner float64-Bytes (little endian), verlustfrei."""
    a = np.ascontiguousarray(a, dtype="<f8")
    return {"shape": list(a.shape), "b64": base64.b64encode(a.tobytes()).decode("ascii")}


def dec(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype="<f8").reshape(d["shape"]).copy()


def inst_to_json(inst):
    out = {k: (int(getattr(inst, k)) if k in ("N", "D", "F") else float(getattr(inst, k))) for k in INST_ATTRS}
    out.update({k: enc(getattr(inst, k)) for k in INST_ARRAYS})
    return out


def inst_from_json(d):
    inst = irp_model.Inst()
    for k in INST_ATTRS:
        setattr(inst, k, d[k])
    for k in INST_ARRAYS:
        setattr(inst, k, dec(d[k]))
    return inst


def same_instance(a, b):
    """Zwei Instanzen bitgleich (alle Felder)?"""
    return all(getattr(a, k) == getattr(b, k) for k in INST_ATTRS) and all(np.array_equal(getattr(a, k), getattr(b, k)) for k in INST_ARRAYS)


def main(argv):
    src = pathlib.Path(argv[0]) if argv else DEFAULT_SOURCES
    meta = json.loads((src / "sweep_meta.json").read_text(encoding="utf-8"))
    raw = np.load(src / "raw_sweep.npz")
    metrics, policies = meta["metrics"], meta["policies"]
    cases = []
    for name, seeds in CASES:
        for s in seeds:
            inst = irp_model.make_instance(s, D=meta["days"], **meta["configs"][name])
            expected = {pol: [float(raw[f"{name}|{pol}"][m, s]) for m in range(len(metrics))] for pol in policies}
            cases.append({"config": name, "overrides": meta["configs"][name], "seed": s, "instance": inst_to_json(inst), "expected": expected})
    ref = {"numpy": np.__version__, "platform": sys.platform, "metrics": metrics, "policies": policies, "days": meta["days"], "cases": cases}
    out = ROOT / "tests" / "data"
    out.mkdir(parents=True, exist_ok=True)
    (out / "irp_reference.json").write_text(json.dumps(ref, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")

    oracle = {"numpy": np.__version__, "tau": irp_oracle.TAU, "cases": []}
    for wagon in ORACLE_WAGONS:
        data = json.loads((src / f"oracle_data_Q{wagon}.json").read_text(encoding="utf-8"))
        for row in data["rows"]:
            if row["seed"] in ORACLE_SEEDS:
                inst = irp_oracle.make_oracle_instance(row["seed"], Q=float(wagon))
                oracle["cases"].append({"wagon": wagon, "seed": row["seed"], "instance": inst_to_json(inst), "expected": {
                    k: row[k] for k in ("status", "J_oracle", "J_oracle_model", "oracle_routes", "R", "R_routes", "P_H3_g0.50", "P_H3_g0.50_routes", "P_H8_g0.25",
                                        "E_L1", "R_short", "P_H3_g0.50_short")}})
    (out / "irp_oracle_frozen.json").write_text(json.dumps(oracle, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8", newline="\n")
    print(f"geschrieben: irp_reference.json ({(out / 'irp_reference.json').stat().st_size // 1024} KB, {len(cases)} Instanzen, numpy {np.__version__}), "
          f"irp_oracle_frozen.json ({(out / 'irp_oracle_frozen.json').stat().st_size // 1024} KB, {len(oracle['cases'])} Kleininstanzen)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
