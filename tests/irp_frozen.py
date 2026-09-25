"""Test-Hilfsmodul: die eingefrorenen Instanzen und Referenzwerte (tests/data/irp_reference.json, irp_oracle_frozen.json) laden.

Eine eingefrorene Instanz enthält Koordinaten, Distanzen, mu, C, I0 und den kompletten Verbrauchsstrom verlustfrei (float64 in Base64): kein Test, der sie
benutzt, hängt vom Zufallsgenerator oder von der NumPy-Version ab (die Ströme von default_rng dürfen sich zwischen Versionen ändern)."""
import base64
import functools
import json
import pathlib

import numpy as np

import irp_model

DATA = pathlib.Path(__file__).resolve().parent / "data"
INST_ATTRS = ("N", "D", "F", "Q", "size", "sigma", "kappa", "p")
INST_ARRAYS = ("xy", "dist", "mu", "C", "I0", "cons")
COUNT_METRICS = ("n_routes", "n_visits", "opt_visits", "stockout_days", "deferred", "route_days")   # ganzzahlig: immer exakt


def dec(d):
    return np.frombuffer(base64.b64decode(d["b64"]), dtype="<f8").reshape(d["shape"]).copy()


def inst_from_json(d):
    inst = irp_model.Inst()
    for k in INST_ATTRS:
        setattr(inst, k, d[k])
    for k in INST_ARRAYS:
        setattr(inst, k, dec(d[k]))
    return inst


@functools.lru_cache(maxsize=1)
def reference():
    return json.loads((DATA / "irp_reference.json").read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=1)
def oracle_frozen():
    return json.loads((DATA / "irp_oracle_frozen.json").read_text(encoding="utf-8"))


def case(config, seed=0):
    """Der eingefrorene Fall (config, seed) als dict mit den Feldern config, seed, overrides, expected und dem Objekt `inst`."""
    for c in reference()["cases"]:
        if c["config"] == config and c["seed"] == seed:
            return dict(c, inst=inst_from_json(c["instance"]))
    raise KeyError((config, seed))


def frozen(config, seed=0):
    """Nur die eingefrorene Instanz."""
    return case(config, seed)["inst"]
