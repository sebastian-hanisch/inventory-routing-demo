"""Bitgleichheit zur Messreihe: die mechanisch aus ir.py abgeleiteten Module (irp_model / irp_routing / irp_policy) müssen die Kennzahlen des Sweeps
(raw_sweep.npz der Messreihe) auf den EINGEFRORENEN Instanzen reproduzieren: 14 Instanzen aus 12 Konfigurationen, alle 32 Regeln, alle 12 Kennzahlen.

Die Instanzen sind eingefroren (Koordinaten, mu, C, I0 und Verbrauchsstrom), damit kein Test vom Zufallsgenerator abhängt: die CI installiert immer das neueste
NumPy, und die Ströme von default_rng dürfen sich zwischen Versionen ändern. Ganzzahlige Kennzahlen (Touren, Besuche, Mitnahmen ...) müssen immer exakt stimmen; die
Gleitkommasummen dürfen sich auf einer anderen Plattform in der letzten Stelle unterscheiden (NumPy summiert Felder je nach Version paarweise oder blockweise):
dort gilt ein relativer Abstand von 1e-9. Auf derselben Plattform und NumPy-Version, mit der eingefroren wurde, gilt zusätzlich strikte Gleichheit."""
import sys

import numpy as np
import pytest

import irp_frozen as FZ
import irp_model as M
import irp_policy as P

REF = FZ.reference()
CASES = [(c["config"], c["seed"]) for c in REF["cases"]]
SAME_PLATFORM = np.__version__ == REF["numpy"] and sys.platform == REF["platform"]


def run_case(config, seed):
    c = FZ.case(config, seed)
    out = {}
    for pol, (H, g, e) in REF["policies"].items():
        r = P.simulate(c["inst"], H, g, early=e)
        out[pol] = [r[k] for k in REF["metrics"]]
    return c, out


@pytest.mark.parametrize("config,seed", CASES, ids=[f"{c}-{s}" for c, s in CASES])
def test_simulation_reproduces_the_measured_metrics(config, seed):
    c, out = run_case(config, seed)
    assert list(out) == list(c["expected"]) and len(out) == 32
    for pol, values in out.items():
        for key, got, want in zip(REF["metrics"], values, c["expected"][pol]):
            if key in FZ.COUNT_METRICS:
                assert got == want, (config, seed, pol, key, got, want)
            else:
                assert got == pytest.approx(want, rel=1e-9, abs=1e-9), (config, seed, pol, key, got, want)


@pytest.mark.skipif(not SAME_PLATFORM, reason="strikte Gleichheit nur mit derselben NumPy-Version und Plattform wie beim Einfrieren")
@pytest.mark.parametrize("config,seed", CASES, ids=[f"{c}-{s}" for c, s in CASES])
def test_simulation_is_bit_identical_on_the_freezing_platform(config, seed):
    c, out = run_case(config, seed)
    for pol, values in out.items():
        assert values == c["expected"][pol], (config, seed, pol)


@pytest.mark.skipif(np.__version__ != REF["numpy"], reason="der Zufallsstrom kann sich zwischen NumPy-Versionen ändern: make_instance nur mit der Einfrier-Version prüfen")
@pytest.mark.parametrize("config,seed", CASES, ids=[f"{c}-{s}" for c, s in CASES])
def test_make_instance_reproduces_the_frozen_instance_with_the_same_numpy(config, seed):
    c = FZ.case(config, seed)
    fresh = M.make_instance(seed, D=REF["days"], **c["overrides"])
    for k in FZ.INST_ATTRS:
        assert getattr(fresh, k) == getattr(c["inst"], k), k
    for k in FZ.INST_ARRAYS:
        assert np.array_equal(getattr(fresh, k), getattr(c["inst"], k)), k


def test_the_reference_covers_every_rule_and_metric_and_the_cell_variety():
    assert len(REF["policies"]) == 32 and len(REF["metrics"]) == 12 and REF["days"] == 120
    assert sum(1 for p in REF["policies"] if p.startswith("P_")) == 28 and sum(1 for p in REF["policies"] if p.startswith("E_")) == 3
    assert {c for c, _ in CASES} >= {"base", "sigma0", "q100", "q600", "n10", "n40", "cluster", "depot_far", "flotte_F1_Q300", "flotte_F2_Q150", "pen100", "kappa2"}
    assert len(CASES) == 14 and [s for c, s in CASES if c == "base"] == [0, 1, 2]
    for c in REF["cases"]:
        inst = FZ.inst_from_json(c["instance"])
        assert inst.cons.shape == (REF["days"], inst.N + 1) and inst.xy.shape == (inst.N + 1, 2) and inst.dist.shape == (inst.N + 1, inst.N + 1)


def test_frozen_instances_pin_down_the_measured_mechanism():
    """Auf den eingefrorenen Basisinstanzen bündelt P(3; 0,5) tatsächlich (Mitnahmen > 0), R nie, und E(2) fährt mehr Besuche ohne Mitnahmen."""
    for s in (0, 1, 2):
        c = FZ.case("base", s)
        e = {p: dict(zip(REF["metrics"], v)) for p, v in c["expected"].items()}
        assert e["R"]["opt_visits"] == 0 and e["P_H3_g0.50"]["opt_visits"] > 0 and e["E_L2"]["opt_visits"] == 0
        assert e["P_H3_g0.50"]["n_routes"] < e["R"]["n_routes"] and e["E_L2"]["n_visits"] > e["R"]["n_visits"]
        assert e["P_H3_g0.10"]["opt_visits"] < e["P_H3_g1.00"]["opt_visits"]
