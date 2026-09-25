"""Exakter Maßstab (CP-SAT): das Modell gegen die unabhängige exakte Rechnung (in tests/test_checks.py), die eingefrorenen Kleininstanzen der Messreihe (nur der
ZIELWERT wird geprüft: er ist eindeutig, der Plan nicht - bei Gleichstand kann CP-SAT einen anderen, gleich guten Plan liefern; in der Messreihe hatte Seed 19 bei
Q = 150 im zweiten Lauf 11 statt 12 Besuche bei gleichem Zielwert), Größenbegrenzung, Pause zwischen zwei Aufrufen, Vergleich mit den Regeln. Keine Wall-Clock-Annahmen."""
import numpy as np
import pytest

import irp_frozen as FZ
import irp_model as M
import irp_oracle as OR
import irp_policy as P

pytest.importorskip("ortools")
FROZEN = FZ.oracle_frozen()
CASES = [(c["wagon"], c["seed"]) for c in FROZEN["cases"]]


def frozen_case(wagon, seed):
    c = next(c for c in FROZEN["cases"] if c["wagon"] == wagon and c["seed"] == seed)
    return c, FZ.inst_from_json(c["instance"])


@pytest.mark.parametrize("wagon,seed", CASES, ids=[f"Q{w}-{s}" for w, s in CASES])
def test_optimum_value_of_the_frozen_small_instances_equals_the_measured_one(wagon, seed):
    c, inst = frozen_case(wagon, seed)
    res = OR.solve_oracle(inst, FROZEN["tau"], time_limit=60, workers=8)
    assert res is not None and res["status"] == "OPTIMAL" == c["expected"]["status"]
    rep = P.replay(inst, res["plans"])
    j_opt = P.objective(rep, inst.p, FROZEN["tau"])
    assert j_opt == pytest.approx(c["expected"]["J_oracle"], abs=1e-6)                      # nur der Zielwert, nicht der Plan
    assert res["obj"] + FROZEN["tau"] * rep["I_start"] == pytest.approx(c["expected"]["J_oracle_model"], abs=1e-6) and rep["short"] == 0.0
    assert res["bound"] == pytest.approx(res["obj"]) and res["tau"] == FROZEN["tau"] and len(res["plans"]) == inst.D


@pytest.mark.parametrize("wagon,seed", CASES[:4], ids=[f"Q{w}-{s}" for w, s in CASES[:4]])
def test_rules_on_the_frozen_small_instances_reproduce_the_measured_costs_and_never_beat_the_optimum(wagon, seed):
    c, inst = frozen_case(wagon, seed)
    e = c["expected"]
    for name, (H, g, early) in {"R": (0.0, 0.5, 0.0), "P_H3_g0.50": (3.0, 0.5, 0.0), "P_H8_g0.25": (8.0, 0.25, 0.0), "E_L1": (0.0, 0.5, 1.0)}.items():
        r = P.simulate(inst, H, g, early=early)
        assert P.objective(r, inst.p, FROZEN["tau"]) == pytest.approx(e[name], abs=1e-6), name
        if r["short"] == 0.0:
            assert P.objective(r, inst.p, FROZEN["tau"]) >= e["J_oracle"] - 1e-6, name       # ohne Fehlmenge nie unter dem bewiesenen Optimum
    assert P.simulate(inst, 0.0)["n_routes"] == e["R_routes"] and P.simulate(inst, 3.0, 0.5)["n_routes"] == e["P_H3_g0.50_routes"]


def test_the_optimum_plan_is_feasible_replays_to_the_model_value_and_uses_one_tour_per_day():
    _, inst = frozen_case(150, 0)
    res = OR.solve_oracle(inst, 0.6, time_limit=60, workers=8)
    assert all(len(p) <= 1 for p in res["plans"]) and res["status"] == "OPTIMAL"
    for plan in res["plans"]:
        for route in plan:
            assert sum(q for _, q in route) <= inst.Q + 1e-9 and len({c for c, _ in route}) == len(route) and all(q >= 0 and q == int(q) for _, q in route)
    rep = P.replay(inst, res["plans"])
    assert rep["short"] == 0.0 and rep["n_routes"] == sum(len(p) for p in res["plans"])


def test_an_infeasible_small_instance_returns_none():
    inst = M.make_instance(1, N=3, D=3, sigma=0.0, Q=2.0, F=1, round_dist=0.1)
    inst.mu = np.array([0.0, 5.0, 5.0, 5.0])
    inst.C = np.array([0.0, 6.0, 6.0, 6.0])
    inst.I0 = np.zeros(4)
    inst.cons = np.tile(inst.mu, (3, 1))
    inst.cons[:, 0] = 0.0
    assert OR.solve_oracle(inst, 0.6, time_limit=10, workers=2) is None                      # Tagesbedarf 15 gegen Wagen 2


def test_size_guard_rejects_large_instances_before_building_a_model():
    OR.check_size(OR.MAX_CUSTOMERS, OR.MAX_DAYS)
    with pytest.raises(ValueError, match="zu groß"):
        OR.check_size(OR.MAX_CUSTOMERS + 1, 3)
    with pytest.raises(ValueError, match="zu groß"):
        OR.check_size(3, OR.MAX_DAYS + 1)
    big = M.make_instance(0, N=20, D=120)
    with pytest.raises(ValueError):
        OR.solve_oracle(big, 0.6, time_limit=1)
    assert (OR.SMALL_CUSTOMERS, OR.SMALL_DAYS) == (7, 6) and OR.SMALL_CUSTOMERS <= OR.MAX_CUSTOMERS and OR.SMALL_DAYS <= OR.MAX_DAYS
    with pytest.raises(ValueError):
        OR.check_size(9, 6)


def test_cooldown_counts_down_from_the_last_call():
    assert OR.cooldown_left(None, 100.0, 4.0) == 0.0
    assert OR.cooldown_left(100.0, 101.0, 4.0) == pytest.approx(3.0) and OR.cooldown_left(100.0, 104.0, 4.0) == 0.0 and OR.cooldown_left(100.0, 200.0, 4.0) == 0.0
    assert OR.cooldown_left(100.0, 100.0, 4.0) == 4.0


def test_make_oracle_instance_is_integer_valued_and_small():
    inst = OR.make_oracle_instance(3, Q=250.0)
    assert (inst.N, inst.D, inst.F, inst.Q, inst.sigma) == (7, 6, 1, 250.0, 0.0)
    for a in (inst.mu, inst.C, inst.I0):
        assert np.array_equal(a, np.round(a))
    assert np.all(inst.C[1:] >= inst.mu[1:] + 1) and np.all(inst.I0 <= inst.C) and np.array_equal(inst.cons[:, 1:], np.tile(inst.mu[1:], (6, 1))) and np.all(inst.cons[:, 0] == 0)
    assert np.allclose(inst.dist / 0.1, np.round(inst.dist / 0.1))


def test_solve_small_compares_the_optimum_with_the_three_rules():
    res = OR.solve_small(150, 0, 3, 0.5, 2, time_limit=60)
    assert res["status"] == "OPTIMAL" and res["seed"] == 0 and res["wagon"] == 150
    cmp_ = res["compare"]
    assert set(cmp_) == {"optimum", "R", "P", "E"} and cmp_["optimum"]["short"] == 0.0
    for k in ("R", "P", "E"):
        assert cmp_[k]["gap"] == pytest.approx(100 * (cmp_[k]["J"] - cmp_["optimum"]["J"]) / cmp_["optimum"]["J"])
        if cmp_[k]["short"] == 0.0:
            assert cmp_[k]["gap"] >= -1e-6                                                       # keine Regel ohne Fehlmenge unter dem Optimum
        assert len(cmp_[k]["plans"]) == 6
    _, inst = frozen_case(150, 0)
    assert cmp_["R"]["J"] == pytest.approx(P.objective(P.simulate(inst, 0.0, record=True), inst.p, 0.6)) and cmp_["optimum"]["J"] == pytest.approx(FROZEN["cases"][0]["expected"]["J_oracle"], abs=1e-6)
    # H und gamma der Regel P gehen ein: mit H = 1 und gamma = 0,1 ist es (fast) reaktiv
    again = OR.solve_small(150, 0, 12, 1.0, 1, time_limit=60)["compare"]
    assert again["P"]["J"] != cmp_["P"]["J"] and again["R"]["J"] == cmp_["R"]["J"] and again["E"]["J"] != cmp_["E"]["J"]


def test_solve_small_returns_none_for_an_infeasible_instance(monkeypatch):
    monkeypatch.setattr(OR, "solve_oracle", lambda *a, **k: None)
    assert OR.solve_small(150, 0, 3, 0.5, 2) is None
