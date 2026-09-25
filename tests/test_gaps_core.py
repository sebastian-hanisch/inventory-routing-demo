"""Tests, die im ersten vollständigen Fehler-Einbau-Lauf überlebende Mutanten des Kerns schließen (Tourenbausteine, Regeln, Instanz, exakter Maßstab, Live-Instanz): Gleichstände der
Savings-Reihenfolge, Rechentoleranzen (1e-9 der Kapazität, 1e-12 der Schwellen), Vorgabewerte der Parameter, Randfälle ohne Verbrauch. Jeder Test läuft auf der unveränderten Kopie."""
import numpy as np
import pytest

import irp_frozen as FZ
import irp_live as LV
import irp_model as M
import irp_oracle as OR
import irp_policy as P
import irp_routing as RT
from test_model_units import line, mini, routes_of

# ---------------------------------------------------------------------------------------------------
# Tourenbausteine
# ---------------------------------------------------------------------------------------------------
SQUARE = mini(xy=[(0, 0), (10, 0), (0, 10), (-10, 0), (0, -10)], mu=[0, 1, 1, 1, 1], C=[0, 9, 9, 9, 9], I0=[0, 5, 5, 5, 5], cons=[[0, 1, 1, 1, 1]])


@pytest.mark.parametrize("perm,want", [((3, 1, 4, 2), [(1, 2), (3, 4)]), ((2, 1, 3, 4), [(1, 4), (2, 3)]), ((3, 2, 1, 4), [(1, 4), (3, 2)]), ((2, 3, 1, 4), [(1, 4), (2, 3)]),
                                       ((3, 4, 1, 2), [(1, 2), (3, 4)]), ((4, 2, 3, 1), [(2, 1), (4, 3)]), ((1, 3, 4, 2), [(1, 2), (3, 4)]), ((3, 2, 4, 1), [(2, 1), (3, 4)])])
def test_savings_ties_are_broken_by_the_customer_numbers_not_by_the_list_order(perm, want):
    """Im Quadrat sind alle Nachbarpaare gleich weit (exakt gleiche Ersparnis); bei Platz für zwei Kunden je Wagen entscheidet allein die Reihenfolge (Ersparnis, Nummer i, Nummer j)."""
    qty = {i: 10.0 for i in range(1, 5)}
    assert sorted(map(tuple, RT.savings_routes(SQUARE.dist, list(perm), qty, 20.0))) == want


def test_savings_capacity_tolerance_is_one_nanounit():
    inst = mini(xy=[(0, 0), (10, 0), (11, 0)], mu=[0, 1, 1], C=[0, 9, 9], I0=[0, 5, 5], cons=[[0, 1, 1]])
    assert sorted(map(sorted, RT.savings_routes(inst.dist, [1, 2], {1: 10.0, 2: 10.0 + 5e-10}, 20.0))) == [[1, 2]]                 # 5e-10 über der Kapazität: noch erlaubt
    assert sorted(map(sorted, RT.savings_routes(inst.dist, [1, 2], {1: 10.0, 2: 10.0 + 2e-9}, 20.0))) == [[1], [2]]                # 2e-9 darüber: nicht mehr


NEAR_TIE_XY = [[0.0, 0.0], [-9.284901146110741, 2.1265544949039414], [-7.699337729942127, -5.608192385531572], [-5.2591557132703874, -1.0780863327263985]]


def test_cheapest_insertion_keeps_the_first_position_on_a_numerical_near_tie():
    """Spiegelsymmetrische, gedrehte Lage: die Einfügung vor Kunde 1 und nach Kunde 2 kosten mathematisch gleich, numerisch ist die spätere um 2e-15 billiger. Die Toleranz von
    1e-12 hält die erste Stelle (Verhalten unabhängig vom Rauschen der letzten Stellen)."""
    inst = mini(xy=NEAR_TIE_XY, mu=[0, 1, 1, 1], C=[0, 9, 9, 9], I0=[0, 5, 5, 5], cons=[[0, 1, 1, 1]])
    d = inst.dist
    first, last = d[0, 3] + d[3, 1] - d[0, 1], d[2, 3] + d[3, 0] - d[2, 0]
    if not (0 < first - last < 1e-12):
        pytest.skip("die Rundung der Plattform erzeugt hier kein Rauschen")
    delta, pos = RT.cheapest_insertion(d, [1, 2], 3)
    assert pos == 0 and delta == first


def test_pickup_prefers_the_first_route_on_a_numerical_near_tie():
    """Zwei getrennte Touren [1] und [2] (Wagen 40, je Ladung 25), Mitnehmer 3 (Menge 10) passt in beide zu mathematisch gleichen, numerisch um 2e-15 verschiedenen Kosten:
    die Toleranz 1e-12 hält die erste Tour."""
    inst = mini(xy=NEAR_TIE_XY, mu=[0, 10, 10, 10], C=[0, 30, 30, 30], I0=[0, 5, 5, 15], cons=[[0, 1, 1, 1]], Q=40.0, F=2)
    d = inst.dist
    a = min(d[0, 3] + d[3, 1] - d[0, 1], d[1, 3] + d[3, 0] - d[1, 0])
    b = min(d[0, 3] + d[3, 2] - d[0, 2], d[2, 3] + d[3, 0] - d[2, 0])
    if not (0 < a - b < 1e-12):
        pytest.skip("die Rundung der Plattform erzeugt hier kein Rauschen")
    plan, st = P.decide(inst, inst.I0.copy(), H=1.0, gamma=1.0)
    with_3 = [r for r in routes_of(plan) if 3 in r]
    assert st["opt_visits"] == 1 and sorted(map(len, routes_of(plan))) == [1, 2] and len(with_3) == 1 and 1 in with_3[0]


# ---------------------------------------------------------------------------------------------------
# Entscheidung: Toleranzen und Vorgabewerte
# ---------------------------------------------------------------------------------------------------
def test_due_and_candidate_thresholds_keep_a_margin_of_1e_12():
    inst = line(Q=300.0)
    just_under = np.array([0, 10 - 5e-13, 50, 50, 50.0])                                        # 5e-13 unter der Meldegrenze: gilt als nicht fällig
    assert P.decide(inst, just_under)[0] == []
    assert routes_of(P.decide(inst, np.array([0, 10 - 2e-12, 50, 50, 50.0]))[0]) == [[1]]           # 2e-12 darunter: fällig
    cand_edge = np.array([0, 5, 50, 20 - 5e-13, 50.0])                                              # Restreichweite genau am Rand der Vorschau (H = 1): kein Kandidat
    assert P.decide(inst, cand_edge, H=1.0, gamma=10.0)[1]["opt_visits"] == 0
    assert P.decide(inst, np.array([0, 5, 50, 20 - 2e-12, 50.0]), H=1.0, gamma=10.0)[1]["opt_visits"] == 1


def test_pickup_wagon_tolerance_and_exact_fit():
    I = np.array([0, 5, 25, 15, 8.0])
    assert P.decide(line(Q=272.0), I, H=1.0, gamma=1.0)[1]["opt_visits"] == 1                       # 187 + 85 = 272 = Q: passt genau
    assert P.decide(line(Q=271.0), I, H=1.0, gamma=1.0)[1]["opt_visits"] == 0
    over = np.array([0, 5, 25, 15 - 5e-10, 8.0])                                                       # Menge 85 + 5e-10: 5e-10 über der Kapazität, noch erlaubt
    assert P.decide(line(Q=272.0), over, H=1.0, gamma=1.0)[1]["opt_visits"] == 1
    over2 = np.array([0, 5, 25, 15 - 3e-9, 8.0])
    assert P.decide(line(Q=272.0), over2, H=1.0, gamma=1.0)[1]["opt_visits"] == 0


def test_pickup_cost_threshold_keeps_a_margin_of_1e_12():
    inst = line(Q=300.0)
    I = np.array([0, 5, 25, 15, 8.0])
    delta = RT.cheapest_insertion(inst.dist, [1, 4], 3)[0]
    gamma = (delta - 5e-13) / (2.0 * inst.dist[0, 3])
    assert P.decide(inst, I, H=1.0, gamma=gamma)[1]["opt_visits"] == 1                               # 5e-13 zu teuer: innerhalb der Toleranz
    assert P.decide(inst, I, H=1.0, gamma=(delta - 5e-9) / (2.0 * inst.dist[0, 3]))[1]["opt_visits"] == 0


def test_default_arguments_are_reactive_with_gamma_one_half():
    inst = line(Q=300.0)
    I = np.array([0, 5, 25, 10.5, 8.0])                                                             # Kunde 3 liegt 0,05 Tage über der Meldegrenze
    assert P.decide(inst, I) == P.decide(inst, I, H=0.0, gamma=0.5, fleet=None, early=0.0) and P.decide(inst, I)[1]["opt_visits"] == 0
    two = mini(xy=[(0, 0), (10, 0), (10, 11)], mu=[0, 10, 10], C=[0, 100, 100], I0=[0, 5, 15], cons=[[0, 10, 10]], Q=300.0)          # Einfügeverhältnis 0,53: über 0,5, unter 0,55
    I2 = np.array([0, 5, 15.0])
    assert P.decide(two, I2, H=1.0)[1]["opt_visits"] == 0 and P.decide(two, I2, H=1.0, gamma=0.5)[1]["opt_visits"] == 0 and P.decide(two, I2, H=1.0, gamma=0.55)[1]["opt_visits"] == 1


def test_simulate_defaults_are_reactive_with_gamma_one_half():
    inst = FZ.frozen("base", 0)
    reactive = P.simulate(inst, 0.0, 0.5)
    assert P.simulate(inst)["routing"] == reactive["routing"] and P.simulate(inst)["opt_visits"] == 0
    assert P.simulate(inst, 3.0)["routing"] == P.simulate(inst, 3.0, 0.5)["routing"] and P.simulate(inst, 3.0)["routing"] != P.simulate(inst, 3.0, 0.55)["routing"]


def test_replay_reports_no_deferrals_or_pickups_of_its_own():
    inst = FZ.frozen("flotte_F1_Q300", 0)
    run = P.simulate(inst, 0.0, record=True)
    assert run["deferred"] > 0
    rp = P.replay(inst, run["plans"])
    assert rp["deferred"] == 0 and rp["opt_visits"] == 0 and rp["routing"] == run["routing"]


def test_make_instance_default_horizon_is_ninety_days():
    assert M.make_instance(1).D == 90 and M.make_instance(1).cons.shape[0] == 90 and M.make_instance(1, D=5).cons.shape[0] == 5


# ---------------------------------------------------------------------------------------------------
# Exakter Maßstab
# ---------------------------------------------------------------------------------------------------
def test_oracle_defaults_and_bookkeeping():
    pytest.importorskip("ortools")
    assert OR.make_oracle_instance(3).Q == 150.0 and OR.make_oracle_instance(3, Q=250.0).Q == 250.0
    inst = OR.make_oracle_instance(0)
    res = OR.solve_oracle(inst, OR.TAU, time_limit=60)
    assert 0.0 <= res["time"] < 600.0 and res["status"] == "OPTIMAL"


def test_compare_small_evaluates_each_rule_with_its_own_parameters():
    pytest.importorskip("ortools")
    inst = OR.make_oracle_instance(0)
    opt = OR.solve_oracle(inst, OR.TAU, time_limit=60)
    cmp_ = OR.compare_small(inst, opt, 8, 0.25, 1)
    for key, (h, g, e) in {"R": (0.0, 0.5, 0.0), "P": (8.0, 0.25, 0.0), "E": (0.0, 0.5, 1.0)}.items():
        r = P.simulate(inst, h, g, early=e, record=True)
        assert cmp_[key]["J"] == P.objective(r, inst.p, OR.TAU) and cmp_[key]["routes"] == r["n_routes"] and cmp_[key]["plans"] == r["plans"]
    assert cmp_["optimum"]["J"] == P.objective(P.replay(inst, opt["plans"]), inst.p, OR.TAU)
    other = OR.compare_small(inst, opt, 3, 0.5, 2)
    assert other["P"]["J"] != cmp_["P"]["J"] and other["E"]["J"] != cmp_["E"]["J"] and other["R"]["J"] == cmp_["R"]["J"]


# ---------------------------------------------------------------------------------------------------
# Live-Instanz
# ---------------------------------------------------------------------------------------------------
def _live_of(monkeypatch, inst, **over):
    monkeypatch.setattr(LV, "make_live_instance", lambda *a, **k: inst)
    return LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)


def test_live_without_any_consumption_has_zero_cost_and_zero_gain(monkeypatch):
    inst = FZ.frozen("base", 0)
    inst.mu = np.zeros_like(inst.mu)
    inst.cons = np.zeros_like(inst.cons)
    live = _live_of(monkeypatch, inst)
    assert live["tau"] == 0.0 and all(rl["J"] == 0.0 and rl["gain"] == 0.0 for rl in live["rules"].values())


def test_live_tau_is_the_cost_rate_even_for_tiny_deliveries(monkeypatch):
    inst = mini(xy=[(0, 0), (10, 0), (0, 20)], mu=[0, 0.001, 0.001], C=[0, 0.005, 0.005], I0=[0, 0.001, 0.001], cons=[[0, 0.001, 0.001]] * 30, Q=100.0, F=2)
    live = _live_of(monkeypatch, inst)
    r = live["rules"]["R"]["res"]
    assert 0 < r["delivered"] < 1 and live["tau"] == pytest.approx(r["routing"] / r["delivered"]) and live["tau"] > 1.0


def test_pickup_days_default_limit_is_eight_days(monkeypatch):
    live = _live_of(monkeypatch, FZ.frozen("base", 0))
    assert len(LV.pickup_days(live)) == 8 and len(LV.pickup_days(live, 9)) == 9 and LV.pickup_days(live) == LV.pickup_days(live, 9)[:8]


def test_day_view_quantity_is_capped_by_the_wagon_on_small_wagons(monkeypatch):
    inst = FZ.frozen("q100", 0)
    live = _live_of(monkeypatch, inst)
    seen_capped = False
    for d in range(1, 121):
        v = LV.day_view(live, "R", d)
        for tour in v["tours"]:
            for s in tour["stops"]:
                assert s["qty"] <= inst.Q + 1e-9 and s["qty"] == pytest.approx(min(inst.C[s["customer"]] - s["fill_before"], inst.Q))
                seen_capped |= s["qty"] == inst.Q < inst.C[s["customer"]] - s["fill_before"]
    assert seen_capped                                                                              # auf dem kleinen Wagen begrenzt der Wagen die Menge


def test_stop_kind_keeps_a_margin_of_1e_12_below_the_threshold(monkeypatch):
    live = _live_of(monkeypatch, FZ.frozen("base", 0))
    thr = LV.threshold(live)[3]
    assert LV.stop_kind(live, "P", thr - 5e-13, 3) == "Mitnahme" and LV.stop_kind(live, "P", thr - 2e-12, 3) == "fällig"
