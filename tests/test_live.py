"""Live-Instanz (irp_live): alle drei Regeln auf einer Instanz, Bestandsverlauf, Tagessicht, Mitnehmer. Die Zahlen laufen über eine EINGEFRORENE Instanz (die Fabrik der Instanz wird
ersetzt), damit nichts vom Zufallsgenerator abhängt; die Invarianten gelten für jede erzeugte Instanz."""
import numpy as np
import pytest

import irp_constants as C
import irp_frozen as FZ
import irp_live as LV
import irp_policy as P
import irp_routing as RT

REF_POL = {"R": "R", "P": "P_H3_g0.50", "E": "E_L2"}


@pytest.fixture
def frozen_live(monkeypatch):
    """solve_live auf der eingefrorenen Basisinstanz (Seed 0) statt auf einer neu gewürfelten."""
    case = FZ.case("base", 0)
    monkeypatch.setattr(LV, "make_live_instance", lambda *a, **k: case["inst"])
    live = LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)
    return live, case


def test_live_reproduces_the_measured_metrics_of_the_frozen_instance(frozen_live):
    live, case = frozen_live
    metrics = FZ.reference()["metrics"]
    for rule, pol in REF_POL.items():
        res = live["rules"][rule]["res"]
        for key in ("routing", "short", "delivered", "served", "n_routes", "n_visits", "opt_visits", "stockout_days", "deferred", "route_days", "util_sum"):
            assert res[key] == pytest.approx(case["expected"][pol][metrics.index(key)], rel=1e-9, abs=1e-9), (rule, key)
        assert res["dI"] == pytest.approx(case["expected"][pol][metrics.index("dI")], abs=1e-9)


def test_objective_gain_and_tau_follow_the_definition(frozen_live):
    live, _ = frozen_live
    r = live["rules"]["R"]["res"]
    assert live["tau"] == pytest.approx(r["routing"] / r["delivered"]) and live["p"] == 5.0 and live["rules"]["R"]["gain"] == 0.0
    for name, rl in live["rules"].items():
        res = rl["res"]
        assert rl["J"] == pytest.approx(res["routing"] + 5.0 * res["short"] - live["tau"] * (res["I_end"] - res["I_start"]))
        assert rl["gain"] == pytest.approx(100.0 * (live["rules"]["R"]["J"] - rl["J"]) / live["rules"]["R"]["J"])
    assert live["rules"]["P"]["gain"] > 5.0 > 0 > live["rules"]["E"]["gain"]                          # Basis: Bündeln spart, früher liefern kostet


def test_bundling_with_no_lookahead_is_reactive_and_the_early_rule_with_the_same_stream_differs(monkeypatch):
    inst = FZ.frozen("base", 1)
    monkeypatch.setattr(LV, "make_live_instance", lambda *a, **k: inst)
    live = LV.solve_live(20, 300, 6, 0.3, 5, 0, 0.5, 1, 1)                                             # H = 0: reaktiv
    assert live["rules"]["P"]["plans"] == live["rules"]["R"]["plans"] and live["rules"]["P"]["J"] == live["rules"]["R"]["J"] and live["rules"]["P"]["gain"] == 0.0
    assert live["rules"]["E"]["plans"] != live["rules"]["R"]["plans"]
    for rl in live["rules"].values():
        assert rl["res"]["served"] + rl["res"]["short"] == pytest.approx(inst.cons.sum())              # derselbe Verbrauchsstrom für alle Regeln


def test_the_instance_data_are_returned_for_the_maps(frozen_live):
    live, case = frozen_live
    inst = case["inst"]
    assert live["N"] == 20 and live["days"] == 120 and live["Q"] == 300.0 and live["F"] == 6 and live["sigma"] == 0.3 and live["kappa"] == 1.0
    assert np.array_equal(live["xy"], inst.xy) and np.array_equal(live["C"], inst.C) and np.array_equal(live["mu"], inst.mu) and np.array_equal(live["I0"], inst.I0)
    assert (live["H"], live["gamma"], live["early"], live["seed"]) == (3.0, 0.5, 2.0, 0)


def test_inventory_path_replays_the_recorded_plans(frozen_live):
    live, case = frozen_live
    inst = case["inst"]
    for rule, rl in live["rules"].items():
        I = rl["I"]
        assert I.shape == (121, 21) and np.array_equal(I[0], inst.I0) and rl["res"]["I_end"] == pytest.approx(float(I[-1][1:].sum()))
        assert np.all(I >= -1e-9) and np.all(I[:, 1:] <= inst.C[1:] + 1e-9) and np.all(I[:, 0] == 0)
        again = LV.inventory_path(inst, rl["plans"])
        assert np.array_equal(again, I)
        step = P.apply_day(inst, I[7], rl["plans"][7], inst.cons[7])[0]
        assert np.array_equal(step, I[8])


def test_make_live_instance_maps_every_control_to_the_measured_configuration():
    for pen, pfac in C.PENALTY_PFAC.items():
        inst = LV.make_live_instance(10, 150, 2, 0.6, pen, 5, days=7)
        assert inst.p == pen == pfac * 100.0 and (inst.N, inst.Q, inst.F, inst.sigma, inst.D, inst.kappa) == (10, 150.0, 2, 0.6, 7, 1.0)
    assert LV.make_live_instance(40, 1000, 1, 0.0, 5, 3).N == 40 and LV.make_live_instance(20, 300, 6, 0.3, 5, 3).D == 120 == C.DAYS
    a, b = LV.make_live_instance(20, 300, 6, 0.3, 5, 9), LV.make_live_instance(20, 300, 6, 0.3, 5, 9)
    assert np.array_equal(a.cons, b.cons) and np.array_equal(a.xy, b.xy)


@pytest.mark.parametrize("customers,capacity,fleet,sigma,penalty", [(10, 100, 1, 0.0, 1), (20, 300, 6, 0.3, 5), (40, 1000, 2, 1.0, 100), (40, 150, 1, 0.6, 25)])
def test_solve_live_invariants_hold_for_generated_instances(customers, capacity, fleet, sigma, penalty):
    live = LV.solve_live(customers, capacity, fleet, sigma, penalty, 12, 1.0, 3, 4)
    assert live["N"] == customers and set(live["rules"]) == {"R", "P", "E"} and live["tau"] > 0
    for name, rl in live["rules"].items():
        res = rl["res"]
        assert len(rl["plans"]) == 120 and len(res["day_route_km"]) == 120 and rl["I"].shape == (121, customers + 1)
        assert res["n_routes"] == sum(len(p) for p in rl["plans"]) and max(len(p) for p in rl["plans"]) <= fleet
        assert res["served"] + res["short"] == pytest.approx(LV.make_live_instance(customers, capacity, fleet, sigma, penalty, 4).cons.sum())
        for plan in rl["plans"]:
            seen = [c for r in plan for c, _ in r]
            assert len(seen) == len(set(seen)) and all(sum(q for _, q in r) <= capacity + 1e-9 for r in plan)
    if sigma == 0.0 and fleet == 6:
        assert all(rl["res"]["short"] == 0.0 for rl in live["rules"].values())


def test_rule_settings_and_threshold(frozen_live):
    assert LV.rule_settings(4, 0.25, 3) == {"R": (0.0, 0.5, 0.0), "P": (4.0, 0.25, 0.0), "E": (0.0, 0.5, 3.0)}
    live, case = frozen_live
    assert np.allclose(LV.threshold(live), case["inst"].mu * 1.3) and np.allclose(LV.threshold(live, 2.0), case["inst"].mu * 3.3) and LV.threshold(live)[0] == 0


def test_stop_kind_boundary_and_the_early_rule(frozen_live):
    live, _ = frozen_live
    i = 3
    thr = LV.threshold(live)[i]
    assert LV.stop_kind(live, "P", thr - 1e-6, i) == "fällig" and LV.stop_kind(live, "P", thr, i) == "Mitnahme" and LV.stop_kind(live, "R", thr + 1.0, i) == "Mitnahme"
    thr_e = LV.threshold(live, live["early"])[i]
    assert LV.stop_kind(live, "E", thr_e - 1e-6, i) == "fällig" and LV.stop_kind(live, "E", thr + 1.0, i) == "fällig" and LV.stop_kind(live, "E", thr_e, i) == "Mitnahme"


def test_day_view_lists_tours_with_quantities_reach_and_kind(frozen_live):
    live, case = frozen_live
    inst = case["inst"]
    for rule in ("R", "P", "E"):
        rl = live["rules"][rule]
        for t in range(0, 120, 7):
            v = LV.day_view(live, rule, t + 1)
            assert v["day"] == t + 1 and v["rule"] == rule and len(v["tours"]) == len(rl["plans"][t]) and np.array_equal(v["I"], rl["I"][t])
            assert v["km"] == rl["res"]["day_route_km"][t] and v["km"] == pytest.approx(sum(x["length"] for x in v["tours"]))
            for tour, route in zip(v["tours"], rl["plans"][t]):
                assert [s["customer"] for s in tour["stops"]] == [c for c, _ in route] and tour["load"] == pytest.approx(sum(s["qty"] for s in tour["stops"])) and tour["load"] <= live["Q"] + 1e-9
                for s in tour["stops"]:
                    c = s["customer"]
                    assert s["fill_before"] == rl["I"][t][c] and s["fill_share"] == pytest.approx(rl["I"][t][c] / inst.C[c]) and s["reach"] == pytest.approx(rl["I"][t][c] / inst.mu[c])
                    assert s["qty"] == pytest.approx(min(dict(route)[c], inst.C[c] - rl["I"][t][c])) and s["kind"] in ("fällig", "Mitnahme")
            assert v["n_stops"] == sum(len(x["stops"]) for x in v["tours"])


def test_pickups_only_exist_for_bundling_and_add_up_to_the_measured_count(frozen_live):
    live, _ = frozen_live
    totals = {r: sum(LV.day_view(live, r, d)["n_pickups"] for d in range(1, 121)) for r in ("R", "P", "E")}
    assert totals["R"] == 0 and totals["E"] == 0 and totals["P"] == live["rules"]["P"]["res"]["opt_visits"] > 0
    for d in range(1, 121):
        v = LV.day_view(live, "P", d)
        assert v["n_pickups"] == sum(1 for x in v["tours"] for s in x["stops"] if s["kind"] == "Mitnahme") <= v["n_stops"]


def test_tour_length_is_the_euclidean_round_trip(frozen_live):
    live, case = frozen_live
    assert LV.tour_length(live, []) == 0.0 and LV.tour_length(live, [5]) == pytest.approx(2 * case["inst"].dist[0, 5])
    for tour in [[3, 7, 11], [1, 2]]:
        assert LV.tour_length(live, tour) == pytest.approx(RT.route_len(case["inst"].dist, tour))


def test_pickup_days_and_the_day_table(frozen_live):
    live, _ = frozen_live
    days = LV.pickup_days(live, 8)
    assert len(days) == 8 and days == sorted(days) and all(LV.day_view(live, "P", d)["n_pickups"] > 0 for d in days)
    assert LV.pickup_days(live, 3) == days[:3] and len(LV.pickup_days(live, 1000)) == sum(1 for d in range(1, 121) if LV.day_view(live, "P", d)["n_pickups"] > 0)
    rows = LV.day_table(live)
    assert len(rows) == 120 and [r["Tag"] for r in rows] == list(range(1, 121)) and set(rows[0]) == {"Tag", "Touren Reaktiv", "km Reaktiv", "Touren Bündeln", "km Bündeln", "Touren Früher liefern", "km Früher liefern", "Mitnehmer"}
    assert sum(r["Mitnehmer"] for r in rows) == live["rules"]["P"]["res"]["opt_visits"] and sum(r["Touren Reaktiv"] for r in rows) == live["rules"]["R"]["res"]["n_routes"]
    assert sum(r["km Bündeln"] for r in rows) == pytest.approx(live["rules"]["P"]["res"]["routing"]) and sum(r["Touren Früher liefern"] for r in rows) == live["rules"]["E"]["res"]["n_routes"]


def test_no_pickups_when_bundling_is_switched_off_by_a_zero_lookahead(monkeypatch):
    monkeypatch.setattr(LV, "make_live_instance", lambda *a, **k: FZ.frozen("base", 2))
    live = LV.solve_live(20, 300, 6, 0.3, 5, 0, 0.5, 2, 2)
    assert LV.pickup_days(live) == [] and live["rules"]["P"]["res"]["opt_visits"] == 0


def test_mean_fill_is_the_share_of_the_tank_at_the_start_of_the_day(frozen_live):
    live, case = frozen_live
    for rule in ("R", "P", "E"):
        want = float((live["rules"][rule]["I"][:120, 1:] / case["inst"].C[1:]).mean())
        assert LV.mean_fill(live, rule) == want and 0.2 < want < 0.9
    assert LV.mean_fill(live, "E") > LV.mean_fill(live, "R")                                          # früher liefern hält die Tanks voller
    assert LV.mean_fill(live, "P") != LV.mean_fill(live, "R")
