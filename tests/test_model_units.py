"""Einzelne Bausteine des Kerns an Handinstanzen mit von Hand gerechneten Ergebnissen: Instanz (nur Invarianten, nie exakte Zufallswerte), Tourenbausteine, die
Entscheidung der drei Regeln, ein Tag, die Simulation über mehrere Tage, Nachspielen, Zielgröße."""
import math

import numpy as np
import pytest

import irp_model as M
import irp_policy as P
import irp_routing as RT


def mini(xy, mu, C, I0, cons, Q=100.0, F=2, sigma=0.0, kappa=1.0, pfac=0.05):
    """Handinstanz aus Koordinaten (Zeile 0 = Depot), Raten, Tanks, Anfangsbestand und Verbrauch je Tag."""
    inst = M.Inst()
    xy = np.array(xy, dtype=float)
    inst.N, inst.size, inst.sigma, inst.Q, inst.F = len(xy) - 1, 100.0, sigma, float(Q), F
    inst.p, inst.kappa = pfac * 100.0, kappa
    inst.xy = xy
    inst.dist = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    inst.mu, inst.C, inst.I0 = (np.array(v, dtype=float) for v in (mu, C, I0))
    inst.cons = np.array(cons, dtype=float)
    inst.D = len(inst.cons)
    return inst


LINE = dict(xy=[(0, 0), (10, 0), (20, 0), (30, 0), (0, 50)], mu=[0, 10, 10, 10, 10], C=[0, 100, 100, 100, 100], I0=[0, 50, 50, 50, 50], cons=[[0, 10, 10, 10, 10]])
D14 = math.sqrt(30 ** 2 + 50 ** 2)                       # Kunde 3 -> Kunde 4


def line(Q=300.0, F=2, sigma=0.0, I0=None):
    d = dict(LINE)
    if I0 is not None:
        d["I0"] = I0
    return mini(Q=Q, F=F, sigma=sigma, **d)


# ---------------------------------------------------------------------------------------------------
# make_instance: nur Invarianten
# ---------------------------------------------------------------------------------------------------
def test_make_instance_shapes_and_ranges():
    inst = M.make_instance(5, N=12, D=30)
    assert inst.N == 12 and inst.D == 30 and inst.Q == 300.0 and inst.F == 6 and inst.sigma == 0.3 and inst.kappa == 1.0 and inst.size == 100.0
    assert inst.xy.shape == (13, 2) and inst.dist.shape == (13, 13) and inst.mu.shape == inst.C.shape == inst.I0.shape == (13,) and inst.cons.shape == (30, 13)
    assert np.allclose(inst.dist, inst.dist.T) and np.all(np.diag(inst.dist) == 0) and inst.p == pytest.approx(5.0)
    assert inst.mu[0] == inst.C[0] == inst.I0[0] == 0 and np.all(inst.cons[:, 0] == 0)
    mu, C, I0 = inst.mu[1:], inst.C[1:], inst.I0[1:]
    assert np.all((mu >= 6) & (mu <= 14)) and np.all((C / mu >= 8 - 1e-9) & (C / mu <= 14 + 1e-9)) and np.all((I0 / C >= 0.25 - 1e-9) & (I0 / C <= 1 + 1e-9))
    assert np.all(inst.xy[1:] >= 0) and np.all(inst.xy[1:] <= 100) and np.all(inst.cons >= 0)
    assert np.hypot(*(inst.xy[1] - inst.xy[2])) == pytest.approx(inst.dist[1, 2])


def test_make_instance_options_change_the_instance_as_documented():
    assert tuple(M.make_instance(1, N=3, D=2, depot="center").xy[0]) == (50.0, 50.0)
    assert tuple(M.make_instance(1, N=3, D=2, depot="corner").xy[0]) == (0.0, 0.0)
    assert tuple(M.make_instance(1, N=3, D=2, depot="far").xy[0]) == (50.0, -50.0)
    assert M.make_instance(1, N=3, D=2, size=200.0).xy[0][0] == 100.0 and M.make_instance(1, N=3, D=2, size=200.0).p == pytest.approx(10.0)
    assert M.make_instance(1, N=3, D=2, pfac=0.25).p == pytest.approx(25.0)
    c = M.make_instance(2, N=30, D=3, clustered=True)
    assert np.all(c.xy[1:] >= 0) and np.all(c.xy[1:] <= 100)
    r = M.make_instance(2, N=6, D=3, round_dist=0.1)
    assert np.allclose(r.dist / 0.1, np.round(r.dist / 0.1))
    t = M.make_instance(3, N=25, D=3, tank_lo=4.0, tank_hi=7.0, mu_lo=20.0, mu_hi=30.0)
    assert np.all((t.mu[1:] >= 20) & (t.mu[1:] <= 30)) and np.all((t.C[1:] / t.mu[1:] >= 4 - 1e-9) & (t.C[1:] / t.mu[1:] <= 7 + 1e-9))
    assert M.make_instance(1, N=3, D=2, kappa=2.0).kappa == 2.0 and M.make_instance(1, N=3, D=2, F=4, Q=77.0).F == 4


def test_make_instance_consumption_stream_deterministic_and_seed_dependent():
    a, b, c = M.make_instance(9, D=20), M.make_instance(9, D=20), M.make_instance(10, D=20)
    assert np.array_equal(a.cons, b.cons) and np.array_equal(a.xy, b.xy) and np.array_equal(a.C, b.C)
    assert not np.array_equal(a.cons, c.cons) and not np.array_equal(a.xy, c.xy)


def test_consumption_is_constant_at_sigma_zero_and_gamma_with_the_right_mean_otherwise():
    d = M.make_instance(4, N=5, D=6, sigma=0.0)
    assert np.array_equal(d.cons[:, 1:], np.tile(d.mu[1:], (6, 1)))
    s = M.make_instance(4, N=3, D=4000, sigma=0.3)
    assert np.all(s.cons[:, 1:].mean(axis=0) == pytest.approx(s.mu[1:], rel=0.03))
    assert np.all(s.cons[:, 1:].std(axis=0) / s.cons[:, 1:].mean(axis=0) == pytest.approx(0.3, abs=0.05))     # Variationskoeffizient sigma
    hi = M.make_instance(4, N=3, D=4000, sigma=1.0)
    assert np.all(hi.cons[:, 1:].std(axis=0) / hi.cons[:, 1:].mean(axis=0) == pytest.approx(1.0, abs=0.15))


# ---------------------------------------------------------------------------------------------------
# Tourenbausteine
# ---------------------------------------------------------------------------------------------------
SQ = mini(xy=[(0, 0), (0, 10), (10, 10), (10, 0)], mu=[0, 1, 1, 1], C=[0, 9, 9, 9], I0=[0, 5, 5, 5], cons=[[0, 1, 1, 1]])


def test_route_len_on_the_hand_instance():
    h = mini(xy=[(0, 0), (3, 0), (3, 4), (0, 4)], mu=[0, 1, 1, 1], C=[0, 9, 9, 9], I0=[0, 5, 5, 5], cons=[[0, 1, 1, 1]])
    assert RT.route_len(h.dist, [1, 2, 3]) == pytest.approx(14.0) and RT.route_len(h.dist, [2]) == pytest.approx(10.0) and RT.route_len(h.dist, []) == 0.0
    assert RT.route_len(h.dist, [1, 3]) == pytest.approx(3 + 5 + 4) and RT.route_len(h.dist, [3, 1]) == pytest.approx(RT.route_len(h.dist, [1, 3]))
    assert isinstance(RT.route_len(h.dist, [1]), float)


def test_two_opt_removes_a_crossing_and_leaves_short_routes_alone():
    bad = [2, 1, 3]
    good = RT.two_opt(SQ.dist, bad)
    assert RT.route_len(SQ.dist, good) == pytest.approx(40.0) and sorted(good) == [1, 2, 3]
    assert RT.route_len(SQ.dist, bad) > 40.0 + 1.0
    assert RT.two_opt(SQ.dist, good) == good                                         # ein lokales Optimum bleibt unverändert
    assert RT.two_opt(SQ.dist, [2, 1]) == [2, 1] and RT.two_opt(SQ.dist, []) == [] and RT.two_opt(SQ.dist, [3]) == [3]
    src = [2, 1, 3]
    RT.two_opt(SQ.dist, src)
    assert src == [2, 1, 3]                                                          # die Eingabe wird nicht verändert


def test_two_opt_reverses_a_segment_at_the_start_and_at_the_end_of_the_route():
    for bad in ([2, 1, 3], [1, 3, 2]):                                                # der erste bzw. der letzte Abschnitt muss gedreht werden
        good = RT.two_opt(SQ.dist, bad)
        assert RT.route_len(SQ.dist, bad) == pytest.approx(10 + 10 * math.sqrt(2) + 10 + 10 * math.sqrt(2))
        assert RT.route_len(SQ.dist, good) == pytest.approx(40.0)
    best = min(RT.route_len(SQ.dist, list(p)) for p in __import__("itertools").permutations([1, 2, 3]))
    assert best == pytest.approx(40.0)


def test_savings_merges_by_largest_saving_and_respects_capacity():
    inst = mini(xy=[(0, 0), (10, 0), (11, 0), (-10, 0)], mu=[0, 1, 1, 1], C=[0] + [9] * 3, I0=[0] + [5] * 3, cons=[[0, 1, 1, 1]])
    qty = {1: 10.0, 2: 10.0, 3: 10.0}
    r = RT.savings_routes(inst.dist, [1, 2, 3], qty, 100.0)
    assert sorted(map(sorted, r)) == [[1, 2], [3]]                                    # 1 und 2 sparen 20, 3 spart 0: eigene Tour
    assert sum(RT.route_len(inst.dist, x) for x in r) == pytest.approx(22 + 20)
    r2 = RT.savings_routes(inst.dist, [1, 2, 3], qty, 15.0)                           # jede Tour höchstens 15: keine Zusammenlegung
    assert sorted(map(sorted, r2)) == [[1], [2], [3]]
    r3 = RT.savings_routes(inst.dist, [1, 2, 3], qty, 20.0)                           # genau 20 passt (Kapazität inklusive)
    assert sorted(map(sorted, r3)) == [[1, 2], [3]]
    assert RT.savings_routes(inst.dist, [], qty, 100.0) == [] and RT.savings_routes(inst.dist, [2], qty, 100.0) == [[2]]


def test_savings_partitions_all_customers_and_is_deterministic():
    custs = list(range(1, 16))
    inst = mini(xy=[(0, 0)] + [((37 * c + 11) % 100, (53 * c + 7) % 100) for c in custs], mu=[0] + [1] * 15, C=[0] + [9] * 15, I0=[0] + [5] * 15, cons=[[0] + [1] * 15])
    qty = {c: 40.0 + 3 * c for c in custs}
    a = RT.savings_routes(inst.dist, custs, qty, 200.0)
    b = RT.savings_routes(inst.dist, custs, qty, 200.0)
    assert a == b and sorted(c for r in a for c in r) == custs and all(sum(qty[c] for c in r) <= 200.0 + 1e-9 for r in a)
    assert len(a) < 15 and len(RT.savings_routes(inst.dist, custs, qty, 1e9)) == 1


def test_cheapest_insertion_position_and_extra_distance():
    h = mini(xy=[(0, 0), (3, 0), (3, 4), (0, 4)], mu=[0, 1, 1, 1], C=[0, 9, 9, 9], I0=[0, 5, 5, 5], cons=[[0, 1, 1, 1]])
    delta, pos = RT.cheapest_insertion(h.dist, [1, 3], 2)
    assert (round(delta, 9), pos) == (2.0, 1)                                         # 1 -> 2 -> 3: 4 + 3 - 5
    delta, pos = RT.cheapest_insertion(h.dist, [], 2)
    assert (round(delta, 9), pos) == (10.0, 0)                                        # leere Tour: Hin- und Rückfahrt
    delta, pos = RT.cheapest_insertion(h.dist, [2], 2)
    assert delta == pytest.approx(0.0) and pos == 0                                   # gleicher Punkt: erste beste Stelle gewinnt
    assert RT.cheapest_insertion(h.dist, [1, 3], 2)[1] == 1


# ---------------------------------------------------------------------------------------------------
# Entscheidung: R, P, E
# ---------------------------------------------------------------------------------------------------
def routes_of(plan):
    return [[c for c, _ in r] for r in plan]


def test_reactive_takes_only_due_customers_sorted_and_merged():
    inst = line(Q=300.0)
    I = np.array([0, 5, 25, 15, 8.0])
    plan, st = P.decide(inst, I)
    assert routes_of(plan) == [[1, 4]] and plan[0] == [(1, 95.0), (4, 92.0)] and st == {"deferred": 0, "opt_visits": 0}


def test_due_is_strictly_below_the_report_threshold():
    inst = line(Q=300.0, sigma=0.5)                                                   # Meldegrenze mu*(1+1*0,5) = 15
    plan, _ = P.decide(inst, np.array([0, 14.9, 50, 15.0, 50.0]))
    assert routes_of(plan) == [[1]]                                                   # Bestand genau 15: nicht fällig
    assert routes_of(P.decide(line(Q=300.0), np.array([0, 9.99, 50, 10.0, 50.0]))[0]) == [[1]]
    inst_k = line(Q=300.0, sigma=0.5)
    inst_k.kappa = 0.0                                                                # Meldegrenze mu
    assert routes_of(P.decide(inst_k, np.array([0, 9.9, 50, 12.0, 50.0]))[0]) == [[1]]


def test_quantity_is_the_fill_to_the_tank_capped_by_the_wagon():
    inst = line(Q=40.0, F=6)
    plan, _ = P.decide(inst, np.array([0, 5, 50, 50, 8.0]))
    assert plan and all(q <= 40.0 for r in plan for _, q in r) and sorted(q for r in plan for _, q in r) == [40.0, 40.0]
    plan, _ = P.decide(line(Q=300.0), np.array([0, 5, 50, 50, 8.0]))
    assert {c: q for r in plan for c, q in r} == {1: 95.0, 4: 92.0}


def test_bundling_adds_a_pickup_only_below_the_gamma_threshold_and_with_room():
    inst = line(Q=300.0)
    I = np.array([0, 5, 25, 15, 8.0])
    plan, st = P.decide(inst, I, H=1.0, gamma=0.5)
    assert routes_of(plan) == [[1, 3, 4]] and st["opt_visits"] == 1 and plan[0][1] == (3, 85.0)           # Einfügung 27,3 <= 0,5 * 2 * 30 = 30
    plan, st = P.decide(inst, I, H=1.0, gamma=0.25)
    assert routes_of(plan) == [[1, 4]] and st["opt_visits"] == 0                                           # 27,3 > 15: zu teuer
    plan, st = P.decide(line(Q=250.0), I, H=1.0, gamma=1.0)
    assert routes_of(plan) == [[1, 4]] and st["opt_visits"] == 0                                           # 187 + 85 > 250: kein Platz im Wagen
    plan, st = P.decide(inst, I, H=0.5, gamma=1.0)
    assert routes_of(plan) == [[1, 4]] and st["opt_visits"] == 0                                           # Restreichweite 1,5 Tage > 1 + 0,5: nicht in der Vorschau
    plan, st = P.decide(inst, I, H=0.6, gamma=1.0)
    assert st["opt_visits"] == 1                                                                           # Grenze: I < mu * (1 + H) = 16 -> Kunde 3 (15) dabei
    plan, _ = P.decide(inst, I, H=0.0, gamma=1.0)
    assert routes_of(plan) == [[1, 4]]                                                                     # H = 0: reaktiv, gamma egal


def test_pickup_gamma_boundary_is_inclusive():
    inst = line(Q=300.0)
    I = np.array([0, 5, 25, 15, 8.0])
    delta = 20 + D14 - 50.99019513592785 + 0                                                              # 27,32
    gamma_exact = (RT.cheapest_insertion(inst.dist, [1, 4], 3)[0]) / (2.0 * inst.dist[0, 3])
    assert P.decide(inst, I, H=1.0, gamma=gamma_exact)[1]["opt_visits"] == 1                               # genau am Rand: Mitnahme erlaubt
    assert P.decide(inst, I, H=1.0, gamma=gamma_exact - 1e-6)[1]["opt_visits"] == 0
    assert delta == pytest.approx(RT.cheapest_insertion(inst.dist, [1, 4], 3)[0], rel=1e-6)


def test_no_pickups_without_a_due_tour():
    plan, st = P.decide(line(), np.array([0, 15, 15, 15, 15.0]), H=5.0, gamma=1.0)
    assert plan == [] and st == {"deferred": 0, "opt_visits": 0}


def test_pickups_are_taken_by_urgency_and_a_full_wagon_skips_them():
    inst = line(Q=300.0, F=2)
    I = np.array([0, 5, 12, 15, 50.0])                                                                     # fällig: 1; Kandidaten: 2 (1,2 Tage), 3 (1,5 Tage)
    plan, st = P.decide(inst, I, H=1.0, gamma=10.0)
    assert st["opt_visits"] == 2 and sorted(routes_of(plan)[0]) == [1, 2, 3]
    plan, st = P.decide(line(Q=190.0, F=2), I, H=1.0, gamma=10.0)                                          # 95 + 88 = 183 <= 190, +85 = 268 > 190
    assert st["opt_visits"] == 1 and sorted(routes_of(plan)[0]) == [1, 2]


def test_a_tight_fleet_defers_the_least_urgent_due_customer():
    inst = line(Q=100.0, F=1)
    plan, st = P.decide(inst, np.array([0, 5, 25, 15, 8.0]))
    assert routes_of(plan) == [[1]] and st["deferred"] == 1                                                # 4 (0,8 Tage) ist weniger dringend als 1 (0,5 Tage)
    plan, st = P.decide(line(Q=100.0, F=2), np.array([0, 5, 25, 15, 8.0]))
    assert sorted(map(sorted, routes_of(plan))) == [[1], [4]] and st["deferred"] == 0
    plan, st = P.decide(line(Q=100.0, F=1), np.array([0, 8, 25, 15, 5.0]))
    assert routes_of(plan) == [[4]] and st["deferred"] == 1                                                # jetzt ist 4 dringender


def test_deferring_stops_when_nothing_is_left():
    inst = line(Q=10.0, F=0)
    plan, st = P.decide(inst, np.array([0, 5, 25, 15, 8.0]))
    assert plan == [] and st["deferred"] == 2


def test_early_delivery_raises_the_threshold_without_pickups():
    inst = line(Q=300.0, F=6)
    I = np.array([0, 5, 25, 15, 8.0])
    plan, st = P.decide(inst, I, early=2.0)                                                                # Meldegrenze 30: alle vier
    assert sorted(c for r in plan for c, _ in r) == [1, 2, 3, 4] and st["opt_visits"] == 0
    plan, st = P.decide(inst, I, early=1.0)                                                                # Meldegrenze 20: 1, 3, 4
    assert sorted(c for r in plan for c, _ in r) == [1, 3, 4]
    assert sorted(c for r in P.decide(inst, I, early=0.0)[0] for c, _ in r) == [1, 4]


def test_the_plan_never_visits_a_customer_twice():
    for s in range(6):
        inst = M.make_instance(s, D=1)
        plan, _ = P.decide(inst, inst.I0 * 0.3, H=6.0, gamma=1.0)
        seen = [c for r in plan for c, _ in r]
        assert len(seen) == len(set(seen)) and all(sum(q for _, q in r) <= inst.Q + 1e-9 for r in plan) and len(plan) <= inst.F


def test_fleet_argument_overrides_the_instance_fleet():
    inst = line(Q=100.0, F=6)
    assert P.decide(inst, np.array([0, 5, 25, 15, 8.0]), fleet=1)[1]["deferred"] == 1 and P.decide(inst, np.array([0, 5, 25, 15, 8.0]))[1]["deferred"] == 0


# ---------------------------------------------------------------------------------------------------
# Ein Tag und die Simulation
# ---------------------------------------------------------------------------------------------------
def test_apply_day_delivers_consumes_and_counts():
    inst = line(Q=300.0)
    I2, km, short, delivered, served, n_short = P.apply_day(inst, np.array([0, 5, 25, 15, 8.0]), [[(1, 95.0), (4, 92.0)]], np.array([0, 10, 10, 10, 10.0]))
    assert list(I2) == [0.0, 90.0, 15.0, 5.0, 90.0] and km == pytest.approx(10 + 50.99019513592785 + 50) and short == 0 and delivered == 187.0
    assert served == 40.0 and n_short == 0


def test_apply_day_shortage_is_lost_and_counted_and_the_tank_is_capped():
    inst = line(Q=300.0)
    I2, km, short, delivered, served, n_short = P.apply_day(inst, np.array([0, 3, 25, 15, 8.0]), [], np.array([0, 10, 10, 10, 10.0]))
    assert list(I2) == [0.0, 0.0, 15.0, 5.0, 0.0] and short == pytest.approx(7 + 2) and served == pytest.approx(3 + 10 + 10 + 8) and n_short == 2 and km == 0.0 and delivered == 0.0
    I3, _, _, delivered, _, _ = P.apply_day(inst, np.array([0, 90.0, 50, 50, 50]), [[(1, 500.0)]], np.array([0, 0, 0, 0, 0.0]))
    assert I3[1] == 100.0 and delivered == 10.0                                                            # 500 angefordert, nur 10 passen in den Tank
    I0 = np.array([0, 5, 25, 15, 8.0])
    P.apply_day(inst, I0, [[(1, 95.0)]], np.array([0, 10, 10, 10, 10.0]))
    assert list(I0) == [0, 5, 25, 15, 8.0]                                                                 # die Eingabe bleibt unverändert


def test_apply_day_a_tiny_shortage_below_tolerance_is_not_a_stockout_day():
    inst = line()
    _, _, short, _, _, n_short = P.apply_day(inst, np.array([0, 10.0, 10, 10, 10]), [], np.array([0, 10 + 1e-10, 10, 10, 10.0]))
    assert n_short == 0 and short == pytest.approx(1e-10)


TWO = dict(xy=[(0, 0), (10, 0), (0, 20)], mu=[0, 10, 5], C=[0, 50, 40], I0=[0, 12, 6], cons=[[0, 10, 5]] * 3)


def test_simulate_reactive_by_hand():
    inst = mini(Q=100.0, F=2, **TWO)
    r = P.simulate(inst, record=True)
    km = 10 + math.sqrt(10 ** 2 + 20 ** 2) + 20                                                            # Depot - 1 - 2 - Depot
    assert r["routing"] == pytest.approx(km) and r["short"] == 0.0 and r["delivered"] == 87.0 and r["served"] == 45.0
    assert (r["n_routes"], r["n_visits"], r["opt_visits"], r["deferred"], r["stockout_days"], r["route_days"]) == (1, 2, 0, 0, 0, 1)
    assert r["util_sum"] == pytest.approx(0.87) and r["day_route_km"] == [0.0, pytest.approx(km), 0.0]
    assert (r["I_start"], r["I_end"], r["dI"]) == (18.0, 60.0, 42.0) and r["plans"][0] == [] and r["plans"][2] == [] and routes_of(r["plans"][1]) == [[1, 2]]


def test_simulate_bundling_does_nothing_without_a_due_tour():
    inst = mini(Q=100.0, F=2, **TWO)
    a, b = P.simulate(inst, 0.0), P.simulate(inst, 3.0, 0.5)
    assert a["routing"] == b["routing"] and b["opt_visits"] == 0 and a["dI"] == b["dI"]


def test_simulate_early_delivery_by_hand():
    inst = mini(Q=100.0, F=2, **TWO)
    r = P.simulate(inst, early=1.0)
    km = 10 + math.sqrt(10 ** 2 + 20 ** 2) + 20
    assert r["routing"] == pytest.approx(km) and r["delivered"] == 72.0 and (r["I_end"], r["dI"]) == (45.0, 27.0) and r["n_routes"] == 1 and r["n_visits"] == 2


def test_simulate_with_shortage_and_a_custom_plan_function():
    inst = mini(Q=100.0, F=2, xy=[(0, 0), (10, 0)], mu=[0, 10], C=[0, 50], I0=[0, 4], cons=[[0, 10], [0, 10]])
    calls = []

    def never(ins, I, t):
        calls.append(t)
        return [], {"deferred": 2, "opt_visits": 3}

    r = P.simulate(inst, plan_fn=never)
    assert calls == [0, 1] and r["short"] == 16.0 and r["served"] == 4.0 and r["stockout_days"] == 2 and r["n_routes"] == 0 and r["deferred"] == 4 and r["opt_visits"] == 6
    assert r["route_days"] == 0 and r["I_end"] == 0.0 and r["dI"] == -4.0 and "plans" not in r and r["day_route_km"] == [0.0, 0.0]
    assert P.objective(r, inst.p, 0.5) == pytest.approx(0 + 5.0 * 16 + 0.5 * 4)


def test_replay_reproduces_a_recorded_run_and_objective_is_the_documented_sum():
    inst = M.make_instance(2, N=10, D=40)
    r = P.simulate(inst, 3.0, 0.5, record=True)
    rp = P.replay(inst, r["plans"])
    for k in ("routing", "short", "delivered", "served", "n_routes", "n_visits", "I_end", "stockout_days", "util_sum"):
        assert rp[k] == r[k], k
    assert rp["opt_visits"] == 0 and rp["plans"] == r["plans"]                                             # beim Nachspielen gibt es keine eigene Entscheidung
    assert P.objective(r, 5.0, 0.6) == pytest.approx(r["routing"] + 5.0 * r["short"] - 0.6 * (r["I_end"] - r["I_start"]))


def test_simulation_does_not_change_the_instance():
    inst = M.make_instance(2, N=10, D=20)
    c, i0, xy = inst.cons.copy(), inst.I0.copy(), inst.xy.copy()
    P.simulate(inst, 4.0, 1.0, record=True)
    P.simulate(inst, 0.0, early=2.0)
    assert np.array_equal(c, inst.cons) and np.array_equal(i0, inst.I0) and np.array_equal(xy, inst.xy)


def test_simulate_counters_add_up_over_the_recorded_plans():
    inst = M.make_instance(6, N=15, D=50)
    r = P.simulate(inst, 3.0, 0.5, record=True)
    assert r["n_routes"] == sum(len(p) for p in r["plans"]) and r["n_visits"] == sum(len(x) for p in r["plans"] for x in p)
    assert r["route_days"] == sum(1 for p in r["plans"] if p) and len(r["plans"]) == len(r["day_route_km"]) == 50
    assert r["routing"] == pytest.approx(sum(r["day_route_km"])) and r["util_sum"] == pytest.approx(sum(q for p in r["plans"] for x in p for _, q in x) / inst.Q, rel=1e-6)
