"""Die Korrektheits-Checks der Messreihe (check.py und check_oracle.py) als Funktionen mit einstellbarer Größe: verkleinert in tests/test_checks.py (CI), in
voller Größe in tools/check_full.py (das Bau-Gate, einmal lokal vor dem Commit). Jede Funktion wirft AssertionError bei einem Fehler und liefert die Kennzahlen zurück.

Die Checks mit Zufallsinstanzen (make_instance) prüfen nur Invarianten, die für JEDE Instanz gelten; Aussagen über bestimmte Zahlen laufen über die
eingefrorenen Instanzen (irp_frozen) oder stehen nur im vollen Lauf (dort mit der lokalen NumPy-Version, mit der die Messreihe gerechnet wurde)."""
import itertools
import math

import numpy as np

import irp_model as M
import irp_policy as P
import irp_routing as RT


def indep_eval(inst, plans):
    """Unabhängige Bewertung eines Plans (reine Python-Schleifen, eigene Distanzberechnung aus den Koordinaten)."""
    N = inst.N
    inv = [float(inst.I0[i]) for i in range(N + 1)]
    routing = short = delivered = served_tot = 0.0
    for t, plan in enumerate(plans):
        seen = set()
        for route in plan:
            assert route, "leere Tour"
            load = 0.0
            pts = [0] + [c for c, _ in route] + [0]
            for a, b in zip(pts[:-1], pts[1:]):
                routing += math.hypot(inst.xy[a][0] - inst.xy[b][0], inst.xy[a][1] - inst.xy[b][1])
            for c, q in route:
                assert c not in seen, "Kunde zweimal am Tag"
                seen.add(c)
                assert 1 <= c <= N
                assert q >= -1e-12
                q = min(q, inst.C[c] - inv[c])                     # Tank kann nicht überlaufen
                load += q
                inv[c] += q
                delivered += q
            assert load <= inst.Q + 1e-9, ("Kapazität verletzt", load)
        assert len(plan) <= inst.F, "zu viele Fahrzeuge"
        for c in range(1, N + 1):
            use = min(inv[c], inst.cons[t][c])
            short += inst.cons[t][c] - use
            served_tot += use
            inv[c] -= use
            assert -1e-9 <= inv[c] <= inst.C[c] + 1e-9
    return routing, short, delivered, served_tot, sum(inv[1:])


def check_no_consumption(n=12):
    """1. Grenzfall ohne Verbrauch: keine Touren, keine Kosten, kein Bestandsverlauf (reaktiv und proaktiv)."""
    inst = M.make_instance(3, N=n)
    inst.mu = np.zeros_like(inst.mu)
    inst.cons = np.zeros_like(inst.cons)
    for H, g in [(0, 0.5), (3, 0.5), (8, 1.0)]:
        r = P.simulate(inst, H, g)
        assert r["routing"] == 0 and r["n_routes"] == 0 and r["short"] == 0 and r["dI"] == 0, r
    return True


def check_h0_equals_reactive(n_seeds=60, days=40):
    """2. H = 0 ist reaktiv, unabhängig von gamma: bitgleiche Tourpläne für gamma 0,1 / 0,5 / 1 / 5 in n_seeds x 4 Konfigurationen. Rückgabe die Zahl der Instanzen."""
    n = 0
    for s in range(n_seeds):
        for kw in [dict(), dict(sigma=0.0), dict(sigma=0.6, Q=150.0, F=2), dict(N=8, clustered=True)]:
            inst = M.make_instance(s, D=days, **kw)
            base = P.simulate(inst, 0.0, 0.5, record=True)
            for g in (0.1, 1.0, 5.0):
                other = P.simulate(inst, 0.0, g, record=True)
                assert other["plans"] == base["plans"] and other["routing"] == base["routing"]
            n += 1
    return n


BALANCE_CONFIGS = [dict(), dict(sigma=0.6), dict(Q=150.0, F=1), dict(N=30, Q=500.0, F=2, sigma=1.0), dict(depot="far", clustered=True)]
BALANCE_POLICIES = [(0, 0.5, 0.0), (2, 0.5, 0.0), (5, 1.0, 0.0), (0, 0.5, 2.0)]


def check_balance(n_seeds=40, days=60, configs=None, policies=None):
    """3. Unabhängige Neubewertung == Simulation und exakte Bestandsbilanz (Endbestand = Anfang + geliefert - bedient; bedient + Fehlmenge = Verbrauch). Rückgabe Zahl der Läufe."""
    n = 0
    for s in range(n_seeds):
        for kw in configs or BALANCE_CONFIGS:
            inst = M.make_instance(s, D=days, **kw)
            for H, g, e in policies or BALANCE_POLICIES:
                r = P.simulate(inst, H, g, early=e, record=True)
                rt, sh, dl, sv, iend = indep_eval(inst, r["plans"])
                assert abs(rt - r["routing"]) < 1e-6 * max(1, rt), (rt, r["routing"])
                assert abs(sh - r["short"]) < 1e-6 and abs(dl - r["delivered"]) < 1e-6 and abs(sv - r["served"]) < 1e-6
                assert abs(iend - r["I_end"]) < 1e-6
                assert abs(r["I_end"] - (r["I_start"] + r["delivered"] - r["served"])) < 1e-6
                assert abs(r["served"] + r["short"] - inst.cons.sum()) < 1e-6
                n += 1
    return n


def hand_instance():
    """Depot (0,0), Kunden (3,0), (3,4), (0,4): eine Tour 3+4+3+4 = 14, je Kunde eine Tour 6+10+8 = 24, zwei Kunden je Tour 18."""
    inst = M.make_instance(1, N=3, D=1)
    inst.xy = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 4.0], [0.0, 4.0]])
    inst.dist = np.sqrt(((inst.xy[:, None, :] - inst.xy[None, :, :]) ** 2).sum(-1))
    return inst


def check_hand_instance():
    """4. Tourkosten gegen die Handinstanz. Rückgabe die Gesamtlänge bei Kapazität für zwei Kunden (18)."""
    inst = hand_instance()
    qty = {1: 10.0, 2: 10.0, 3: 10.0}
    routes = RT.savings_routes(inst.dist, [1, 2, 3], qty, 100.0)
    assert len(routes) == 1 and abs(RT.route_len(inst.dist, routes[0]) - 14.0) < 1e-9, routes      # 3+4+3+4
    routes = RT.savings_routes(inst.dist, [1, 2, 3], qty, 10.0)                                     # jede Tour ein Kunde
    assert len(routes) == 3 and abs(sum(RT.route_len(inst.dist, r) for r in routes) - (6 + 10 + 8)) < 1e-9
    routes = RT.savings_routes(inst.dist, [1, 2, 3], qty, 20.0)
    tot = sum(RT.route_len(inst.dist, r) for r in routes)
    assert len(routes) == 2 and abs(tot - 18.0) < 1e-9, (routes, tot)      # {2,3}: 5+3+4 = 12 plus {1}: 6 (Optimum, per Hand)
    return tot


def brute_cvrp(dist, custs, qty, Q):
    """Exaktes CVRP durch Enumeration aller Zerlegungen in Touren und aller Reihenfolgen (n <= 6)."""
    n = len(custs)

    def tsp(sub):
        if len(sub) == 1:
            return 2 * dist[0, sub[0]]
        b = float("inf")
        for perm in itertools.permutations(sub):
            b = min(b, RT.route_len(dist, list(perm)))
        return b

    memo = {}

    def cost_of(mask):
        if mask not in memo:
            sub = [custs[i] for i in range(n) if mask >> i & 1]
            memo[mask] = tsp(sub) if sum(qty[c] for c in sub) <= Q + 1e-9 else float("inf")
        return memo[mask]

    full = (1 << n) - 1
    dp = {0: 0.0}
    for mask in range(1, full + 1):
        low = mask & (-mask)
        best_m = float("inf")
        sub = mask
        while sub:
            if sub & low:
                c = cost_of(sub)
                if c < float("inf") and (mask ^ sub) in dp:
                    best_m = min(best_m, c + dp[mask ^ sub])
            sub = (sub - 1) & mask
        dp[mask] = best_m
    return dp[full]


def check_savings_vs_brute_force(n_inst=120):
    """5. Savings + 2-opt gegen das exakte CVRP (3 bis 6 Kunden): nie besser als das Optimum. Rückgabe (Mittel, Median, P90, Max, Anteil exakt optimal in %) der Lücken."""
    rng = np.random.default_rng(11)
    gaps = []
    for s in range(n_inst):
        n = int(rng.integers(3, 7))
        inst = M.make_instance(500 + s, N=n, D=1)
        custs = list(range(1, n + 1))
        Qv = float(rng.choice([150.0, 250.0, 400.0]))
        qty = {c: float(rng.uniform(40, 120)) for c in custs}
        routes = RT.savings_routes(inst.dist, custs, qty, Qv)
        assert sorted(c for r in routes for c in r) == custs
        for r in routes:
            assert sum(qty[c] for c in r) <= Qv + 1e-9
        heur = sum(RT.route_len(inst.dist, r) for r in routes)
        opt = brute_cvrp(inst.dist, custs, qty, Qv)
        assert heur >= opt - 1e-9, (heur, opt)
        gaps.append(100 * (heur - opt) / opt)
    gaps = np.array(gaps)
    return float(gaps.mean()), float(np.median(gaps)), float(np.percentile(gaps, 90)), float(gaps.max()), float(100 * (gaps < 1e-9).mean())


def check_branches(n_seeds=5, days=90):
    """6. Zweig-Tests (Nullspalten-Falle): jede Regelvariante greift. Mitnahmen nur bei H > 0, gamma steuert sie, frühe Meldegrenze erhöht die Besuche ohne
    Mitnahmen, die enge Flotte stellt Kunden zurück und erzeugt Fehlmengen, sigma = 1 erzeugt Fehlmengen, sigma = 0 KORREKT keine (die Meldegrenze deckt den
    Tagesverbrauch), und mit ausreichender Flotte wird nie zurückgestellt. Über mehrere Seeds summiert (jede einzelne Instanz darf Ausnahmen haben)."""
    out = {"picks_P": 0, "picks_R": 0, "gamma_low": 0, "gamma_high": 0, "visits_E": 0, "visits_R": 0, "picks_E": 0, "deferred_tight": 0, "short_tight": 0, "deferred_free": 0,
           "short_sigma1": 0.0, "short_sigma0": 0.0}
    for s in range(n_seeds):
        inst = M.make_instance(s, D=days)
        r_r = P.simulate(inst, 0.0)
        r_p = P.simulate(inst, 3.0, 0.5)
        out["picks_P"] += r_p["opt_visits"]
        out["picks_R"] += r_r["opt_visits"]
        out["gamma_low"] += P.simulate(inst, 3.0, 0.1)["opt_visits"]
        out["gamma_high"] += P.simulate(inst, 3.0, 1.0)["opt_visits"]
        r_e = P.simulate(inst, 0.0, early=2.0)
        out["visits_E"] += r_e["n_visits"]
        out["visits_R"] += r_r["n_visits"]
        out["picks_E"] += r_e["opt_visits"]
        tight = M.make_instance(s, D=days, Q=300.0, F=1, N=30)
        r_t = P.simulate(tight, 0.0)
        out["deferred_tight"] += r_t["deferred"]
        out["short_tight"] += r_t["short"]
        out["deferred_free"] += P.simulate(M.make_instance(s, D=days, F=6), 0.0)["deferred"]
        out["short_sigma1"] += P.simulate(M.make_instance(s, D=days, sigma=1.0), 0.0)["short"]
        out["short_sigma0"] += P.simulate(M.make_instance(s, D=days, sigma=0.0), 0.0)["short"]
    assert out["picks_P"] > 0 and out["picks_R"] == 0 and out["picks_E"] == 0
    assert out["gamma_low"] < out["gamma_high"]
    assert out["visits_E"] > out["visits_R"]
    assert out["deferred_tight"] > 0 and out["short_tight"] > 0
    assert out["short_sigma1"] > 0 and out["short_sigma0"] == 0.0, (out["short_sigma1"], out["short_sigma0"])
    return out


def check_shortage_share(n_seeds=10, days=90, configs=None):
    """7. Fehlmengen proaktiv (H = 3, gamma = 0,5) gegen reaktiv auf demselben Verbrauchsstrom: Anzahl Instanzen mit MEHR Fehlmenge beim Bündeln je Konfiguration.
    Keine harte Regel (Messung), nur Rückgabe der Zählung."""
    out = []
    for kw in configs or [dict(), dict(sigma=0.0), dict(sigma=0.6), dict(Q=150.0, F=2), dict(Q=300.0, F=1, N=30)]:
        v = 0
        for s in range(n_seeds):
            inst = M.make_instance(s, D=days, **kw)
            if P.simulate(inst, 3.0, 0.5)["short"] > P.simulate(inst, 0.0)["short"] + 1e-9:
                v += 1
        out.append((str(kw), v, n_seeds))
    return out


def check_determinism():
    """8. Determinismus: gleiche Seeds -> gleiche Instanz, gleicher Verbrauchsstrom, gleiche Kosten; der Strom ist von der Regel unabhängig."""
    a, b = M.make_instance(7), M.make_instance(7)
    assert np.array_equal(a.cons, b.cons) and np.array_equal(a.xy, b.xy)
    assert P.simulate(a, 2.0, 0.5)["routing"] == P.simulate(b, 2.0, 0.5)["routing"]
    c1 = a.cons.copy()
    P.simulate(a, 4.0, 1.0)
    assert np.array_equal(c1, a.cons)
    return True


# ---------------------------------------------------------------- Exakter Maßstab (check_oracle.py)
def tiny_instance(seed, N=3, D=4, Q=6):
    rng = np.random.default_rng(seed)
    inst = M.make_instance(seed, N=N, D=D, sigma=0.0, Q=float(Q), F=1, round_dist=0.1)
    inst.mu = np.concatenate([[0], rng.integers(1, 3, size=N)]).astype(float)              # Verbrauch 1..2
    inst.C = np.concatenate([[0], rng.integers(3, 6, size=N)]).astype(float)               # Tank 3..5
    inst.I0 = np.concatenate([[0], [rng.integers(0, int(c) + 1) for c in inst.C[1:]]]).astype(float)
    inst.cons = np.tile(inst.mu, (D, 1))
    inst.cons[:, 0] = 0
    return inst


def dp_optimum(inst, tau):
    """Exaktes Optimum: Zustand = Bestandsvektor am Tagesanfang; je Tag alle Mengenvektoren (Summe <= Q, Tank), Tourkosten = beste Reihenfolge der besuchten Kunden."""
    N, D, Q = inst.N, inst.D, int(inst.Q)
    S = 10
    dist = np.round(inst.dist * S).astype(int)
    route_memo = {}

    def route_cost(vs):
        if not vs:
            return 0
        if vs not in route_memo:
            best = None
            for perm in itertools.permutations(vs):
                c = dist[0, perm[0]] + dist[perm[-1], 0] + sum(dist[a, b] for a, b in zip(perm[:-1], perm[1:]))
                if best is None or c < best:
                    best = c
            route_memo[vs] = int(best)
        return route_memo[vs]

    ranges = [range(int(inst.C[i]) + 1) for i in range(1, N + 1)]
    states = {tuple(int(x) for x in inst.I0[1:]): 0}
    for t in range(D):
        nxt = {}
        for inv, cost in states.items():
            for q in itertools.product(*ranges):
                if sum(q) > Q:
                    continue
                ok = True
                new = []
                for i in range(N):
                    tot = inv[i] + q[i]
                    if tot > inst.C[i + 1]:
                        ok = False
                        break
                    r = tot - int(inst.cons[t][i + 1])
                    if r < 0:
                        ok = False
                        break
                    new.append(r)
                if not ok:
                    continue
                vs = tuple(i + 1 for i in range(N) if q[i] > 0)
                c = cost + route_cost(vs)
                new = tuple(new)
                if new not in nxt or c < nxt[new]:
                    nxt[new] = c
        states = nxt
    tau_i = int(round(tau * S))
    best = min(c - tau_i * sum(inv) for inv, c in states.items())
    return best / S


def check_oracle_vs_dp(seeds=range(40), tau=0.6, min_feasible=12):
    """9. CP-SAT-Modell == unabhängige exakte Rechnung (DP über Bestandszustände + Enumeration der Reihenfolgen) auf winzigen Ganzzahl-Instanzen (3 Kunden, 4 Tage,
    Q = 6): gleiche Zielwerte; der CP-SAT-Plan, im Simulator nachgespielt, hat exakt den Modellzielwert und keine Fehlmenge. Rückgabe die Zahl der zulässigen Instanzen."""
    import irp_oracle as OR
    n = 0
    for seed in seeds:
        inst = tiny_instance(seed)
        res = OR.solve_oracle(inst, tau, time_limit=30, workers=8, seed=1)
        if res is None:
            continue                                        # unzulässig (Tagesbedarf nicht deckbar)
        assert res["status"] == "OPTIMAL", res["status"]
        dp = dp_optimum(inst, tau)
        assert abs(dp - res["obj"]) < 1e-6, (seed, dp, res["obj"])
        rep = P.replay(inst, res["plans"])
        assert abs(P.objective(rep, inst.p, tau) - (res["obj"] + tau * rep["I_start"])) < 1e-6 and rep["short"] == 0
        n += 1
    assert n >= min_feasible, n
    return n
