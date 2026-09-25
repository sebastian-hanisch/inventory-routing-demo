"""Regeln und Simulation des Inventory-Routing-Modells: decide (R reaktiv, P(H, gamma) buendeln, E(L) frueher liefern), apply_day, simulate,
replay, objective. Tagesablauf: Bestand I (Tagesanfang) -> Regel waehlt Touren (Lieferung am selben Morgen, Vorlaufzeit 0) -> Tank wird gefuellt
-> Tagesverbrauch (stochastisch, fuer alle Regeln derselbe Strom) -> Fehlmenge geht verloren (Strafe p je Einheit) -> naechster Tag.
Zielgroesse J = Fahrstrecke + p*Fehlmenge - tau*(Endbestand - Anfangsbestand).
Mechanisch aus messreihe_inventory_routing/ir.py uebernommen, Logik unveraendert. Nur NumPy, kein Streamlit."""
import numpy as np

from irp_routing import cheapest_insertion, savings_routes, route_len, two_opt


# ---------------------------------------------------------------------------- Politik
def decide(inst, I, H=0.0, gamma=0.5, fleet=None, early=0.0):
    """Tourplan fuer heute. Rueckgabe: (Touren als Liste von (Kunde, Menge)-Listen, Zaehler-Dict).
    faellig: I_i < mu_i*(1+kappa*sigma) (heute unter die Meldegrenze). optional: faellig binnen H weiterer Tage.
    H=0 == rein reaktiv. early > 0: die Meldegrenze wird um `early` Tage angehoben (fruehere Pflichtbesuche
    OHNE Tourenbezug) - Gegenprobe "einfach frueher liefern" statt Buendeln."""
    N, dist, Q = inst.N, inst.dist, inst.Q
    F = inst.F if fleet is None else fleet
    thr = inst.mu * (1.0 + inst.kappa * inst.sigma + early)
    idx = np.arange(1, N + 1)
    due = [int(i) for i in idx if I[i] < thr[i] - 1e-12]
    urg = lambda i: (I[i] / inst.mu[i], i)
    due.sort(key=urg)
    qty = {}
    for i in range(1, N + 1):
        qty[i] = min(inst.C[i] - I[i], Q)
    stats = {"deferred": 0, "opt_visits": 0}
    routes = savings_routes(dist, due, qty, Q)
    while len(routes) > F and due:
        due = due[:-1]                 # am wenigsten dringenden Kunden zurueckstellen
        stats["deferred"] += 1
        routes = savings_routes(dist, due, qty, Q)
    if H > 0 and routes:
        loads = [sum(qty[i] for i in r) for r in routes]
        cand = [int(i) for i in idx if i not in due and I[i] < thr[i] + inst.mu[i] * H - 1e-12]
        cand.sort(key=urg)
        for j in cand:
            bestc = None
            for r_idx, r in enumerate(routes):
                if loads[r_idx] + qty[j] > Q + 1e-9:
                    continue
                delta, pos = cheapest_insertion(dist, r, j)
                if bestc is None or delta < bestc[0] - 1e-12:
                    bestc = (delta, r_idx, pos)
            if bestc is None:
                continue
            delta, r_idx, pos = bestc
            if delta <= gamma * 2.0 * dist[0, j] + 1e-12:
                routes[r_idx] = routes[r_idx][:pos] + [j] + routes[r_idx][pos:]
                loads[r_idx] += qty[j]
                stats["opt_visits"] += 1
        routes = [two_opt(dist, r) for r in routes]
    plan = [[(i, qty[i]) for i in r] for r in routes]
    return plan, stats


# ---------------------------------------------------------------------------- Simulation
def apply_day(inst, I, plan, cons_t):
    """Wendet Lieferungen und Tagesverbrauch an. Rueckgabe: (neues I, Fahrstrecke, Fehlmenge, geliefert, bedient, Kunden mit Fehlmenge)."""
    I = I.copy()
    dist_cost = 0.0
    delivered = 0.0
    for route in plan:
        cs = [c for c, _ in route]
        dist_cost += route_len(inst.dist, cs)
        for c, q in route:
            q = min(q, inst.C[c] - I[c])
            I[c] += q
            delivered += q
    served = np.minimum(I, cons_t)
    short = float((cons_t - served).sum())
    n_short = int(((cons_t - served) > 1e-9).sum())
    I = I - served
    return I, dist_cost, short, delivered, float(served.sum()), n_short


def simulate(inst, H=0.0, gamma=0.5, plan_fn=None, fleet=None, early=0.0, record=False):
    """Simuliert inst.D Tage. plan_fn(inst, I, t) -> (plan, stats) ueberschreibt die Standardpolitik."""
    I = inst.I0.copy()
    plans = []
    res = {"routing": 0.0, "short": 0.0, "delivered": 0.0, "served": 0.0, "n_routes": 0, "n_visits": 0,
           "opt_visits": 0, "deferred": 0, "stockout_days": 0, "route_days": 0, "util_sum": 0.0,
           "day_route_km": [], "stops": 0}
    for t in range(inst.D):
        if plan_fn is None:
            plan, st = decide(inst, I, H, gamma, fleet, early)
        else:
            plan, st = plan_fn(inst, I, t)
        I2, dc, sh, dl, sv, ns = apply_day(inst, I, plan, inst.cons[t])
        res["routing"] += dc
        res["short"] += sh
        res["delivered"] += dl
        res["served"] += sv
        res["n_routes"] += len(plan)
        res["n_visits"] += sum(len(r) for r in plan)
        res["opt_visits"] += st.get("opt_visits", 0)
        res["deferred"] += st.get("deferred", 0)
        res["route_days"] += 1 if plan else 0
        res["util_sum"] += sum(sum(q for _, q in r) / inst.Q for r in plan)
        res["stockout_days"] += ns
        res["day_route_km"].append(dc)
        if record:
            plans.append(plan)
        I = I2
    res["I_end"] = float(I[1:].sum())
    res["I_start"] = float(inst.I0[1:].sum())
    res["dI"] = res["I_end"] - res["I_start"]
    if record:
        res["plans"] = plans
    return res


def replay(inst, plans):
    """Bewertet einen vorgegebenen Tourplan (Liste je Tag: Liste von Touren [(Kunde, Menge)...]) im selben Modell."""
    return simulate(inst, plan_fn=lambda ins, I, t: (plans[t], {}), record=True)


def objective(res, p, tau):
    return res["routing"] + p * res["short"] - tau * res["dI"]
