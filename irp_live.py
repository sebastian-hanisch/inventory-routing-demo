"""Live-Instanz: EINE Instanz über 120 Tage mit allen drei Regeln (R reaktiv, P(H, gamma) bündeln, E(L) früher liefern) auf demselben Verbrauchsstrom.

Die Rechnung läuft über irp_model / irp_policy (unverändert aus der Messreihe); hier stehen nur das Zusammenstellen der Ergebnisse, der
Bestandsverlauf je Tag (aus den aufgezeichneten Plänen nachgespielt) und die Tagessicht (Touren, Halteliste, Mitnehmer). Gemessen: Instanz und je
Regel zusammen unter 0,05 s, deshalb kein Knopf; die App speichert das Ergebnis je Einstellung (st.cache_data).

Zielgröße wie in der Messreihe: J = Fahrstrecke + p * Fehlmenge - tau * (Endbestand - Anfangsbestand). tau ist hier der Fahrkostensatz je gelieferter
Einheit der reaktiven Regel auf DIESER Instanz (in der Messreihe das Mittel über 200 Instanzen; die Wahl ändert den Basisgewinn kaum: 17,0 / 17,0 / 17,1 %
bei tau = 0 / 1 / 2 · tau_ref)."""
import numpy as np

import irp_constants as C
import irp_model
import irp_policy

RULE_NAMES = {"R": "Reaktiv", "P": "Bündeln", "E": "Früher liefern"}
RES_KEYS = ("routing", "short", "delivered", "served", "n_routes", "n_visits", "opt_visits", "deferred", "stockout_days", "route_days", "util_sum",
            "I_end", "I_start", "dI", "day_route_km")


def make_live_instance(customers, capacity, fleet, sigma, penalty, seed, days=C.DAYS):
    """Instanz der Reglerstellung (gleiche Parameter wie die Zellen der Messreihe; die Strafe über pfac wie im Sweep)."""
    return irp_model.make_instance(int(seed), N=int(customers), Q=float(capacity), F=int(fleet), sigma=float(sigma), pfac=C.PENALTY_PFAC[int(penalty)], D=days)


def inventory_path(inst, plans):
    """Bestand am Tagesanfang: Zeile t = Bestand vor der Lieferung des Tages t (Zeile `days` = Endbestand). Aus den aufgezeichneten Plänen nachgespielt."""
    I = inst.I0.copy()
    out = [I.copy()]
    for t, plan in enumerate(plans):
        I = irp_policy.apply_day(inst, I, plan, inst.cons[t])[0]
        out.append(I.copy())
    return np.array(out)


def rule_settings(H, gamma, early):
    """Die drei Regeln als (H, gamma, early)-Tripel für irp_policy.simulate."""
    return {"R": (0.0, 0.5, 0.0), "P": (float(H), float(gamma), 0.0), "E": (0.0, 0.5, float(early))}


def solve_live(customers, capacity, fleet, sigma, penalty, H, gamma, early, seed, days=C.DAYS):
    """Alle drei Regeln auf einer Instanz. Rückgabe dict: Instanzdaten, tau, p und je Regel res (Kennzahlen), plans, I (Bestandsverlauf), J, gain (in %
    der reaktiven Kosten, positiv = billiger als reaktiv)."""
    inst = make_live_instance(customers, capacity, fleet, sigma, penalty, seed, days)
    rules = {}
    for name, (h, g, e) in rule_settings(H, gamma, early).items():
        res = irp_policy.simulate(inst, h, g, early=e, record=True)
        rules[name] = {"res": {k: res[k] for k in RES_KEYS}, "plans": res["plans"], "I": inventory_path(inst, res["plans"])}
    r = rules["R"]["res"]
    tau = r["routing"] / r["delivered"] if r["delivered"] > 0 else 0.0
    for rule in rules.values():
        rule["J"] = irp_policy.objective(rule["res"], inst.p, tau)
    j_r = rules["R"]["J"]
    for name, rule in rules.items():
        rule["gain"] = 100.0 * (j_r - rule["J"]) / j_r if j_r > 0 else 0.0
    return {"seed": int(seed), "N": inst.N, "Q": inst.Q, "F": inst.F, "sigma": inst.sigma, "kappa": inst.kappa, "p": inst.p, "days": inst.D, "tau": tau,
            "xy": inst.xy, "C": inst.C, "mu": inst.mu, "I0": inst.I0, "H": float(H), "gamma": float(gamma), "early": float(early), "rules": rules}


def threshold(live, early=0.0):
    """Meldegrenze je Kunde (Index 0 = Depot): mu * (1 + kappa * sigma + early)."""
    return live["mu"] * (1.0 + live["kappa"] * live["sigma"] + early)


def stop_kind(live, rule, I_i, i):
    """'fällig' (Bestand unter der Meldegrenze der Regel) oder 'Mitnahme' (nur bei Bündeln: Kunde ohne Fälligkeit auf einer vorhandenen Tour)."""
    early = live["early"] if rule == "E" else 0.0
    return "fällig" if I_i < threshold(live, early)[i] - 1e-12 else "Mitnahme"


def day_view(live, rule, day):
    """Tagessicht einer Regel: `day` zählt ab 1. Rückgabe dict mit Bestand zu Tagesbeginn (I), Touren (Halteliste mit Menge, Bestand vorher, Reichweite in
    Tagen, Art), Tourlänge, Last, Kennzahlen des Tages."""
    t = int(day) - 1
    rl = live["rules"][rule]
    I = rl["I"][t]
    plan = rl["plans"][t]
    tours = []
    for k, route in enumerate(plan, 1):
        stops = []
        for c, q in route:
            q_eff = min(q, live["C"][c] - I[c])
            stops.append({"customer": int(c), "qty": float(q_eff), "fill_before": float(I[c]), "fill_share": float(I[c] / live["C"][c]),
                          "reach": float(I[c] / live["mu"][c]), "kind": stop_kind(live, rule, I[c], c)})
        tours.append({"tour": k, "stops": stops, "load": float(sum(s["qty"] for s in stops)), "length": tour_length(live, [c for c, _ in route])})
    return {"day": int(day), "rule": rule, "I": I, "tours": tours, "km": rl["res"]["day_route_km"][t], "n_stops": sum(len(x["stops"]) for x in tours),
            "n_pickups": sum(1 for x in tours for s in x["stops"] if s["kind"] == "Mitnahme")}


def tour_length(live, customers):
    """Länge der Tour Depot -> customers -> Depot (Euklid) aus den Koordinaten der Instanz."""
    if not customers:
        return 0.0
    xy = live["xy"]
    pts = [0] + list(customers) + [0]
    return float(sum(np.hypot(*(xy[a] - xy[b])) for a, b in zip(pts[:-1], pts[1:])))


def mean_fill(live, rule):
    """Mittlerer Füllstand der Kunden am Tagesanfang über alle Tage, in Teilen des Tanks (0 bis 1)."""
    return float((live["rules"][rule]["I"][:live["days"], 1:] / live["C"][1:]).mean())


def pickup_days(live, limit=8):
    """Die ersten Tage (ab 1), an denen die Regel Bündeln mindestens einen Mitnehmer auf eine Tour nimmt."""
    days = []
    for t, plan in enumerate(live["rules"]["P"]["plans"]):
        I = live["rules"]["P"]["I"][t]
        if any(stop_kind(live, "P", I[c], c) == "Mitnahme" for route in plan for c, _ in route):
            days.append(t + 1)
            if len(days) >= limit:
                break
    return days


def day_table(live):
    """Tabelle über alle Tage: Touren und Kilometer je Regel und Zahl der Mitnehmer beim Bündeln."""
    rows = []
    for t in range(live["days"]):
        row = {"Tag": t + 1}
        for name in ("R", "P", "E"):
            plan = live["rules"][name]["plans"][t]
            row[f"Touren {RULE_NAMES[name]}"] = len(plan)
            row[f"km {RULE_NAMES[name]}"] = float(live["rules"][name]["res"]["day_route_km"][t])
        I = live["rules"]["P"]["I"][t]
        row["Mitnehmer"] = sum(1 for route in live["rules"]["P"]["plans"][t] for c, _ in route if stop_kind(live, "P", I[c], c) == "Mitnahme")
        rows.append(row)
    return rows
