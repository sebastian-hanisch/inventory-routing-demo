"""Exakter Maßstab für die Kleininstanz (Multi-Tage-Modell) per CP-SAT (OR-Tools): deterministischer Verbrauch, eine Tour je Tag.

Entscheidung: welche Kunden an welchem Tag (Kreis durch das Depot, AddCircuit mit optionalen Knoten), welche Menge; Bestand >= 0 immer
(keine Fehlmengen), Tank-Obergrenze, Tourkapazität Q. Ziel wie in irp_policy: Fahrstrecke - tau*(Endbestand - Anfangsbestand). Alle Zahlen
ganzzahlig (Distanz auf 0,1 gerundet, Faktor 10).

Aus messreihe_inventory_routing/oracle.py übernommen (Modell unverändert; die Plan-Rekonstruktion liest die Bogenvariablen jetzt aus einem
Dict statt über Variablennamen). OR-Tools wird erst in solve_oracle importiert (lazy), damit die übrige App und die Tests ohne OR-Tools laufen.
Arbeiter: mit 8 (Standard) löst CP-SAT die Kleininstanz in 0,3 bis 1,2 s, mit 1 bis 4 Arbeitern dauert dieselbe Instanz 4 bis 25 s (gemessen; im
Portfolio der 8 Arbeiter steckt eine schnelle Teilstrategie). Größenbegrenzung: check_size lehnt Instanzen über MAX_CUSTOMERS Kunden oder MAX_DAYS Tagen ab (das Modell wächst mit Kunden² · Tagen)."""
import time

import numpy as np

import irp_model
import irp_policy

MAX_CUSTOMERS = 8                   # größere Instanzen brauchen Minuten bis Stunden: nicht live
MAX_DAYS = 7
SMALL_CUSTOMERS, SMALL_DAYS = 7, 6  # die Kleininstanz der Messreihe und des Exakt-Tabs
TAU = 0.6                           # Bewertung des Restbestands in der Kleininstanz (wie in der Messreihe)


def check_size(n_customers, n_days):
    """ValueError, wenn die Instanz für den Live-Löser zu groß ist."""
    if n_customers > MAX_CUSTOMERS or n_days > MAX_DAYS:
        raise ValueError(f"Instanz zu groß für den exakten Löser: {n_customers} Kunden und {n_days} Tage (erlaubt: höchstens "
                         f"{MAX_CUSTOMERS} Kunden und {MAX_DAYS} Tage)")


def make_oracle_instance(seed, N=SMALL_CUSTOMERS, D=SMALL_DAYS, Q=150.0):
    inst = irp_model.make_instance(seed, N=N, D=D, sigma=0.0, Q=Q, F=1, round_dist=0.1, tank_lo=3.0, tank_hi=6.0)
    inst.mu = np.round(inst.mu)
    inst.C = np.maximum(np.round(inst.C), inst.mu + 1)
    inst.I0 = np.minimum(np.round(inst.I0), inst.C)
    inst.cons = np.tile(inst.mu, (D, 1))
    inst.cons[:, 0] = 0.0
    return inst


def solve_oracle(inst, tau, time_limit=60.0, workers=8, seed=1):
    """Bewiesenes (oder beste gefundene) Optimum der Kleininstanz. Rückgabe None, wenn unzulässig (Tagesbedarf nicht deckbar), sonst dict mit
    status, obj (Zielwert des Modells, ohne den Anfangsbestand-Term), bound, plans (je Tag eine Liste mit einer Tour [(Kunde, Menge)...]),
    time, tau."""
    from ortools.sat.python import cp_model                     # lazy: nur der Exakt-Tab braucht OR-Tools

    check_size(inst.N, inst.D)
    N, D, Q = inst.N, inst.D, int(inst.Q)
    S = 10
    dist = np.round(inst.dist * S).astype(int)
    tau_i = int(round(tau * S))
    m = cp_model.CpModel()
    q = {}
    I = {}
    vis = {}
    arc = {}
    obj = []
    for t in range(D):
        arcs = []
        for i in range(N + 1):
            for j in range(N + 1):
                if i == j:
                    continue
                lit = m.NewBoolVar("x_%d_%d_%d" % (t, i, j))
                arc[t, i, j] = lit
                arcs.append((i, j, lit))
                obj.append(int(dist[i, j]) * lit)
        skip = [m.NewBoolVar("skip_%d_%d" % (t, i)) for i in range(N + 1)]
        for i in range(N + 1):
            arcs.append((i, i, skip[i]))
        m.AddCircuit(arcs)
        for i in range(1, N + 1):
            vis[t, i] = skip[i].Not()
            m.AddImplication(skip[0], skip[i])          # kein Depot in der Tour -> keine Kunden
        for i in range(1, N + 1):
            q[t, i] = m.NewIntVar(0, int(inst.C[i]), "q_%d_%d" % (t, i))
            m.Add(q[t, i] == 0).OnlyEnforceIf(skip[i])
        m.Add(sum(q[t, i] for i in range(1, N + 1)) <= Q)
    for i in range(1, N + 1):
        prev = int(inst.I0[i])
        for t in range(D):
            I[t, i] = m.NewIntVar(0, int(inst.C[i]), "I_%d_%d" % (t, i))
            m.Add(prev + q[t, i] <= int(inst.C[i]))                  # Tank vor dem Verbrauch
            m.Add(I[t, i] == prev + q[t, i] - int(inst.cons[t][i]))  # Bestand am Tagesende >= 0 (Domain)
            prev = I[t, i]
    end_inv = sum(I[D - 1, i] for i in range(1, N + 1))
    m.Minimize(sum(obj) - tau_i * end_inv)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = seed
    t0 = time.time()
    status = solver.Solve(m)
    el = time.time() - t0
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None
    plans = []
    for t in range(D):
        succ = {}
        for i in range(N + 1):
            for j in range(N + 1):
                if i != j and solver.BooleanValue(arc[t, i, j]):
                    succ[i] = j
        route = []
        cur = succ.get(0)
        while cur is not None and cur != 0:
            route.append(cur)
            cur = succ.get(cur)
        plans.append([[(c, float(solver.Value(q[t, c]))) for c in route]] if route else [])
    return {"status": solver.StatusName(status), "obj": solver.ObjectiveValue() / S, "bound": solver.BestObjectiveBound() / S,
            "plans": plans, "time": el, "tau": tau}


def objective_of_plan(inst, res, tau):
    """Zielwert eines nachgespielten Plans im Modell der Simulation (Fahrstrecke + Strafe·Fehlmenge − τ·Bestandsänderung)."""
    return irp_policy.objective(res, inst.p, tau)


def compare_small(inst, opt, H, gamma, early, tau=TAU):
    """Optimum gegen die Regeln R, P(H, gamma) und E(early) auf derselben Kleininstanz. `opt` ist das Ergebnis von solve_oracle.
    Rückgabe dict: J je Regel (Zielwert der Simulation), Lücken in % über dem Optimum, Touren, Fehlmengen und das nachgespielte Optimum."""
    rep = irp_policy.replay(inst, opt["plans"])
    j_opt = objective_of_plan(inst, rep, tau)
    rules = {"R": (0.0, 0.5, 0.0), "P": (float(H), float(gamma), 0.0), "E": (0.0, 0.5, float(early))}
    out = {"optimum": {"J": j_opt, "routes": rep["n_routes"], "routing": rep["routing"], "short": rep["short"], "visits": rep["n_visits"],
                       "status": opt["status"], "time": opt["time"]}}
    for name, (h, g, e) in rules.items():
        r = irp_policy.simulate(inst, h, g, early=e, record=True)
        j = irp_policy.objective(r, inst.p, tau)
        out[name] = {"J": j, "gap": 100.0 * (j - j_opt) / j_opt, "routes": r["n_routes"], "routing": r["routing"], "short": r["short"],
                     "visits": r["n_visits"], "plans": r["plans"]}
    out["optimum"]["plans"] = rep["plans"]
    return out


def cooldown_left(last, now, cooldown):
    """Verbleibende Pause in Sekunden seit dem letzten Aufruf (`last` und `now` aus derselben monotonen Uhr; None = noch nie gerechnet)."""
    if last is None:
        return 0.0
    return max(0.0, cooldown - (now - last))


def solve_small(wagon, seed, H, gamma, early, time_limit=20.0, workers=8):
    """Kleininstanz (7 Kunden, 6 Tage, Wagen `wagon`, Seed `seed`) exakt lösen und mit den Regeln R, P(H, gamma), E(early) vergleichen. Rückgabe None, wenn die
    Instanz unzulässig ist, sonst dict mit status, time, compare (siehe compare_small). ImportError, wenn OR-Tools fehlt; ValueError bei zu großer Instanz."""
    inst = make_oracle_instance(int(seed), Q=float(wagon))
    check_size(inst.N, inst.D)
    opt = solve_oracle(inst, TAU, time_limit=time_limit, workers=workers)
    if opt is None:
        return None
    return {"status": opt["status"], "time": opt["time"], "seed": int(seed), "wagon": int(wagon), "compare": compare_small(inst, opt, H, gamma, early)}
