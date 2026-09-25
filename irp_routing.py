"""Tourenbausteine des Inventory-Routing-Modells: Tourlaenge, 2-opt, Savings (Clarke-Wright) und billigste Einfuegung. Mechanisch aus
messreihe_inventory_routing/ir.py uebernommen, Logik unveraendert. Nur Standardbibliothek."""


def route_len(dist, route):
    """Laenge der Tour Depot -> route -> Depot."""
    if not route:
        return 0.0
    s = dist[0, route[0]] + dist[route[-1], 0]
    for a, b in zip(route[:-1], route[1:]):
        s += dist[a, b]
    return float(s)


def two_opt(dist, route):
    best = list(route)
    n = len(best)
    if n < 3:
        return best
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                a = 0 if i == 0 else best[i - 1]
                b = best[i]
                c = best[j]
                d = 0 if j == n - 1 else best[j + 1]
                delta = dist[a, c] + dist[b, d] - dist[a, b] - dist[c, d]
                if delta < -1e-9:
                    best[i:j + 1] = best[i:j + 1][::-1]
                    improved = True
    return best


def savings_routes(dist, custs, qty, Q):
    """Clarke-Wright (parallel) fuer die Kunden `custs` mit Mengen qty[i]; Kapazitaet Q je Tour."""
    if not custs:
        return []
    routes = {i: [i] for i in custs}
    load = {i: qty[i] for i in custs}
    where = {i: i for i in custs}          # Kunde -> Schluessel seiner Tour
    sv = []
    for a in range(len(custs)):
        for b in range(a + 1, len(custs)):
            i, j = custs[a], custs[b]
            sv.append((dist[0, i] + dist[0, j] - dist[i, j], i, j))
    sv.sort(key=lambda x: (-x[0], x[1], x[2]))
    for s, i, j in sv:
        if s <= 1e-12:
            break
        ri, rj = where[i], where[j]
        if ri == rj:
            continue
        r1, r2 = routes[ri], routes[rj]
        if load[ri] + load[rj] > Q + 1e-9:
            continue
        # i und j muessen Endpunkte sein
        if r1[-1] == i and r2[0] == j:
            new = r1 + r2
        elif r1[0] == i and r2[-1] == j:
            new = r2 + r1
        elif r1[-1] == i and r2[-1] == j:
            new = r1 + r2[::-1]
        elif r1[0] == i and r2[0] == j:
            new = r1[::-1] + r2
        else:
            continue
        routes[ri] = new
        load[ri] += load[rj]
        for c in r2:
            where[c] = ri
        del routes[rj], load[rj]
    return [two_opt(dist, r) for r in routes.values()]


def cheapest_insertion(dist, route, j):
    """(Mehrstrecke, Position) der billigsten Einfuegung von j in die Tour."""
    best = (float("inf"), 0)
    prev = 0
    for pos in range(len(route) + 1):
        nxt = route[pos] if pos < len(route) else 0
        delta = dist[prev, j] + dist[j, nxt] - dist[prev, nxt]
        if delta < best[0] - 1e-12:
            best = (delta, pos)
        prev = nxt
    return best
