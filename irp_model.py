"""Instanz des Inventory-Routing-Modells (ein Depot, N Kunden, Tank, Verbrauch): Standort, Verbrauchsraten, Tankgroessen, Anfangsbestand und
der Verbrauchsstrom je Tag (fuer alle Regeln derselbe). Mechanisch aus messreihe_inventory_routing/ir.py uebernommen, Logik unveraendert.
Nur NumPy, kein Streamlit."""
import numpy as np


class Inst:
    pass


def make_instance(seed, N=20, size=100.0, sigma=0.3, Q=300.0, F=6, pfac=0.05, D=90, depot="center",
                  clustered=False, mu_lo=6.0, mu_hi=14.0, tank_lo=8.0, tank_hi=14.0, kappa=1.0, round_dist=None):
    rng = np.random.default_rng(seed)
    inst = Inst()
    inst.N, inst.size, inst.sigma, inst.Q, inst.F, inst.D = N, size, sigma, Q, F, D
    inst.p = pfac * size
    inst.kappa = kappa
    if clustered:
        k = 4
        cen = rng.uniform(0.15 * size, 0.85 * size, size=(k, 2))
        lab = rng.integers(0, k, size=N)
        pts = cen[lab] + rng.normal(0, 0.07 * size, size=(N, 2))
        pts = np.clip(pts, 0, size)
    else:
        pts = rng.uniform(0, size, size=(N, 2))
    dep = {"center": (size / 2, size / 2), "corner": (0.0, 0.0), "far": (size / 2, -0.5 * size)}[depot]
    xy = np.vstack([np.array(dep)[None, :], pts])
    Dm = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    if round_dist:
        Dm = np.round(Dm / round_dist) * round_dist
    inst.xy, inst.dist = xy, Dm
    mu = rng.uniform(mu_lo, mu_hi, size=N)
    inst.mu = np.concatenate([[0.0], mu])
    inst.C = np.concatenate([[0.0], mu * rng.uniform(tank_lo, tank_hi, size=N)])
    inst.I0 = np.concatenate([[0.0], inst.C[1:] * rng.uniform(0.25, 1.0, size=N)])
    crng = np.random.default_rng(seed * 7919 + 13)
    if sigma > 0:
        shape = 1.0 / (sigma * sigma)
        cons = crng.gamma(shape, sigma * sigma, size=(D, N + 1)) * inst.mu[None, :]
    else:
        cons = np.tile(inst.mu, (D, 1))
    cons[:, 0] = 0.0
    inst.cons = cons
    return inst
