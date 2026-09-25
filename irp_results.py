"""Laden und Auswerten der vorgerechneten Messreihe (data/irp_results.json, erzeugt von tools/build_results.py aus sweep_data.json und den
Orakel-Läufen der Messreihe): Zellen, Zell-Zuordnung, Gitter, Urteil in drei Zuständen, Kapazitätshebel, Regime-Tabelle, exakter Maßstab.

Die App rechnet die Messreihe NIE live (Sweep etwa 126 s auf 12 Prozessen); sie liest nur diese Datei. Reine Rechnung, kein Streamlit."""
import functools
import json
import pathlib

import irp_constants as C
import irp_format as F

DATA_PATH = pathlib.Path(__file__).resolve().parent / "data" / "irp_results.json"

STATE_TEXT = {C.STATE_LOHNT: "Bündeln lohnt", C.STATE_WENIG: "Bündeln bringt hier wenig", C.STATE_TEURER: "Diese Mitnahmeregel ist hier teurer als reaktiv"}

# Reihenfolge der Reglerparameter für die Zell-Zuordnung (bei gleichem Abstand gewinnt die Zelle, deren variierter Parameter früher steht)
PARAM_PRIORITY = ("capacity", "fleet", "customers", "sigma", "penalty")
PARAM_LABELS = {"capacity": "Wagenkapazität", "fleet": "Touren je Tag", "customers": "Kunden", "sigma": "Verbrauchsschwankung", "penalty": "Fehlmengenstrafe"}
GROUP_LABELS = {"basis": "Basisfall", "sigma": "Verbrauchsschwankung", "kapazitaet": "Wagenkapazität", "dichte": "Dichte", "geometrie": "Geometrie",
                "strafe": "Fehlmengenstrafe", "flotte": "Flotte", "meldegrenze": "Meldegrenze", "tank": "Tankgröße"}


def verdict(mean, se, factor=C.SE_FACTOR):
    """Urteil in drei Zuständen: 'pos', 'neg', 'none' - ein Vorzeichen nur, wenn der Betrag des Mittels mehr als `factor` Standardfehler beträgt."""
    if mean > factor * se:
        return "pos"
    if mean < -factor * se:
        return "neg"
    return "none"


def state_of(mean, se):
    """Meldungszustand der Hauptansicht aus dem gemessenen Gewinn einer Regel: 'lohnt' (mindestens 5 % und über 2 Standardfehler), 'teurer' (unter
    minus 2 Standardfehler), sonst 'wenig'."""
    if mean >= C.WIN_MIN_PCT and verdict(mean, se) == "pos":
        return C.STATE_LOHNT
    if verdict(mean, se) == "neg":
        return C.STATE_TEURER
    return C.STATE_WENIG


@functools.lru_cache(maxsize=4)
def _load(path):
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def load_results(path=None):
    """Die vorgerechnete Messreihe (nicht verändern: das Ergebnis ist zwischengespeichert)."""
    return _load(str(path or DATA_PATH))


def cells(data):
    return data["cells"]


def _de(v, digits=1):
    return f"{v:.{digits}f}".replace(".", ",")


# ----------------------------------------------------------------------------------------------------------------
# Zellen finden, beschriften, Gitter
# ----------------------------------------------------------------------------------------------------------------
def by_name(data, name):
    hits = [c for c in cells(data) if c["name"] == name]
    if len(hits) != 1:
        raise KeyError((name, len(hits)))
    return hits[0]


def find_cell(data, **overrides):
    """Die Zelle, die sich vom Basisfall genau in `overrides` unterscheidet (cfg-Schlüssel wie in irp_constants.BASE_CFG, plus penalty)."""
    want = dict(C.BASE_CFG, **overrides)
    hits = [c for c in cells(data) if all(c["cfg"].get(k) == v for k, v in want.items())]
    if len(hits) != 1:
        raise KeyError((overrides, len(hits)))
    return hits[0]


def p_key(H, gamma):
    return f"P_H{int(H)}_g{float(gamma):.2f}"


def parse_p_key(key):
    """(H, gamma) aus dem Schlüssel einer P-Regel, z. B. 'P_H8_g0.25' -> (8, 0.25)."""
    return int(key.split("_H")[1].split("_")[0]), float(key.split("_g")[1])


def e_key(L):
    return f"E_L{int(L)}"


def p_gain(cell, H, gamma):
    """Gemessener Gewinn (mean, se, median, q1, q3, share_loss) von P(H, gamma) in der Zelle."""
    return cell["P"][p_key(H, gamma)]


def e_gain(cell, L):
    return cell["E"][e_key(L)]


def cfg_setting(cfg):
    """(Kunden, Wagen, Touren je Tag, sigma, Strafe) einer Zelle, wenn sie ganz auf den Reglerstufen liegt und sonst dem Basisfall entspricht
    (gleiche Geometrie, Meldegrenze, Tankgröße), sonst None."""
    base = C.BASE_CFG
    if any(cfg[k] != base[k] for k in ("kappa", "depot", "clustered", "tank_lo", "tank_hi")):
        return None
    n, q, f, s, pen = cfg["N"], int(cfg["Q"]), cfg["F"], cfg["sigma"], cfg["penalty"]
    if (n not in C.CUSTOMER_OPTIONS or q not in C.CAPACITY_OPTIONS or f not in C.FLEET_OPTIONS or s not in C.SIGMA_OPTIONS
            or int(pen) not in C.PENALTY_OPTIONS or abs(pen - int(pen)) > 1e-9):
        return None
    return n, q, f, s, int(pen)


def setting_text(setting):
    n, q, f, s, pen = setting
    return f"{n} Kunden, Wagen {q}, {f} Tour{'en' if f > 1 else ''} je Tag, Verbrauchsschwankung {F.fmt_level(s)}, Strafe {pen}"


def cell_label(cell):
    """Beschriftung einer Zelle aus dem Unterschied zum Basisfall (deutsch)."""
    cfg, base = cell["cfg"], C.BASE_CFG
    parts = []
    if cfg["N"] != base["N"]:
        parts.append(f"{cfg['N']} Kunden")
    if cfg["Q"] != base["Q"]:
        parts.append(f"Wagen {int(cfg['Q'])}")
    if cfg["F"] != base["F"]:
        parts.append(f"{cfg['F']} Tour{'en' if cfg['F'] > 1 else ''} je Tag")
    if cfg["sigma"] != base["sigma"]:
        parts.append(f"Schwankung {F.fmt_level(cfg['sigma'])}")
    if cfg["pfac"] != base["pfac"]:
        parts.append(f"Strafe {int(cfg['penalty'])}")
    if cfg["kappa"] != base["kappa"]:
        parts.append(f"Meldegrenze κ = {_de(cfg['kappa'], 0)}")
    if cfg["clustered"]:
        parts.append("geklumpte Kunden")
    if cfg["depot"] != base["depot"]:
        parts.append({"far": "Depot weit außen", "corner": "Depot in der Ecke"}[cfg["depot"]])
    if (cfg["tank_lo"], cfg["tank_hi"]) != (base["tank_lo"], base["tank_hi"]):
        parts.append(f"Tank {_de(cfg['tank_lo'], 0)} bis {_de(cfg['tank_hi'], 0)} Tage")
    return ", ".join(parts) if parts else "Basisfall"


def cell_group(cell):
    name = cell["name"]
    for prefix, key in (("sigma", "sigma"), ("q", "kapazitaet"), ("n", "dichte"), ("cluster", "geometrie"), ("depot", "geometrie"), ("pen", "strafe"),
                        ("flotte", "flotte"), ("kappa", "meldegrenze"), ("tank", "tank")):
        if name.startswith(prefix):
            return GROUP_LABELS[key]
    return GROUP_LABELS["basis"]


def cell_setting_text(cell):
    s = cfg_setting(cell["cfg"])
    return setting_text(s) if s else cell_label(cell)


# ----------------------------------------------------------------------------------------------------------------
# Zell-Zuordnung: die Regler stehen auf gemessenen Stufen, aber die Messreihe variiert je Zelle nur EINEN Parameter
# ----------------------------------------------------------------------------------------------------------------
def _level_index(setting):
    n, q, f, s, pen = setting
    return dict(customers=C.CUSTOMER_OPTIONS.index(n), capacity=C.CAPACITY_OPTIONS.index(q), fleet=C.FLEET_OPTIONS.index(f),
                sigma=C.SIGMA_OPTIONS.index(s), penalty=C.PENALTY_OPTIONS.index(pen))


BASE_SETTING = (C.CUSTOMER_DEFAULT, C.CAPACITY_DEFAULT, C.FLEET_DEFAULT, C.SIGMA_DEFAULT, C.PENALTY_DEFAULT)


def assignable_cells(data):
    """Die gemessenen Zellen, die ganz auf den Reglerstufen liegen: [(Zelle, Einstellung)]."""
    out = []
    for c in cells(data):
        s = cfg_setting(c["cfg"])
        if s is not None:
            out.append((c, s))
    return out


def differing_params(setting_a, setting_b):
    ia, ib = _level_index(setting_a), _level_index(setting_b)
    return [p for p in PARAM_PRIORITY if ia[p] != ib[p]]


def nearest_cell(data, customers, capacity, fleet, sigma, penalty):
    """Die nächstliegende gemessene Zelle zu einer Reglerstellung. Rückgabe (Zelle, exakt, abweichende Parameter): exakt heißt, die Zelle hat genau
    diese Einstellung. Abstand = Summe der Stufenabstände; bei Gleichstand gewinnt die Zelle, deren variierter Parameter in PARAM_PRIORITY früher
    steht, dann die frühere in der Messreihe."""
    want = (int(customers), int(capacity), int(fleet), float(sigma), int(penalty))
    iw = _level_index(want)
    best = None
    for pos, (cell, setting) in enumerate(assignable_cells(data)):
        ic = _level_index(setting)
        dist = sum(abs(iw[p] - ic[p]) for p in PARAM_PRIORITY)
        varied = differing_params(setting, BASE_SETTING)
        rank = PARAM_PRIORITY.index(varied[0]) if varied else -1
        key = (dist, rank, pos)
        if best is None or key < best[0]:
            best = (key, cell, setting)
    _, cell, setting = best
    diff = differing_params(want, setting)
    return cell, not diff, diff


def assignment_note(customers, capacity, fleet, sigma, penalty, cell, diff):
    """Hinweis zur Vergleichsspalte; leer, wenn die gemessene Zelle genau die Einstellung ist."""
    if not diff:
        return ""
    want = setting_text((customers, capacity, fleet, sigma, penalty))
    return (f"Für Ihre Einstellung ({want}) gibt es keine eigene Messreihe: die Messreihe variiert je Zelle nur einen Parameter "
            f"ausgehend vom Basisfall. Die Vergleichsspalte zeigt die nächstliegende gemessene Zelle ({cell_setting_text(cell)}); "
            f"abweichend: {', '.join(PARAM_LABELS[p] for p in diff)}.")


def coverage(data):
    """Abdeckung der Reglerstellungen durch gemessene Zellen (Bau-Vorprüfung AP 0): Zahl aller Kombinationen aus Kunden, Wagen, Touren, sigma und
    Strafe, davon exakt gemessen, Verteilung nach dem Abstand zur nächstliegenden Zelle, und die Zellen, die auf keiner Reglerstufenkombination liegen."""
    import itertools
    combos = list(itertools.product(C.CUSTOMER_OPTIONS, C.CAPACITY_OPTIONS, C.FLEET_OPTIONS, C.SIGMA_OPTIONS, C.PENALTY_OPTIONS))
    by_dist = {}
    for combo in combos:
        _, exact, diff = nearest_cell(data, *combo)
        by_dist[len(diff)] = by_dist.get(len(diff), 0) + 1
    assignable = {c["name"] for c, _ in assignable_cells(data)}
    return {"total": len(combos), "exact": by_dist.get(0, 0), "by_distance": dict(sorted(by_dist.items())),
            "cells": len(cells(data)), "assignable": sorted(assignable), "unreachable": [c["name"] for c in cells(data) if c["name"] not in assignable]}


# ----------------------------------------------------------------------------------------------------------------
# Meldung und Tabellen
# ----------------------------------------------------------------------------------------------------------------
def judge(cell, H, gamma):
    """Meldungszustand und Kennzahlen der Regel P(H, gamma) in der Zelle."""
    g = p_gain(cell, H, gamma)
    return {"state": state_of(g["mean"], g["se"]), "verdict": verdict(g["mean"], g["se"]), **g}


def heat_matrix(cell):
    """Gewinn (Mittel) je Vorschau H (Spalten) und Mitnahmeschwelle gamma (Zeilen)."""
    return [[p_gain(cell, H, g)["mean"] for H in C.HORIZON_OPTIONS] for g in C.GAMMA_OPTIONS]


def grid_summary(cell):
    """Zellen des H-gamma-Gitters: Zahl positiver Zellen (Mittel > 0), Zahl im Mittel teurer, beste Zelle (in-sample) und Zahl der Zellen höchstens
    3 Prozentpunkte unter der besten."""
    means = {k: v["mean"] for k, v in cell["P"].items()}
    best = max(means.values())
    return {"n": len(means), "positive": sum(1 for v in means.values() if v > 0), "negative": sum(1 for v in means.values() if v < 0),
            "best_key": max(means, key=means.get), "best": best, "near_best": sum(1 for v in means.values() if v >= best - 3.0)}


def _row_gain(cell, H, gamma):
    g = p_gain(cell, H, gamma)
    return dict(g, verdict=verdict(g["mean"], g["se"]), state=state_of(g["mean"], g["se"]))


def factor_cells(data, factor):
    """Zellen, die nur einen Parameter des Basisfalls variieren: 'capacity' (Q), 'sigma', 'penalty'; jeweils inklusive Basisfall, nach Stufe sortiert."""
    out = []
    for c in cells(data):
        cfg = c["cfg"]
        others = {"capacity": ("N", "F", "sigma", "pfac"), "sigma": ("N", "Q", "F", "pfac"), "penalty": ("N", "Q", "F", "sigma")}[factor]
        if any(cfg[k] != C.BASE_CFG[k] for k in others) or any(cfg[k] != C.BASE_CFG[k] for k in ("kappa", "depot", "clustered", "tank_lo", "tank_hi")):
            continue
        out.append(c)
    key = {"capacity": lambda c: c["cfg"]["Q"], "sigma": lambda c: c["cfg"]["sigma"], "penalty": lambda c: c["cfg"]["penalty"]}[factor]
    return sorted(out, key=key)


def capacity_rows(data, H, gamma):
    """Kapazitätshebel: je Wagenkapazität Q/q, Gewinn der gewählten Regel, beste Zelle des Gitters (kreuzvalidiert), Touren und Auslastung."""
    rows = []
    for c in factor_cells(data, "capacity"):
        g = _row_gain(c, H, gamma)
        rows.append({"Q": int(c["cfg"]["Q"]), "Q_over_q": c["Q_over_qL"], "gain": g["mean"], "se": g["se"], "state": g["state"], "cv_gain": c["cv_gain"],
                     "routes_R": c["routes_R"], "routes_P": c["routes_P"], "util_R": c["util_R"], "util_P": c["util_P"], "name": c["name"]})
    return rows


def uncertainty_rows(data, H, gamma):
    """Verbrauchsschwankung und Fehlmengenstrafe: Gewinn der gewählten Regel je Stufe (die Basis gehört zu beiden Reihen)."""
    out = {}
    for factor, key in (("sigma", "sigma"), ("penalty", "penalty")):
        rows = []
        for c in factor_cells(data, factor):
            g = _row_gain(c, H, gamma)
            rows.append({"level": c["cfg"][key], "gain": g["mean"], "se": g["se"], "state": g["state"], "name": c["name"]})
        out[factor] = rows
    return out


def rule_rows(cell, H, gamma, L):
    """Regeln im Vergleich in einer Zelle: früher liefern E(1..3), reaktiv (Bezug), die gewählte P(H, gamma) und die beste Zelle (kreuzvalidiert)."""
    rows = [(f"Früher liefern, {l} Tag{'e' if l > 1 else ''}", e_gain(cell, l)["mean"], e_gain(cell, l)["se"]) for l in C.EARLY_OPTIONS]
    rows.append(("Reaktiv (Bezug)", 0.0, 0.0))
    g = p_gain(cell, H, gamma)
    rows.append((f"Bündeln P({H}; {F.fmt_level(gamma)})", g["mean"], g["se"]))
    rows.append(("Beste Zelle, kreuzvalidiert", cell["cv_gain"], None))
    return rows


def mechanism(cell):
    """Mechanismus der Standardregel P(3; 0,5) und von früher liefern E(2) gegenüber reaktiv (je 120 Tage, Mittel je Instanz), jeweils (reaktiv, bündeln, früher):
    Touren, Besuche, Auslastung, Fahr-km, Fehlmenge, Kunden-Tage mit Fehlmenge; dazu die Zahl der Mitnahmen des Bündelns."""
    return {"routes": (cell["routes_R"], cell["routes_P"], cell["routes_E"]), "visits": (cell["visits_R"], cell["visits_P"], cell["visits_E"]), "opt_visits": cell["opt_visits_P"],
            "util": (cell["util_R"], cell["util_P"], cell["util_E"]), "km": (cell["km_R"], cell["km_P"], cell["km_E"]), "short": (cell["short_R"], cell["short_P"], cell["short_E"]),
            "stockout_days": (cell["stockout_R"], cell["stockout_P"], cell["stockout_E"])}


def regime_rows(data, H, gamma):
    """Alle Zellen mit Gewinn der gewählten Regel (Mittel, SE, Median, Quartile, Anteil Verlust), Q/q und Urteil in drei Zuständen."""
    rows = []
    for c in cells(data):
        g = _row_gain(c, H, gamma)
        rows.append({"cell": c, "name": c["name"], "group": cell_group(c), "label": cell_label(c), "gain": g["mean"], "se": g["se"], "median": g["median"],
                     "q1": g["q1"], "q3": g["q3"], "share_loss": g["share_loss"], "verdict": g["verdict"], "state": g["state"],
                     "Q_over_q": c["Q_over_qL"], "short_R": c["short_R"]})
    return rows


def negative_pairs(data):
    """Alle (Zelle, Regel)-Paare der proaktiven Regeln mit im Mittel negativem Gewinn, schlimmste zuerst: [(Zelle, P-Schlüssel, Mittel, SE)]."""
    out = [(c, k, v["mean"], v["se"]) for c in cells(data) for k, v in c["P"].items() if v["mean"] < 0]
    return sorted(out, key=lambda t: t[2])


def negative_summary(data):
    """(im Mittel teurer, gesamt, davon mit gamma = 1, davon mit gamma <= 0,25)."""
    neg = negative_pairs(data)
    total = sum(len(c["P"]) for c in cells(data))
    return len(neg), total, sum(1 for _, k, _, _ in neg if k.endswith("g1.00")), sum(1 for _, k, _, _ in neg if float(k.split("_g")[1]) <= 0.25)


def shortage_rows(data):
    """Knappe Flotte: Fehlmenge reaktiv gegen Standardregel und Gewinn in den drei Flotten-Zellen."""
    rows = []
    for name in ("flotte_F1_Q300", "flotte_F1_Q250", "flotte_F2_Q150"):
        c = by_name(data, name)
        g = c["default"]
        rows.append({"cell": c, "label": cell_label(c), "gain": g["mean"], "se": g["se"], "short_R": c["short_R"], "short_P": c["short_P"], "cmp": c["short_cmp"]})
    return rows


# ----------------------------------------------------------------------------------------------------------------
# Exakter Maßstab
# ----------------------------------------------------------------------------------------------------------------
ORACLE_POLICIES = (("R", "Reaktiv"), ("P_H3_g0.50", "Bündeln P(3; 0,5)"), ("P_H8_g0.25", "Bündeln P(8; 0,25)"), ("E_L1", "Früher liefern E(1)"))


def oracle_agg(data, wagon):
    return data["oracle"][f"Q{int(wagon)}"]["agg"]


def oracle_rows(data, wagon):
    return data["oracle"][f"Q{int(wagon)}"]["rows"]


def oracle_summary(data, wagon):
    """Lücken zum Optimum je Regel (Mittel, SE, Median), Anteil der reaktiven Lücke, den P(3; 0,5) schließt, Gewinn des Optimums und mittlere Touren."""
    agg, rows = oracle_agg(data, wagon), oracle_rows(data, wagon)
    n = len(rows)
    return {"n": n, "n_optimal": agg["n_optimal"], "gap": {k: agg[f"{k}_gap_to_oracle"] for k, _ in ORACLE_POLICIES},
            "closed_P3": agg["P_H3_g0.50_share_of_R_gap_closed"], "closed_P8": agg["P_H8_g0.25_share_of_R_gap_closed"],
            "gain_P3": agg["P_H3_g0.50_gain_over_R"], "oracle_gain": agg["oracle_gain_over_R"],
            "routes": {"optimum": sum(r["oracle_routes"] for r in rows) / n, "R": sum(r["R_routes"] for r in rows) / n,
                       "P3": sum(r["P_H3_g0.50_routes"] for r in rows) / n, "P8": sum(r["P_H8_g0.25_routes"] for r in rows) / n},
            "time_max": max(r["time"] for r in rows)}

