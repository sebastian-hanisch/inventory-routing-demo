"""Auswertung von raw_sweep.npz -> sweep_data.json, sweep_report.txt (aus messreihe_inventory_routing/analyze.py; die Rechnung ist unverändert,
neu ist nur die Zerlegung in Funktionen, die Ein-/Ausgabeordner als Parameter und je Konfiguration der Vergleich der Fehlmengen `short_cmp`).

Zielgröße J = Fahrstrecke + p*Fehlmenge - tau*(Endbestand - Anfangsbestand); tau = Fahrkosten je gelieferter Einheit der reaktiven Regel in derselben
Konfiguration (Batchmittel). Gewinn = 100*(J_reaktiv - J_Regel)/J_reaktiv (positiv = Regel billiger).

Aufruf (im Projektordner):  python tools/analyze.py [Ordner mit raw_sweep.npz und sweep_meta.json]  (Standard tools/_out; schreibt dorthin)"""
import json
import math
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_POLICY = "P_H3_g0.50"
EARLY_POLICY = "E_L2"


def pct_gain(JR, JP):
    """Gewinn in % der mittleren Reaktivkosten: Mittel +- SE der gepaarten Differenz, Median der Einzelgewinne (% je Instanz), Quartile, Anteil Verlust."""
    diff = JR - JP
    m = JR.mean()
    n = len(diff)
    se = diff.std(ddof=1) / math.sqrt(n)
    per = 100 * diff / JR
    return {"mean": 100 * diff.mean() / m, "se": 100 * se / m, "median": float(np.median(per)), "q1": float(np.percentile(per, 25)),
            "q3": float(np.percentile(per, 75)), "share_loss": float((diff < -1e-9).mean())}


def ranks(v):
    o = np.argsort(v)
    r = np.empty(len(v))
    r[o] = np.arange(len(v))
    return r


def analyze(meta, raw):
    """Aggregate aller Konfigurationen. Rückgabe (data, report_lines): data wie sweep_data.json der Messreihe, dazu je Konfiguration `short_cmp`."""
    M = {k: i for i, k in enumerate(meta["metrics"])}
    POL = list(meta["policies"])
    CONF = list(meta["configs"])
    out = []
    data = {}

    def p(s=""):
        out.append(s)

    def get(c, pol, key):
        return raw["%s|%s" % (c, pol)][M[key]]

    def J_arr(c, pol, tau, pen):
        return get(c, pol, "routing") + pen * get(c, pol, "short") - tau * get(c, pol, "dI")

    PPOL = [q for q in POL if q.startswith("P_")]
    EPOL = [q for q in POL if q.startswith("E_")]

    p("Zielgroesse J = Fahrstrecke + p*Fehlmenge - tau*(Endbestand-Anfangsbestand); Gewinn in % der mittleren reaktiven J (positiv = billiger als reaktiv)")
    p("%d Instanzen je Konfiguration, %d Tage, gepaart (derselbe Verbrauchsstrom fuer alle Regeln)" % (meta["n_inst"], meta["days"]))
    p("")
    summary = {}
    for c in CONF:
        pen = meta["penalty"][c]
        routingR = get(c, "R", "routing")
        tau = routingR.sum() / get(c, "R", "delivered").sum()
        JR = J_arr(c, "R", tau, pen)
        gains = {q: pct_gain(JR, J_arr(c, q, tau, pen)) for q in POL if q != "R"}
        # beste Regel mit Kreuzvalidierung (Waehlen auf Haelfte A, Bewerten auf B und umgekehrt)
        n = len(JR)
        A = np.arange(n) % 2 == 0
        B = ~A
        cv = []
        chosen = []
        for tr, te in ((A, B), (B, A)):
            cands = {q: (JR[tr] - J_arr(c, q, tau, pen)[tr]).mean() for q in PPOL}
            bestq = max(cands, key=cands.get)
            if cands[bestq] < 0:
                bestq, bg = "R", np.zeros(te.sum())
            else:
                bg = (JR[te] - J_arr(c, bestq, tau, pen)[te])
            chosen.append(bestq)
            cv.append(100 * bg.sum() / JR[te].sum())
        cv_gain = float(np.mean(cv))
        insample = max(PPOL, key=lambda q: gains[q]["mean"])
        delivR = get(c, "R", "delivered").sum()
        visR = get(c, "R", "n_visits").sum()
        qL = delivR / visR                       # mittlere Liefermenge je Besuch (reaktiv)
        Q = meta["configs"][c].get("Q", 300.0)
        sR = get(c, "R", "short")
        sP = get(c, DEFAULT_POLICY, "short")
        row = {"tau": tau, "JR": float(JR.mean()), "Q_over_qL": Q / qL, "qL": qL, "short_R": float(get(c, "R", "short").mean()),
               "default": gains[DEFAULT_POLICY], "cv_gain": cv_gain, "cv_chosen": chosen, "best_insample": insample, "best_insample_gain": gains[insample],
               "E": {q: gains[q] for q in EPOL}, "P": {q: gains[q] for q in PPOL},
               "routes_R": float(get(c, "R", "n_routes").mean()), "routes_P": float(get(c, DEFAULT_POLICY, "n_routes").mean()),
               "visits_R": float(get(c, "R", "n_visits").mean()), "visits_P": float(get(c, DEFAULT_POLICY, "n_visits").mean()),
               "opt_visits_P": float(get(c, DEFAULT_POLICY, "opt_visits").mean()),
               "short_P": float(get(c, DEFAULT_POLICY, "short").mean()), "stockout_R": float(get(c, "R", "stockout_days").mean()),
               "stockout_P": float(get(c, DEFAULT_POLICY, "stockout_days").mean()), "deferred_R": float(get(c, "R", "deferred").mean()),
               "util_R": float(get(c, "R", "util_sum").sum() / max(get(c, "R", "n_routes").sum(), 1)),
               "util_P": float(get(c, DEFAULT_POLICY, "util_sum").sum() / max(get(c, DEFAULT_POLICY, "n_routes").sum(), 1)),
               "km_R": float(routingR.mean()), "km_P": float(get(c, DEFAULT_POLICY, "routing").mean()),
               "short_cmp": {"more": float((sP > sR + 1e-9).mean()), "less": float((sP < sR - 1e-9).mean()), "equal": float((abs(sP - sR) <= 1e-9).mean())},
               # Mechanismus von "früher liefern" E(2) (Gegenprobe zu bündeln); in der Messreihe nur im Bericht (sweep_report Zusatz 2), hier je Konfiguration
               "routes_E": float(get(c, EARLY_POLICY, "n_routes").mean()), "visits_E": float(get(c, EARLY_POLICY, "n_visits").mean()), "km_E": float(get(c, EARLY_POLICY, "routing").mean()),
               "util_E": float(get(c, EARLY_POLICY, "util_sum").sum() / max(get(c, EARLY_POLICY, "n_routes").sum(), 1)), "short_E": float(get(c, EARLY_POLICY, "short").mean()),
               "stockout_E": float(get(c, EARLY_POLICY, "stockout_days").mean())}
        summary[c] = row
        d = row["default"]
        e = row["E"]["E_L2"]
        p("%-16s tau %6.3f J_R %6.0f Q/q %5.2f kurzR %6.1f | Standard %+6.1f +-%3.1f (%2.0f%% Verlust) | beste (kreuzvalid.) %+6.1f | E_L2 %+6.1f +-%3.1f | Touren R->P %5.1f -> %5.1f | beste (H,g) %s" % (
            c, tau, JR.mean(), row["Q_over_qL"], row["short_R"], d["mean"], d["se"], 100 * d["share_loss"], cv_gain,
            e["mean"], e["se"], row["routes_R"], row["routes_P"], insample.replace("P_", "")))
    data["configs"] = summary

    # Rangkorrelation Gewinn ~ Q/qL ueber alle Konfigurationen ohne Massenfehlmengen
    cs = [c for c in CONF if summary[c]["short_R"] < 200]
    x = np.array([summary[c]["Q_over_qL"] for c in cs])
    y = np.array([summary[c]["default"]["mean"] for c in cs])
    rho = float(np.corrcoef(ranks(x), ranks(y))[0, 1])
    p("")
    p("Rangkorrelation Gewinn (Standardregel) gegen Q/mittlere Liefermenge ueber %d Konfigurationen (ohne Massenfehlmengen): rho = %.2f" % (len(cs), rho))
    data["spearman_gain_vs_Q_over_qL"] = {"rho": rho, "n": len(cs)}

    # Verteilung in der Basis
    c = "base"
    pen = meta["penalty"][c]
    tau = summary[c]["tau"]
    JR = J_arr(c, "R", tau, pen)
    JP = J_arr(c, DEFAULT_POLICY, tau, pen)
    per = 100 * (JR - JP) / JR
    p("Basis: Verteilung der Einzelgewinne (Standardregel, %% je Instanz): Min %.1f, Q1 %.1f, Median %.1f, Q3 %.1f, P90 %.1f, Max %.1f; Instanzen mit Verlust %.1f %%" % (
        per.min(), np.percentile(per, 25), np.median(per), np.percentile(per, 75), np.percentile(per, 90), per.max(), 100 * (per < 0).mean()))
    data["base_distribution"] = {"min": float(per.min()), "q1": float(np.percentile(per, 25)), "median": float(np.median(per)), "q3": float(np.percentile(per, 75)),
                                 "p90": float(np.percentile(per, 90)), "max": float(per.max()), "loss_share": float((per < 0).mean())}
    neg = []
    for c in CONF:
        for q in PPOL:
            g = summary[c]["P"][q]
            if g["mean"] < 0:
                neg.append((c, q, g["mean"], g["se"]))
    neg.sort(key=lambda z: z[2])
    for c, q, m_, s_ in neg[:25]:
        p("   %-16s %-12s %+5.1f +- %3.1f" % (c, q, m_, s_))
    p("   insgesamt %d von %d Regel-Konfiguration-Paaren im Mittel schlechter als reaktiv" % (len(neg), len(PPOL) * len(CONF)))
    data["n_negative_pairs"] = [len(neg), len(PPOL) * len(CONF)]
    en = sum(1 for c in CONF for q in EPOL if summary[c]["E"][q]["mean"] < 0)
    p("Regeln E (frueher liefern, ohne Tourenbezug): %d von %d Paaren mit negativem Gewinn" % (en, len(CONF) * len(EPOL)))
    data["E_negative_pairs"] = [en, len(CONF) * len(EPOL)]
    return data, out


def main(argv):
    folder = pathlib.Path(argv[0]) if argv else ROOT / "tools" / "_out"
    meta = json.loads((folder / "sweep_meta.json").read_text(encoding="utf-8"))
    raw = np.load(folder / "raw_sweep.npz")
    data, out = analyze(meta, raw)
    (folder / "sweep_data.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    (folder / "sweep_report.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
