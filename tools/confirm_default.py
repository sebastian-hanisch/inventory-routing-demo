"""Bestätigung der Standardregel P(3; 0,5) auf frischen Instanzen (AP 0 des Detailplans): die Wahl von H = 3 und gamma = 0,5 erfolgte nach einer Erkundung auf den Seeds 0 bis 59,
die im Sweep (Seeds 0 bis 199) enthalten sind. Hier läuft dieselbe Auswertung (Gewinn gegenüber reaktiv in % der mittleren reaktiven Kosten, Mittel ± Standardfehler der gepaarten
Differenz, Median, Quartile, Anteil Verlust) auf den Seeds 200 bis 299, die weder bei der Wahl noch im Sweep vorkamen. tau wie in der Messreihe: Fahrkosten je gelieferter Einheit
der reaktiven Regel, hier aus den 100 Instanzen (mit dem tau der Messreihe ist das Ergebnis dasselbe).

Aufruf (im Projektordner):  python tools/confirm_default.py [Konfiguration ...]   (Standard: base q100 q600 flotte_F1_Q300)"""
import math
import pathlib
import sys

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import irp_model  # noqa: E402
import irp_policy  # noqa: E402
import irp_results as R  # noqa: E402

CONFIGS = {"base": {}, "q100": {"Q": 100.0}, "q600": {"Q": 600.0}, "flotte_F1_Q300": {"F": 1, "Q": 300.0}}
SEEDS = range(200, 300)


def confirm(overrides, seeds=SEEDS, H=3.0, gamma=0.5, days=120, tau=None):
    """Gewinn von P(H, gamma) gegenüber reaktiv auf den Instanzen `seeds` der Konfiguration `overrides`. Rückgabe dict mit tau, mean, se, median, q1, q3, min, max, share_loss, n."""
    pfac = overrides.get("pfac", 0.05)
    pen = pfac * 100.0
    reactive, bundling = [], []
    for s in seeds:
        inst = irp_model.make_instance(s, D=days, **overrides)
        reactive.append(irp_policy.simulate(inst, 0.0, 0.5))
        bundling.append(irp_policy.simulate(inst, H, gamma))
    if tau is None:
        tau = sum(r["routing"] for r in reactive) / sum(r["delivered"] for r in reactive)
    cost = lambda rs: np.array([r["routing"] + pen * r["short"] - tau * r["dI"] for r in rs])
    jr, jp = cost(reactive), cost(bundling)
    diff = jr - jp
    per = 100 * diff / jr
    n = len(diff)
    return {"tau": float(tau), "mean": float(100 * diff.mean() / jr.mean()), "se": float(100 * diff.std(ddof=1) / math.sqrt(n) / jr.mean()), "median": float(np.median(per)),
            "q1": float(np.percentile(per, 25)), "q3": float(np.percentile(per, 75)), "min": float(per.min()), "max": float(per.max()), "share_loss": float((diff < -1e-9).mean()), "n": n}


def main(argv):
    names = argv or list(CONFIGS)
    data = R.load_results()
    for name in names:
        cell = R.by_name(data, name)
        sweep = R.p_gain(cell, 3, 0.5)
        new = confirm(CONFIGS[name])
        diff = new["mean"] - sweep["mean"]
        se_diff = math.hypot(sweep["se"], new["se"])
        print(f"{name}: Sweep (Seeds 0-199) {sweep['mean']:+.2f} +- {sweep['se']:.2f} %, Median {sweep['median']:.2f}, Quartile [{sweep['q1']:.2f}; {sweep['q3']:.2f}], Verlust {100 * sweep['share_loss']:.0f} %")
        print(f"   frisch (Seeds 200-299, tau {new['tau']:.3f}): {new['mean']:+.2f} +- {new['se']:.2f} %, Median {new['median']:.2f}, Quartile [{new['q1']:.2f}; {new['q3']:.2f}], "
              f"Min {new['min']:.2f}, Max {new['max']:.2f}, Verlust {100 * new['share_loss']:.0f} %; Differenz der Mittel {diff:+.2f} ({diff / se_diff:+.1f} SE der Differenz)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
