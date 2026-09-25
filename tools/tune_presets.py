"""Preset-Abstimmung: sucht die Anzeige-Seeds der Beispielszenarien. Der Seed bestimmt nur die GEZEIGTE Instanz (die Messreihe steht auf den Seeds 0 bis 199),
deshalb wird nur außerhalb der Stichprobe gesucht (200 bis 299). Für jeden Seed werden die fünf Presets auf der echten Instanz gerechnet und die qualitativen
Kriterien aus irp_stories.day_criteria geprüft; ausgegeben werden die Seeds, die alle fünf erfüllen, mit dem Gewinn von Bündeln im Standard (typisch = nah am
Mittel der Messreihe).

Aufruf (im Projektordner):  python tools/tune_presets.py [von bis]   (Standard 200 300)"""
import pathlib
import sys

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import irp_constants as C  # noqa: E402
import irp_live as LV  # noqa: E402
import irp_results as R  # noqa: E402
import irp_stories as ST  # noqa: E402


def solve_preset(p, seed, gamma=None):
    return LV.solve_live(p["customers"], p["capacity"], p["fleet"], p["sigma"], p["penalty"], p["horizon"], p["gamma"] if gamma is None else gamma, p["early"], seed)


def evaluate(seed, presets=None):
    """Ergebnis je Preset auf der Instanz `seed`: (alle Kriterien erfüllt, Liste der Texte, Gewinn von Bündeln in %)."""
    out = {}
    for name, p in (presets or C.PRESETS).items():
        live = solve_preset(p, seed)
        alt = solve_preset(p, seed, gamma=ST.GENEROUS_GAMMA_OK) if name == "Zu großzügig" else None
        crit = ST.day_criteria(name, live, alt)
        out[name] = (all(ok for ok, _ in crit), crit, live["rules"]["P"]["gain"])
    return out


def main(argv):
    lo, hi = (int(argv[0]), int(argv[1])) if len(argv) >= 2 else (200, 300)
    data = R.load_results()
    mean_gain = R.p_gain(R.find_cell(data), C.DEFAULT_H, C.DEFAULT_GAMMA)["mean"]
    good = []
    for seed in range(lo, hi):
        res = evaluate(seed)
        flags = "".join("+" if ok else "-" for ok, _, _ in res.values())
        std = res["Standard"][2]
        print(f"Seed {seed}: {flags}  Gewinn Bündeln Standard {std:+.1f} % (Messreihe Mittel {mean_gain:+.1f} %)", flush=True)
        if all(ok for ok, _, _ in res.values()):
            good.append((abs(std - mean_gain), seed))
    good.sort()
    print("\nSeeds, die alle fünf Tageskriterien erfüllen (nach Nähe des Gewinns zum Mittel der Messreihe):", [s for _, s in good])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
