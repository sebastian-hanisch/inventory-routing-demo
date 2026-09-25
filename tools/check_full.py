"""Das VOLLE Bau-Gate: alle Korrektheits-Checks der Messreihe (messreihe_inventory_routing/check.py und check_oracle.py) in voller Größe mit den irp_-Modulen und dem echten OR-Tools, dazu die
bitgleiche Wiederholung des GANZEN Sweeps gegen raw_sweep.npz der Messreihe und die Wiederholung der Kleininstanzen gegen die Orakel-Läufe. Läuft einmal lokal vor dem Commit (etwa 4 Minuten),
nicht in der CI: dort laufen dieselben Prüfungen verkleinert und auf eingefrorenen Instanzen (tests/test_checks.py, tests/test_frozen_reference.py).

Die Zahlen des vollen Laufs (Zählungen mit Zufallsinstanzen) gelten für die NumPy-Version, mit der die Messreihe gerechnet wurde; mit einer anderen NumPy-Version können sich die Zufallsströme
und damit diese Zählungen ändern (die Invarianten nicht). Der Lauf meldet die Version und prüft die Zählungen nur, wenn sie übereinstimmt.

Aufruf (im Projektordner):  python tools/check_full.py [Ordner_der_Messreihe]   (Standard ../bestand-planung/messreihe_inventory_routing)"""
import json
import multiprocessing as mp
import pathlib
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "tests", ROOT / "tools"):
    sys.path.insert(0, str(p))

import irp_checks as K  # noqa: E402
import irp_model as M  # noqa: E402
import irp_policy as P  # noqa: E402
import sweep as SW  # noqa: E402

DEFAULT_SOURCES = ROOT.parent / "bestand-planung" / "messreihe_inventory_routing"
MEASURED_WITH_NUMPY = "2.5.3"


def say(text):
    print(text, flush=True)


def sweep_chunk(task):
    return SW.run_chunk(task)


def main(argv):
    src = pathlib.Path(argv[0]) if argv else DEFAULT_SOURCES
    t0 = time.time()
    stream_ok = np.__version__ == MEASURED_WITH_NUMPY
    say(f"NumPy {np.__version__} ({'wie bei der Messreihe: die Zählungen werden geprüft' if stream_ok else 'anders als bei der Messreihe: nur Invarianten'})")

    assert K.check_no_consumption(12)
    say("1. Grenzfall ohne Verbrauch: keine Touren, keine Kosten, kein Bestandsverlauf (reaktiv, H = 3, H = 8).")
    n2 = K.check_h0_equals_reactive(60, 40)
    assert n2 == 240
    say(f"2. H = 0 == reaktiv: bitgleiche Tourpläne für gamma 0,1 / 0,5 / 1 / 5 in {n2} Instanzen (4 Konfigurationen).")
    n3 = K.check_balance(40, 60)
    assert n3 == 800
    say(f"3. Unabhängige Neubewertung == Simulation und exakte Bestandsbilanz: {n3} Läufe.")
    tot = K.check_hand_instance()
    say(f"4. Handinstanz: 14 / 24 / {tot:.1f} (Optimum von Hand).")
    mean, median, p90, worst, share = K.check_savings_vs_brute_force(120)
    say(f"5. Savings + 2-opt gegen exaktes CVRP (3-6 Kunden, 120 Instanzen): nie besser als das Optimum; Lücke Mittel {mean:.2f} %, Median {median:.2f} %, P90 {p90:.2f} %, Max {worst:.2f} %, exakt optimal in {share:.0f} %.")
    if stream_ok:
        assert (round(mean, 2), round(median, 2), round(p90, 2), round(worst, 2), round(share)) == (0.12, 0.0, 0.0, 6.13, 93), (mean, median, p90, worst, share)

    out = K.check_branches(n_seeds=1, days=90)                       # nur Seed 0: die Zahlen der Messreihe (check.py Test 6)
    sh1 = sum(P.simulate(M.make_instance(s, sigma=1.0), 0.0)["short"] for s in range(20))
    sh0 = sum(P.simulate(M.make_instance(s, sigma=0.0), 0.0)["short"] for s in range(20))
    say(f"6. Zweig-Tests (Seed 0, 90 Tage): Mitnahmen {out['picks_P']} (reaktiv {out['picks_R']}); gamma steuert sie ({out['gamma_low']} bei 0,1 gegen {out['gamma_high']} bei 1,0); früh liefern: "
        f"{out['visits_E']} gegen {out['visits_R']} Besuche ohne Mitnahmen; enge Flotte (F = 1, N = 30): zurückgestellt {out['deferred_tight']}, Fehlmenge {out['short_tight']:.0f}; "
        f"sigma = 1: Fehlmenge {sh1:.0f} in 20 Instanzen, sigma = 0: {sh0:.0f} (korrekt: die Meldegrenze deckt den Tagesverbrauch).")
    assert out["deferred_free"] == 0 and sh0 == 0.0 and sh1 > 0
    if stream_ok:
        assert (out["gamma_low"], out["gamma_high"], out["visits_E"], out["visits_R"], out["deferred_tight"], round(out["short_tight"]), round(sh1)) == (29, 97, 217, 171, 833, 7668, 5606)

    rows = K.check_shortage_share(n_seeds=60, days=90)
    say("7. Fehlmengen proaktiv (H = 3, gamma = 0,5) > reaktiv (60 Instanzen je Konfiguration): " + "; ".join(f"{a}: {b}/{c}" for a, b, c in rows))
    if stream_ok:
        assert [r[1] for r in rows] == [16, 0, 10, 19, 0], rows
    assert K.check_determinism()
    say("8. Determinismus: gleiche Seeds -> gleiche Instanz, gleicher Verbrauchsstrom, gleiche Kosten.")
    n9 = K.check_oracle_vs_dp(seeds=range(24), min_feasible=15)
    say(f"9. CP-SAT-Modell == unabhängige exakte Rechnung (DP über Bestandszustände) auf {n9} von 24 Ganzzahl-Instanzen (3 Kunden, 4 Tage, Q = 6): Zielwerte gleich, Plan im Simulator nachgespielt == Modellzielwert.")

    if (src / "raw_sweep.npz").exists():
        raw = np.load(src / "raw_sweep.npz")
        meta = json.loads((src / "sweep_meta.json").read_text(encoding="utf-8"))
        assert list(meta["configs"]) == list(SW.CONFIGS) and {k: list(v) for k, v in SW.POLICIES.items()} == meta["policies"] and meta["metrics"] == SW.METRICS
        tasks = [(c, lo, min(lo + SW.CHUNK, SW.N_INST)) for c in SW.CONFIGS for lo in range(0, SW.N_INST, SW.CHUNK)]
        bad = 0
        n_fields = 0
        with mp.Pool(min(12, mp.cpu_count())) as pool:
            for cname, lo, hi, res, pen in pool.imap_unordered(sweep_chunk, tasks):
                for pol, arr in res.items():
                    ref = raw[f"{cname}|{pol}"][:, lo:hi]
                    n_fields += arr.size
                    if not np.array_equal(arr, ref):
                        bad += 1
        say(f"10. Wiederholung des ganzen Sweeps mit den irp_-Modulen (27 Konfigurationen x 32 Regeln x 200 Instanzen x 12 Kennzahlen = {n_fields} Werte) gegen raw_sweep.npz: "
            f"{bad} Blöcke abweichend.")
        assert bad == 0
    else:
        say("10. Wiederholung des Sweeps: übersprungen (raw_sweep.npz der Messreihe nicht gefunden).")

    import irp_oracle as OR
    import irp_policy as PL
    worst_diff = 0.0
    n11 = 0
    for wagon in (150, 250):
        path = src / f"oracle_data_Q{wagon}.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["rows"]:
            inst = OR.make_oracle_instance(row["seed"], Q=float(wagon))
            res = OR.solve_oracle(inst, OR.TAU, time_limit=120, workers=8, seed=1)
            rep = PL.replay(inst, res["plans"])
            j = PL.objective(rep, inst.p, OR.TAU)
            worst_diff = max(worst_diff, abs(j - row["J_oracle"]))
            for name, (H, g, e) in {"R": (0.0, 0.5, 0.0), "P_H3_g0.50": (3.0, 0.5, 0.0), "P_H8_g0.25": (8.0, 0.25, 0.0), "E_L1": (0.0, 0.5, 1.0)}.items():
                worst_diff = max(worst_diff, abs(PL.objective(PL.simulate(inst, H, g, early=e), inst.p, OR.TAU) - row[name]))
            assert res["status"] == "OPTIMAL" == row["status"]
            n11 += 1
    if n11:
        say(f"11. Kleininstanzen (7 Kunden, 6 Tage): {n11} Instanzen mit CP-SAT (8 Arbeiter) neu gelöst, alle bewiesen optimal; Zielwerte von Optimum und vier Regeln gleich denen der Messreihe (größte Abweichung {worst_diff:.1e}).")
        assert worst_diff < 1e-6
    else:
        say("11. Kleininstanzen: übersprungen (Orakel-Läufe der Messreihe nicht gefunden).")
    say(f"\nalle Checks bestanden ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    mp.freeze_support()
    sys.exit(main(sys.argv[1:]))
