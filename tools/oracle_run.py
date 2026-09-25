"""Exakter Maßstab auf Kleininstanzen (N = 7 Kunden, 6 Tage, eine Tour je Tag, deterministischer Verbrauch): R, P, E gegen das CP-SAT-Optimum (aus
messreihe_inventory_routing/oracle_run.py, Rechnung unverändert; jetzt in Funktionen und mit irp_oracle / irp_policy).

Aufruf (im Projektordner):  python tools/oracle_run.py [Instanzen] [Zeitlimit_s] [Wagen] [Ausgabeordner]
  Standard 20 Instanzen, 120 s, Wagen 150, Ausgabe tools/_out -> oracle_data_Q<Wagen>.json, oracle_report_Q<Wagen>.txt (die Messreihe rechnete 30 Instanzen).
Rechenzeit: unter 2 Minuten je Wagen (jede Instanz 0 bis 2 s bis zum bewiesenen Optimum); nie in der CI."""
import json
import pathlib
import sys
import time

import numpy as np

sys.dont_write_bytecode = True
ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import irp_oracle  # noqa: E402
import irp_policy  # noqa: E402

TAU = 0.6
POL = {"R": (0.0, 0.5, 0.0), "P_H3_g0.50": (3.0, 0.5, 0.0), "P_H8_g0.25": (8.0, 0.25, 0.0), "E_L1": (0.0, 0.5, 1.0)}


def solve_row(seed, wagon, time_limit, workers=8):
    """Eine Kleininstanz: Optimum, Nachspielen, die vier Regeln. Rückgabe die Zeile der Messreihe (oder {'seed', 'status': 'NONE'})."""
    inst = irp_oracle.make_oracle_instance(seed, Q=float(wagon))
    res = irp_oracle.solve_oracle(inst, TAU, time_limit=time_limit, workers=workers, seed=1)
    if res is None:
        return {"seed": seed, "status": "NONE"}
    rep = irp_policy.replay(inst, res["plans"])
    J_or_replay = irp_policy.objective(rep, inst.p, TAU)
    J_or_model = res["obj"] + TAU * rep["I_start"]
    row = {"seed": seed, "status": res["status"], "time": res["time"], "J_oracle": J_or_replay, "J_oracle_model": J_or_model,
           "bound_J": res["bound"] + TAU * rep["I_start"], "oracle_short": rep["short"], "oracle_routes": rep["n_routes"],
           "oracle_routing": rep["routing"], "oracle_visits": rep["n_visits"]}
    assert abs(J_or_replay - J_or_model) < 1e-6, ("Modell und Replay weichen ab", J_or_replay, J_or_model)
    assert rep["short"] == 0.0
    for name, (H, g, e) in POL.items():
        r = irp_policy.simulate(inst, H, g, early=e)
        row[name] = irp_policy.objective(r, inst.p, TAU)
        row[name + "_routes"] = r["n_routes"]
        row[name + "_short"] = r["short"]
        row[name + "_routing"] = r["routing"]
        if r["short"] == 0.0:
            assert row[name] >= row["J_oracle"] - 1e-6, ("Regel ohne Fehlmenge besser als das bewiesene Optimum", name, row[name], row["J_oracle"])
        row[name + "_beats_oracle_with_shortage"] = bool(r["short"] > 0 and row[name] < row["J_oracle"] - 1e-6)
    return row


def aggregate(rows, wagon, time_limit, total_time=0.0, workers=8):
    """Kennzahlen aller Zeilen: Lücken zum Optimum je Regel, Gewinn gegenüber R und der Anteil der Lücke von R, den P schließt. Rückgabe (agg, report_lines)."""
    out = []
    ok = [r for r in rows if r["status"] in ("OPTIMAL", "FEASIBLE")]
    opt = [r for r in ok if r["status"] == "OPTIMAL"]
    agg = {"n": len(ok), "n_optimal": len(opt), "tau": TAU, "total_time": total_time}
    out.append("Exakter Maszstab: %d Instanzen (N=7, 6 Tage, Q=%d, 1 Tour/Tag, sigma=0), davon %d bewiesen optimal (Zeitlimit %.0f s, %d Arbeiter); tau=%.1f" % (
        len(ok), wagon, len(opt), time_limit, workers, TAU))
    for name in list(POL) + ["J_oracle"]:
        v = np.array([r[name] for r in ok])
        agg[name] = float(v.mean())
    JO = np.array([r["J_oracle"] for r in ok])
    for name in POL:
        v = np.array([r[name] for r in ok])
        gap = 100 * (v - JO) / JO
        agg[name + "_gap_to_oracle"] = {"mean": float(gap.mean()), "se": float(gap.std(ddof=1) / np.sqrt(len(gap))), "median": float(np.median(gap)),
                                        "q1": float(np.percentile(gap, 25)), "q3": float(np.percentile(gap, 75)), "min": float(gap.min()), "max": float(gap.max())}
        nb = sum(1 for r in ok if r[name + "_beats_oracle_with_shortage"])
        out.append("%-12s J %8.1f   Luecke zum Optimum: Mittel %5.1f +- %3.1f %%  Median %5.1f  [Q1 %5.1f .. Q3 %5.1f]  Min %5.1f  Max %5.1f   Fehlmenge>0 in %d Instanzen (davon %d mit J unter dem Optimum, weil das Optimum keine Fehlmenge erlaubt)" % (
            name, v.mean(), gap.mean(), gap.std(ddof=1) / np.sqrt(len(gap)), np.median(gap), np.percentile(gap, 25), np.percentile(gap, 75), gap.min(), gap.max(),
            sum(1 for r in ok if r[name + "_short"] > 1e-9), nb))
    JR = np.array([r["R"] for r in ok])
    for name in ("P_H3_g0.50", "P_H8_g0.25"):
        v = np.array([r[name] for r in ok])
        gain = 100 * (JR - v) / JR
        closed = (JR - v) / np.where(JR - JO > 1e-9, JR - JO, np.nan)
        agg[name + "_gain_over_R"] = {"mean": float(gain.mean()), "se": float(gain.std(ddof=1) / np.sqrt(len(gain))), "median": float(np.median(gain))}
        agg[name + "_share_of_R_gap_closed"] = float(np.nanmean(closed))
        out.append("%-12s Gewinn gegenueber R: %5.1f +- %3.1f %% (Median %5.1f); schliesst im Mittel %.0f %% des Rueckstands von R zum Optimum" % (
            name, gain.mean(), gain.std(ddof=1) / np.sqrt(len(gain)), np.median(gain), 100 * np.nanmean(closed)))
    gapO = 100 * (JR - JO) / JR
    agg["oracle_gain_over_R"] = {"mean": float(gapO.mean()), "se": float(gapO.std(ddof=1) / np.sqrt(len(gapO))), "median": float(np.median(gapO))}
    out.append("Optimum gegenueber R: Gewinn %5.1f +- %3.1f %% (Median %5.1f)" % (gapO.mean(), gapO.std(ddof=1) / np.sqrt(len(gapO)), np.median(gapO)))
    out.append("Optimum: mittlere Touren %.2f, Besuche %.2f; R: Touren %.2f; P_H3: Touren %.2f; P_H8: Touren %.2f" % (
        np.mean([r["oracle_routes"] for r in ok]), np.mean([r["oracle_visits"] for r in ok]), np.mean([r["R_routes"] for r in ok]),
        np.mean([r["P_H3_g0.50_routes"] for r in ok]), np.mean([r["P_H8_g0.25_routes"] for r in ok])))
    return agg, out


def main(argv):
    n_inst = int(argv[0]) if len(argv) > 0 else 20
    time_limit = float(argv[1]) if len(argv) > 1 else 120.0
    wagon = int(float(argv[2])) if len(argv) > 2 else 150
    out_dir = pathlib.Path(argv[3]) if len(argv) > 3 else ROOT / "tools" / "_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for s in range(n_inst):
        row = solve_row(s, wagon, time_limit)
        rows.append(row)
        if row["status"] != "NONE":
            print(s, row["status"], "%.0fs" % row["time"], "oracle %.1f  R %.1f  P3 %.1f  P8 %.1f" % (row["J_oracle"], row["R"], row["P_H3_g0.50"], row["P_H8_g0.25"]), flush=True)
    agg, out = aggregate(rows, wagon, time_limit, time.time() - t0)
    tag = "Q%d" % wagon
    (out_dir / f"oracle_data_{tag}.json").write_text(json.dumps({"agg": agg, "rows": rows}, indent=1), encoding="utf-8")
    (out_dir / f"oracle_report_{tag}.txt").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
