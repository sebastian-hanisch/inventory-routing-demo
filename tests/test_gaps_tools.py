"""Tests, die im ersten vollständigen Fehler-Einbau-Lauf überlebende Mutanten der Werkzeuge schließen (analyze, build_results, oracle_run, confirm_default): die Kennzahlen gegen eine
unabhängige Rechnung mit NumPy, Randfälle der Toleranzen, Berichtszeilen, Kommandozeilen-Einstiegspunkte."""
import json
import math
import pathlib
import subprocess
import sys

import numpy as np
import pytest

import irp_model as M
import irp_policy as P
import irp_results as R
from test_tools import AN, BR, CD, OQ, METRICS, synthetic_rows, synthetic_sweep, synthetic_cell

ROOT = pathlib.Path(__file__).resolve().parent.parent


def J(raw, c, pol, tau, pen=5.0):
    a = raw[f"{c}|{pol}"]
    return a[0] + pen * a[1] - tau * a[2]


# ---------------------------------------------------------------------------------------------------
# analyze
# ---------------------------------------------------------------------------------------------------
def test_analyze_base_distribution_and_spearman_against_an_independent_computation():
    meta, raw = synthetic_sweep(n=30, seed=7)
    data, report = AN.analyze(meta, raw)
    tau = raw["base|R"][0].sum() / raw["base|R"][3].sum()
    jr, jp = J(raw, "base", "R", tau), J(raw, "base", "P_H3_g0.50", tau)
    per = 100 * (jr - jp) / jr
    d = data["base_distribution"]
    for key, want in (("min", per.min()), ("q1", np.percentile(per, 25)), ("median", np.median(per)), ("q3", np.percentile(per, 75)), ("p90", np.percentile(per, 90)), ("max", per.max())):
        assert d[key] == pytest.approx(want, rel=1e-12), key
    assert d["loss_share"] == float((per < 0).mean())
    xs, ys = [], []
    for c in ("base", "q100", "q600"):
        t = raw[f"{c}|R"][0].sum() / raw[f"{c}|R"][3].sum()
        Q = meta["configs"][c].get("Q", 300.0)
        xs.append(Q / (raw[f"{c}|R"][3].sum() / raw[f"{c}|R"][6].sum()))
        a, b = J(raw, c, "R", t), J(raw, c, "P_H3_g0.50", t)
        ys.append(100 * (a - b).mean() / a.mean())
    rx, ry = np.argsort(np.argsort(xs)).astype(float), np.argsort(np.argsort(ys)).astype(float)
    assert data["spearman_gain_vs_Q_over_qL"]["rho"] == pytest.approx(float(np.corrcoef(rx, ry)[0, 1]))
    line = next(x for x in report if x.startswith("Basis: Verteilung"))
    assert line == ("Basis: Verteilung der Einzelgewinne (Standardregel, %% je Instanz): Min %.1f, Q1 %.1f, Median %.1f, Q3 %.1f, P90 %.1f, Max %.1f; Instanzen mit Verlust %.1f %%"
                    % (per.min(), np.percentile(per, 25), np.median(per), np.percentile(per, 75), np.percentile(per, 90), per.max(), 100 * (per < 0).mean()))


def test_analyze_treats_zero_gain_and_rounding_noise_as_no_loss():
    meta, raw = synthetic_sweep(n=20, seed=2)
    raw["base|P_H3_g0.50"][:, 0] = raw["base|R"][:, 0]                                       # Instanz 0: Regel = reaktiv, Gewinn genau 0
    raw["base|P_H3_g0.50"][0, 2] = raw["base|R"][0, 2] + 5e-10                                # Instanz 2: um 5e-10 teurer (unter der Toleranz 1e-9)
    data, _ = AN.analyze(meta, raw)
    tau = raw["base|R"][0].sum() / raw["base|R"][3].sum()
    jr, jp = J(raw, "base", "R", tau), J(raw, "base", "P_H3_g0.50", tau)
    assert data["base_distribution"]["loss_share"] == float(((100 * (jr - jp) / jr) < 0).mean())
    assert data["configs"]["base"]["default"]["share_loss"] == float(((jr - jp) < -1e-9).mean())
    assert data["configs"]["base"]["default"]["share_loss"] <= data["base_distribution"]["loss_share"]              # die Toleranz 1e-9 zählt das Rauschen nicht als Verlust


def test_analyze_short_comparison_uses_a_tolerance_of_one_nanounit():
    meta, raw = synthetic_sweep(n=10, seed=4)
    raw["base|P_H3_g0.50"][1] = raw["base|R"][1].copy()
    raw["base|P_H3_g0.50"][1, :4] += 5e-10                                                    # innerhalb der Toleranz: gleich
    raw["base|P_H3_g0.50"][1, 4:6] += 1e-6                                                    # deutlich mehr
    raw["base|P_H3_g0.50"][1, 6:8] -= 1e-6                                                    # deutlich weniger
    data, _ = AN.analyze(meta, raw)
    assert data["configs"]["base"]["short_cmp"] == {"more": 0.2, "less": 0.2, "equal": 0.6}
    raw["base|P_H3_g0.50"][1, :4] = raw["base|R"][1, :4] - 5e-10
    assert AN.analyze(meta, raw)[0]["configs"]["base"]["short_cmp"] == {"more": 0.2, "less": 0.2, "equal": 0.6}


def test_analyze_cross_validation_keeps_a_small_positive_gain():
    meta, raw = synthetic_sweep(n=20, seed=5)
    for key in list(raw):
        if key.startswith("base|P_") or key.startswith("base|E_"):
            raw[key] = raw["base|R"].copy()
    raw["base|P_H3_g0.50"][0] -= 0.05                                                          # billiger um 0,05 Kosteneinheiten (unter 1): trotzdem gewählt
    data, _ = AN.analyze(meta, raw)
    c = data["configs"]["base"]
    assert c["cv_chosen"] == ["P_H3_g0.50", "P_H3_g0.50"] and c["cv_gain"] > 0.0 and c["cv_gain"] == pytest.approx(100 * 0.05 / (raw["base|R"][0] + 5.0 * raw["base|R"][1] - c["tau"] * raw["base|R"][2]).mean(), rel=1e-3)


def test_analyze_spearman_excludes_configs_with_mass_shortages():
    meta, raw = synthetic_sweep(n=12, seed=6)
    raw["q100|R"][1] = 200.0                                                                   # Fehlmenge im Mittel genau 200: ausgeschlossen
    raw["q600|R"][1] = 200.5
    data, _ = AN.analyze(meta, raw)
    assert data["spearman_gain_vs_Q_over_qL"]["n"] == 1
    raw["q600|R"][1] = 199.5
    assert AN.analyze(meta, raw)[0]["spearman_gain_vs_Q_over_qL"]["n"] == 2


def test_analyze_report_lists_at_most_25_negative_pairs_worst_first():
    rng = np.random.default_rng(1)
    n = 12
    pols = {"R": [0.0, 0.5, 0.0]}
    for h in (1, 2, 3, 4, 6, 8, 12):
        for g in (0.10, 0.25, 0.5, 1.0):
            pols["P_H%d_g%.2f" % (h, g)] = [float(h), g, 0.0]
    pols.update({"E_L1": [0.0, 0.5, 1.0], "E_L2": [0.0, 0.5, 2.0], "E_L3": [0.0, 0.5, 3.0]})
    meta = {"metrics": METRICS, "policies": pols, "configs": {"base": {}}, "n_inst": n, "days": 120, "penalty": {"base": 5.0}}
    raw = {}
    for k, pol in enumerate(pols):
        arr = np.zeros((12, n))
        arr[0] = 1000.0 * (1 + 0.01 * k) + rng.normal(0, 1, n)                                 # jede Regel teurer als reaktiv, um so mehr, je später
        arr[3] = 1500.0
        arr[5] = 100
        arr[6] = 200
        arr[10] = 90
        raw[f"base|{pol}"] = arr
    raw["base|R"][0] = 1000.0
    data, report = AN.analyze(meta, raw)
    neg_lines = [x for x in report if x.startswith("   base ")]
    assert len(neg_lines) == 25 and any("insgesamt 28 von 28" in x for x in report)
    means = [float(x.split()[-3]) for x in neg_lines]
    assert means == sorted(means)                                                              # schlimmste zuerst
    assert data["n_negative_pairs"] == [28, 28]


def test_analyze_command_line_entry_point(tmp_path):
    meta, raw = synthetic_sweep(n=10, seed=8)
    (tmp_path / "sweep_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    np.savez_compressed(tmp_path / "raw_sweep.npz", **raw)
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "analyze.py"), str(tmp_path)], capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and (tmp_path / "sweep_data.json").exists() and (tmp_path / "sweep_report.txt").exists()
    text = (tmp_path / "sweep_data.json").read_text(encoding="utf-8")
    assert text.startswith("{\n \"configs\"")                                                  # eingerückt mit einem Leerzeichen


# ---------------------------------------------------------------------------------------------------
# build_results
# ---------------------------------------------------------------------------------------------------
def test_build_results_main_reads_the_sources_and_the_sweep_path(tmp_path, monkeypatch):
    meta = {"n_inst": 200, "days": 120, "policies": {"R": [0.0, 0.5, 0.0]}, "configs": {"base": {}}, "penalty": {"base": 5.0}}
    sweep = {"configs": {"base": synthetic_cell()}, "spearman_gain_vs_Q_over_qL": {"rho": 0.7, "n": 1}, "base_distribution": {"min": 1.0}, "n_negative_pairs": [0, 28], "E_negative_pairs": [0, 3]}
    orow = synthetic_rows()[0]
    agg = {k: 1.0 for k in BR.ORACLE_AGG_KEYS}
    src = tmp_path / "src"
    src.mkdir()
    (src / "sweep_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (src / "oracle_data_Q150.json").write_text(json.dumps({"agg": agg, "rows": [orow]}), encoding="utf-8")
    (src / "oracle_data_Q250.json").write_text(json.dumps({"agg": agg, "rows": [orow, orow]}), encoding="utf-8")
    other = tmp_path / "other_sweep.json"
    other.write_text(json.dumps(sweep), encoding="utf-8")
    monkeypatch.setattr(BR, "OUT", tmp_path / "out" / "irp_results.json")
    assert BR.main([str(src), str(other)]) == 0
    out = json.loads((tmp_path / "out" / "irp_results.json").read_text(encoding="utf-8"))
    assert [c["name"] for c in out["cells"]] == ["base"] and len(out["oracle"]["Q150"]["rows"]) == 1 and len(out["oracle"]["Q250"]["rows"]) == 2
    fallback = src / "sweep_data.json"                                                          # ohne zweites Argument: erst tools/_out, sonst die Datei im Quellordner
    fallback.write_text(json.dumps(sweep), encoding="utf-8")
    monkeypatch.setattr(BR, "ROOT", tmp_path / "leer")
    assert BR.main([str(src)]) == 0 and BR.OUT.exists()


# ---------------------------------------------------------------------------------------------------
# oracle_run
# ---------------------------------------------------------------------------------------------------
def test_oracle_run_rule_table_and_target_value():
    assert OQ.POL == {"R": (0.0, 0.5, 0.0), "P_H3_g0.50": (3.0, 0.5, 0.0), "P_H8_g0.25": (8.0, 0.25, 0.0), "E_L1": (0.0, 0.5, 1.0)} and OQ.TAU == 0.6


def test_oracle_aggregate_report_lines_word_for_word():
    rows = synthetic_rows()
    rows[1]["R_short"] = 4.0
    rows[1]["R_beats_oracle_with_shortage"] = False
    agg, lines = OQ.aggregate(rows, 150, 120.0, 3.5)
    JO, JR, P3 = np.array([100.0, 120.0, 140.0]), np.array([130.0, 150.0, 140.0]), np.array([110.0, 130.0, 150.0])
    gap = 100 * (JR - JO) / JO
    want_r = ("%-12s J %8.1f   Luecke zum Optimum: Mittel %5.1f +- %3.1f %%  Median %5.1f  [Q1 %5.1f .. Q3 %5.1f]  Min %5.1f  Max %5.1f   Fehlmenge>0 in %d Instanzen (davon %d mit J unter dem Optimum, weil das Optimum keine Fehlmenge erlaubt)"
              % ("R", JR.mean(), gap.mean(), gap.std(ddof=1) / math.sqrt(3), np.median(gap), np.percentile(gap, 25), np.percentile(gap, 75), gap.min(), gap.max(), 1, 0))
    assert lines[1] == want_r and agg["total_time"] == 3.5 and agg["R"] == pytest.approx(JR.mean())
    gain = 100 * (JR - P3) / JR
    closed = (JR - P3) / np.where(JR - JO > 1e-9, JR - JO, np.nan)
    want_p = ("%-12s Gewinn gegenueber R: %5.1f +- %3.1f %% (Median %5.1f); schliesst im Mittel %.0f %% des Rueckstands von R zum Optimum"
              % ("P_H3_g0.50", gain.mean(), gain.std(ddof=1) / math.sqrt(3), np.median(gain), 100 * np.nanmean(closed)))
    assert want_p in lines
    assert agg["P_H3_g0.50_gain_over_R"]["se"] == pytest.approx(gain.std(ddof=1) / math.sqrt(3)) and agg["P_H3_g0.50_gain_over_R"]["median"] == pytest.approx(np.median(gain))
    gap_o = 100 * (JR - JO) / JR
    assert agg["oracle_gain_over_R"]["se"] == pytest.approx(gap_o.std(ddof=1) / math.sqrt(3)) and any(x == "Optimum gegenueber R: Gewinn %5.1f +- %3.1f %% (Median %5.1f)" % (gap_o.mean(), gap_o.std(ddof=1) / math.sqrt(3), np.median(gap_o)) for x in lines)
    assert "Exakter Maszstab: 3 Instanzen (N=7, 6 Tage, Q=150, 1 Tour/Tag, sigma=0), davon 3 bewiesen optimal (Zeitlimit 120 s, 8 Arbeiter); tau=0.6" == lines[0]


def test_oracle_aggregate_counts_shortage_rows_that_beat_the_optimum():
    rows = synthetic_rows()
    rows[0]["E_L1_short"] = 3.0
    rows[0]["E_L1_beats_oracle_with_shortage"] = True
    rows[2]["E_L1_short"] = 1.0
    _, lines = OQ.aggregate(rows, 150, 120.0)
    e_line = next(x for x in lines if x.startswith("E_L1"))
    assert "Fehlmenge>0 in 2 Instanzen (davon 1 mit J unter dem Optimum" in e_line


def test_oracle_run_main_defaults_and_argument_order(tmp_path, monkeypatch):
    seen = []

    def fake(s, wagon, tl):
        seen.append((s, wagon, tl))
        return dict(synthetic_rows()[s % 3], seed=s)
    monkeypatch.setattr(OQ, "solve_row", fake)
    monkeypatch.setattr(OQ, "ROOT", tmp_path)
    assert OQ.main([]) == 0
    assert seen[0] == (0, 150, 120.0) and len(seen) == 20 and (tmp_path / "tools" / "_out" / "oracle_data_Q150.json").exists()
    seen.clear()
    assert OQ.main(["4", "30", "250"]) == 0
    assert seen == [(s, 250, 30.0) for s in range(4)] and (tmp_path / "tools" / "_out" / "oracle_data_Q250.json").exists()


# ---------------------------------------------------------------------------------------------------
# confirm_default
# ---------------------------------------------------------------------------------------------------
def test_confirm_default_matches_an_independent_computation():
    seeds = range(200, 208)
    res = CD.confirm({}, seeds=seeds)
    reactive = [P.simulate(M.make_instance(s, D=120), 0.0, 0.5) for s in seeds]
    bundle = [P.simulate(M.make_instance(s, D=120), 3.0, 0.5) for s in seeds]
    tau = sum(r["routing"] for r in reactive) / sum(r["delivered"] for r in reactive)
    jr = np.array([r["routing"] + 5.0 * r["short"] - tau * r["dI"] for r in reactive])
    jp = np.array([r["routing"] + 5.0 * r["short"] - tau * r["dI"] for r in bundle])
    diff = jr - jp
    assert res["tau"] == pytest.approx(tau) and res["mean"] == pytest.approx(100 * diff.mean() / jr.mean()) and res["se"] == pytest.approx(100 * diff.std(ddof=1) / math.sqrt(8) / jr.mean())
    per = 100 * diff / jr
    assert (res["median"], res["q1"], res["q3"], res["min"], res["max"]) == pytest.approx((np.median(per), np.percentile(per, 25), np.percentile(per, 75), per.min(), per.max()))
    pen = CD.confirm({"pfac": 1.0}, seeds=range(200, 204))
    ref = [P.simulate(M.make_instance(s, D=120, pfac=1.0), 0.0, 0.5) for s in range(200, 204)]
    bun = [P.simulate(M.make_instance(s, D=120, pfac=1.0), 3.0, 0.5) for s in range(200, 204)]
    t2 = sum(r["routing"] for r in ref) / sum(r["delivered"] for r in ref)
    a = np.array([r["routing"] + 100.0 * r["short"] - t2 * r["dI"] for r in ref])
    b = np.array([r["routing"] + 100.0 * r["short"] - t2 * r["dI"] for r in bun])
    assert pen["mean"] == pytest.approx(100 * (a - b).mean() / a.mean())
    assert CD.CONFIGS == {"base": {}, "q100": {"Q": 100.0}, "q600": {"Q": 600.0}, "flotte_F1_Q300": {"F": 1, "Q": 300.0}}


def test_confirm_default_command_line_entry_point():
    r = subprocess.run([sys.executable, str(ROOT / "tools" / "confirm_default.py"), "q100"], capture_output=True, text=True, timeout=300, encoding="utf-8")
    assert r.returncode == 0 and r.stdout.startswith("q100: Sweep (Seeds 0-199) +0.40 +- 0.05 %") and "Differenz der Mittel" in r.stdout
    sweep = R.p_gain(R.by_name(R.load_results(), "q100"), 3, 0.5)
    assert f"Median {sweep['median']:.2f}" in r.stdout and f"Verlust {100 * sweep['share_loss']:.0f} %" in r.stdout
