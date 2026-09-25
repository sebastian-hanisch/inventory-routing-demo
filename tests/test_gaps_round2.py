"""Tests, die nach dem zweiten Fehler-Einbau-Lauf noch echte Lücken schließen: Gleichstand-Regel der Zellenzuordnung, gerundete Anzeige der Tagestabelle, Verlustanteile in den Meldungen."""
import pytest

import irp_constants as C
import irp_frozen as FZ
import irp_live as LV
import irp_presets as PS
import irp_results as R
import irp_ui_panel as UI

DATA = R.load_results()


@pytest.fixture(scope="module")
def live():
    inst = FZ.frozen("base", 0)
    original = LV.make_live_instance
    LV.make_live_instance = lambda *a, **k: inst
    try:
        return LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)
    finally:
        LV.make_live_instance = original


def test_day_table_rounds_reach_to_one_decimal_and_quantity_to_whole_units(live):
    df = UI.halteliste_frame(LV.day_view(live, "P", 5))
    assert len(df) > 0
    assert all(abs(v * 10 - round(v * 10)) < 1e-9 for v in df["Reichweite (Tage)"])
    assert any(abs(v * 100 - round(v * 100)) > 1e-6 for v in [s["reach"] for t in LV.day_view(live, "P", 5)["tours"] for s in t["stops"]])   # die Rohwerte sind feiner
    assert all(float(v).is_integer() for v in df["Liefermenge"])


def test_message_and_distribution_sentence_print_the_loss_share_in_whole_percent():
    seen = set()
    for cell in R.cells(DATA):
        for (H, g) in ((3, 0.5), (12, 1.0), (1, 0.0), (3, 0.0)):
            try:
                j = R.judge(cell, H, g)
            except (KeyError, TypeError):
                continue
            if j["share_loss"] < 0.1 or j["state"] in seen:
                continue
            seen.add(j["state"])
            fake = {"H": H, "gamma": g, "rules": {"P": {"gain": 1.0}}}
            state, text = UI.message(fake, cell)
            assert state == j["state"] and f"in {round(100 * j['share_loss'])} % der Instanzen ein Verlust" in text
            assert f"in {round(100 * R.p_gain(cell, H, g)['share_loss'])} % der Instanzen ein Verlust" in UI.distribution_sentence(cell, H, g)
    assert len(seen) >= 2                                                            # mindestens zwei der drei Zustände wurden mit echtem Verlustanteil geprüft


def test_base_cell_wins_every_tie_it_takes_part_in():
    """Gleichstand-Regel: bei gleichem Abstand gewinnt die Basiszelle (Rang -1) vor jeder anderen Zelle."""
    import itertools
    ties = 0
    for combo in itertools.product(C.CUSTOMER_OPTIONS, C.CAPACITY_OPTIONS, C.FLEET_OPTIONS, C.SIGMA_OPTIONS, C.PENALTY_OPTIONS):
        iw = R._level_index(tuple(combo))
        dists = []
        for cell, setting in R.assignable_cells(DATA):
            ic = R._level_index(setting)
            dists.append((sum(abs(iw[p] - ic[p]) for p in R.PARAM_PRIORITY), cell["name"], setting == R.BASE_SETTING))
        dmin = min(d for d, _, _ in dists)
        at_min = [(n, is_base) for d, n, is_base in dists if d == dmin]
        if len(at_min) > 1 and any(b for _, b in at_min):
            ties += 1
            assert R.nearest_cell(DATA, *combo)[0]["name"] == next(n for n, b in at_min if b)
    assert ties > 0                                                                    # der Fall kommt vor, der Test prüft ihn also wirklich


def test_grid_summary_and_negative_pairs_treat_zero_as_neither_positive_nor_negative_and_keep_the_three_point_boundary(monkeypatch):
    cell = {"P": {"a": {"mean": 0.0, "se": 0.1}, "b": {"mean": 1.0, "se": 0.1}, "c": {"mean": -1.0, "se": 0.1}}}
    s = R.grid_summary(cell)
    assert (s["n"], s["positive"], s["negative"], s["near_best"], s["best_key"]) == (3, 1, 1, 3, "b")
    assert R.grid_summary({"P": {"a": {"mean": 4.0}, "b": {"mean": 1.0}, "c": {"mean": 0.9}}})["near_best"] == 2         # genau 3 Prozentpunkte unter der besten gehören dazu
    monkeypatch.setattr(R, "cells", lambda data: [dict(cell, name="x")])
    assert [(k, m) for _, k, m, _ in R.negative_pairs(None)] == [("c", -1.0)]


def test_analyze_quartiles_and_report_line_on_widely_spread_gains():
    """Mit breit gestreuten Einzelgewinnen unterscheiden sich die Perzentile 25/75/90 schon in der einen Nachkommastelle des Berichts (die schmale Standardstreuung des Testdatensatzes verdeckte das)."""
    import numpy as np
    from test_tools import AN, synthetic_sweep
    meta, raw = synthetic_sweep(n=40, seed=11)
    n = 40
    raw["base|P_H3_g0.50"][0] = raw["base|R"][0] * np.linspace(0.4, 1.3, n) ** 2                    # Gewinne von etwa +84 % bis -69 %
    data, report = AN.analyze(meta, raw)
    tau = raw["base|R"][0].sum() / raw["base|R"][3].sum()
    jr = raw["base|R"][0] + 5.0 * raw["base|R"][1] - tau * raw["base|R"][2]
    jp = raw["base|P_H3_g0.50"][0] + 5.0 * raw["base|P_H3_g0.50"][1] - tau * raw["base|P_H3_g0.50"][2]
    per = 100 * (jr - jp) / jr
    d = data["configs"]["base"]["default"]
    assert d["q1"] == pytest.approx(np.percentile(per, 25), rel=1e-12) and d["q3"] == pytest.approx(np.percentile(per, 75), rel=1e-12) and d["median"] == pytest.approx(np.median(per), rel=1e-12)
    line = next(x for x in report if x.startswith("Basis: Verteilung"))
    assert line == ("Basis: Verteilung der Einzelgewinne (Standardregel, %% je Instanz): Min %.1f, Q1 %.1f, Median %.1f, Q3 %.1f, P90 %.1f, Max %.1f; Instanzen mit Verlust %.1f %%"
                    % (per.min(), np.percentile(per, 25), np.median(per), np.percentile(per, 75), np.percentile(per, 90), per.max(), 100 * (per < 0).mean()))
    assert line != ("Basis: Verteilung der Einzelgewinne (Standardregel, %% je Instanz): Min %.1f, Q1 %.1f, Median %.1f, Q3 %.1f, P90 %.1f, Max %.1f; Instanzen mit Verlust %.1f %%"
                    % (per.min(), np.percentile(per, 26), np.median(per), np.percentile(per, 76), np.percentile(per, 91), per.max(), 100 * (per < 1).mean()))


def _spread_sweep():
    """Testdatensatz mit gezielten Randfällen: Gewinn genau 0, Rauschen unter der Toleranz, Gewinn zwischen 0 und 1 %, je Konfiguration andere Rangfolge der Regeln."""
    import numpy as np
    from test_tools import synthetic_sweep
    meta, raw = synthetic_sweep(n=40, seed=5)
    rng = np.random.default_rng(1)
    for c in meta["configs"]:
        for pol in meta["policies"]:
            if pol != "R":
                raw[f"{c}|{pol}"][0] = raw[f"{c}|R"][0] * (1 - rng.uniform(-0.3, 0.3, 40))
    raw["base|P_H3_g0.50"][:, 6] = raw["base|R"][:, 6]                                        # Instanz 6: Regel = reaktiv, Gewinn genau 0
    raw["base|P_H3_g0.50"][:, 2] = raw["base|R"][:, 2]
    raw["base|P_H3_g0.50"][0, 2] += 5e-10                                                    # Instanz 2: um 5e-10 teurer, unter der Toleranz 1e-9, kein Verlust
    raw["base|P_H3_g0.50"][:, 5] = raw["base|R"][:, 5]
    tau = raw["base|R"][0].sum() / raw["base|R"][3].sum()
    jr5 = raw["base|R"][0, 5] + 5.0 * raw["base|R"][1, 5] - tau * raw["base|R"][2, 5]
    raw["base|P_H3_g0.50"][0, 5] -= 0.004 * jr5                                              # Instanz 5: Gewinn +0,4 % (zwischen 0 und 1)
    for pol in ("P_H1_g0.10", "E_L1"):                                                         # Konfiguration q100: Regel identisch zu reaktiv: Mittel genau 0
        raw[f"q100|{pol}"] = raw["q100|R"].copy()
    tau = raw["q100|R"][0].sum() / raw["q100|R"][3].sum()
    jr = raw["q100|R"][0] + 5.0 * raw["q100|R"][1] - tau * raw["q100|R"][2]
    for pol in ("P_H12_g1.00", "E_L2"):                                                        # Mittel zwischen 0 und 1 %
        raw[f"q100|{pol}"] = raw["q100|R"].copy()
        raw[f"q100|{pol}"][0] -= 0.004 * jr
    return meta, raw


def test_analyze_full_row_and_counts_against_an_independent_computation():
    import numpy as np
    from test_tools import AN
    meta, raw = _spread_sweep()
    data, report = AN.analyze(meta, raw)
    M = {k: i for i, k in enumerate(meta["metrics"])}
    g = lambda c, pol, k: raw[f"{c}|{pol}"][M[k]]
    ppol = [q for q in meta["policies"] if q.startswith("P_")]
    epol = [q for q in meta["policies"] if q.startswith("E_")]
    neg = 0
    eneg = 0
    cs, xs, ys = [], [], []
    for c in meta["configs"]:
        tau = g(c, "R", "routing").sum() / g(c, "R", "delivered").sum()
        J = lambda pol: g(c, pol, "routing") + 5.0 * g(c, pol, "short") - tau * g(c, pol, "dI")
        jr = J("R")
        row = data["configs"][c]
        diffs = {q: jr - J(q) for q in meta["policies"] if q != "R"}
        for q in ppol:
            neg += diffs[q].mean() < 0
        for q in epol:
            eneg += diffs[q].mean() < 0
        # Kreuzvalidierung: erste Runde wählt auf den geraden Instanzen (Index 0, 2, ...) und bewertet auf den ungeraden
        A = np.arange(40) % 2 == 0
        chosen, cv = [], []
        for tr, te in ((A, ~A), (~A, A)):
            cands = {q: diffs[q][tr].mean() for q in ppol}
            best = max(cands, key=cands.get)
            chosen.append(best if cands[best] >= 0 else "R")
            cv.append(0.0 if cands[best] < 0 else 100 * diffs[best][te].sum() / jr[te].sum())
        assert row["cv_chosen"] == chosen and row["cv_gain"] == pytest.approx(float(np.mean(cv)), rel=1e-12)
        nR, nP, nE = g(c, "R", "n_routes").sum(), g(c, "P_H3_g0.50", "n_routes").sum(), g(c, "E_L2", "n_routes").sum()
        assert row["util_R"] == pytest.approx(g(c, "R", "util_sum").sum() / nR, rel=1e-12) and row["util_P"] == pytest.approx(g(c, "P_H3_g0.50", "util_sum").sum() / nP, rel=1e-12)
        assert row["util_E"] == pytest.approx(g(c, "E_L2", "util_sum").sum() / nE, rel=1e-12)
        assert row["routes_E"] == pytest.approx(g(c, "E_L2", "n_routes").mean()) and row["visits_E"] == pytest.approx(g(c, "E_L2", "n_visits").mean())
        cq = meta["configs"][c].get("Q", 300.0) / (g(c, "R", "delivered").sum() / g(c, "R", "n_visits").sum())
        assert row["Q_over_qL"] == pytest.approx(cq, rel=1e-12)
        if row["short_R"] < 200:
            cs.append(c)
            xs.append(row["Q_over_qL"])
            ys.append(row["default"]["mean"])
    assert data["n_negative_pairs"] == [int(neg), len(ppol) * 3] and data["E_negative_pairs"] == [int(eneg), len(epol) * 3]
    rank = lambda v: np.argsort(np.argsort(v)).astype(float)
    assert data["spearman_gain_vs_Q_over_qL"]["rho"] == pytest.approx(float(np.corrcoef(rank(xs), rank(ys))[0, 1]), rel=1e-12) and abs(data["spearman_gain_vs_Q_over_qL"]["rho"]) < 1.0 - 1e-9
    # Randfälle der Verlustanteile
    base = data["configs"]["base"]["default"]
    tau = g("base", "R", "routing").sum() / g("base", "R", "delivered").sum()
    jr = g("base", "R", "routing") + 5.0 * g("base", "R", "short") - tau * g("base", "R", "dI")
    jp = g("base", "P_H3_g0.50", "routing") + 5.0 * g("base", "P_H3_g0.50", "short") - tau * g("base", "P_H3_g0.50", "dI")
    assert base["share_loss"] == float(((jr - jp) < -1e-9).mean())
    per = 100 * (jr - jp) / jr
    assert data["base_distribution"]["loss_share"] == float((per < 0).mean()) and np.any((per > 0) & (per < 1)) and np.any(per == 0)
    line = next(x for x in report if x.startswith("Basis: Verteilung"))
    assert line.endswith("Instanzen mit Verlust %.1f %%" % (100 * (per < 0).mean()))
    # Berichtszeilen mit Verlust- und Paarzahlen
    assert any(x == "   insgesamt %d von %d Regel-Konfiguration-Paaren im Mittel schlechter als reaktiv" % (neg, len(ppol) * 3) for x in report)
    assert any(x == "Regeln E (frueher liefern, ohne Tourenbezug): %d von %d Paaren mit negativem Gewinn" % (eneg, len(epol) * 3) for x in report)
    assert any("%2.0f%% Verlust" % (100 * base["share_loss"]) in x for x in report if x.startswith("base"))


def test_oracle_aggregate_gap_quartiles_and_the_closed_share_guard():
    import numpy as np
    from test_tools import OQ, synthetic_rows
    rows = synthetic_rows()
    for r, (jo, jr, jp) in zip(rows, ((100.0, 200.0, 105.0), (120.0, 121.0, 300.0), (140.0, 140.0 + 1e-9, 90.0))):
        r["J_oracle"], r["R"], r["P_H3_g0.50"] = jo, jr, jp
    agg, lines = OQ.aggregate(rows, 150, 120.0)
    JO, JR, P3 = np.array([100.0, 120.0, 140.0]), np.array([200.0, 121.0, 140.0 + 1e-9]), np.array([105.0, 300.0, 90.0])
    for name, v in (("R", JR), ("P_H3_g0.50", P3)):
        gap = 100 * (v - JO) / JO
        d = agg[name + "_gap_to_oracle"]
        assert (d["q1"], d["q3"], d["median"], d["min"], d["max"]) == pytest.approx((np.percentile(gap, 25), np.percentile(gap, 75), np.median(gap), gap.min(), gap.max()), rel=1e-12)
        line = next(x for x in lines if x.startswith(name.ljust(12)) and "Luecke" in x)
        assert "[Q1 %5.1f .. Q3 %5.1f]" % (np.percentile(gap, 25), np.percentile(gap, 75)) in line
    # Anteil des geschlossenen Rückstands: Instanzen, in denen R höchstens 1e-9 über dem Optimum liegt, zählen nicht mit (Nenner ungültig)
    closed = (JR - P3) / np.where(JR - JO > 1e-9, JR - JO, np.nan)
    assert agg["P_H3_g0.50_share_of_R_gap_closed"] == pytest.approx(float(np.nanmean(closed)), rel=1e-12)


def test_confirm_default_report_prints_the_difference_of_means_and_the_loss_shares(monkeypatch, capsys):
    import math
    from test_tools import CD
    sweep = {"mean": 17.0, "se": 0.3, "median": 16.5, "q1": 14.8, "q3": 18.1, "share_loss": 0.6}
    new = {"tau": 1.234, "mean": 16.0, "se": 0.4, "median": 16.0, "q1": 14.0, "q3": 18.0, "min": 9.0, "max": 25.0, "share_loss": 0.4, "n": 100}
    monkeypatch.setattr(CD.R, "p_gain", lambda cell, H, g: sweep)
    monkeypatch.setattr(CD.R, "by_name", lambda data, name: {"name": name})
    monkeypatch.setattr(CD, "confirm", lambda overrides, **k: new)
    assert CD.main(["base"]) == 0
    text = capsys.readouterr().out
    se_diff = math.hypot(0.3, 0.4)
    assert "Verlust 60 %" in text and "Verlust 40 %" in text
    assert f"Differenz der Mittel {-1.0:+.2f} ({-1.0 / se_diff:+.1f} SE der Differenz)" in text


def test_oracle_signature_defaults():
    import inspect
    import irp_oracle as OR
    d = inspect.signature(OR.solve_oracle).parameters
    assert (d["time_limit"].default, d["workers"].default, d["seed"].default) == (60.0, 8, 1)
    s = inspect.signature(OR.solve_small).parameters
    assert (s["time_limit"].default, s["workers"].default) == (20.0, 8)
