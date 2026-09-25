"""Tests, die im ersten vollständigen Fehler-Einbau-Lauf überlebende Mutanten der Ergebnis-Auswertung, der Abnahmekriterien, des Panels und der Konstanten schließen: genaue Texte und
Zahlenformate, Randfälle der Zell-Zuordnung und der Gitter-Zusammenfassung, unveränderliche Regler-Spezifikation."""
import copy
import dataclasses

import pytest

import irp_constants as C
import irp_format as F
import irp_frozen as FZ
import irp_live as LV
import irp_presets as PR
import irp_results as R
import irp_stories as ST
import irp_ui_panel as UI

DATA = R.load_results()
BASE = R.find_cell(DATA)


# ---------------------------------------------------------------------------------------------------
# Ergebnis-Auswertung
# ---------------------------------------------------------------------------------------------------
def test_cfg_setting_checks_every_control_and_tolerates_rounding_noise_in_the_penalty():
    good = copy.deepcopy(BASE["cfg"])
    assert R.cfg_setting(good) == (20, 300, 6, 0.3, 5)
    for key, bad in (("N", 25), ("Q", 250.0), ("F", 3), ("sigma", 0.45), ("penalty", 7.0), ("kappa", 0.0), ("depot", "far"), ("clustered", True), ("tank_lo", 4.0), ("tank_hi", 20.0)):
        cfg = dict(good, **{key: bad})
        assert R.cfg_setting(cfg) is None, key
    assert R.cfg_setting(dict(good, penalty=5.0 + 1e-10)) == (20, 300, 6, 0.3, 5)                     # Rundungsrauschen der Datei ist keine andere Strafe
    assert R.cfg_setting(dict(good, penalty=5.0 + 1e-6)) is None


def test_grid_summary_counts_against_an_independent_computation_in_every_cell():
    for cell in R.cells(DATA):
        means = [v["mean"] for v in cell["P"].values()]
        gs = R.grid_summary(cell)
        assert gs["n"] == 28 and gs["positive"] == sum(1 for v in means if v > 0) and gs["negative"] == sum(1 for v in means if v < 0)
        assert gs["best"] == max(means) and gs["near_best"] == sum(1 for v in means if v >= max(means) - 3.0) and cell["P"][gs["best_key"]]["mean"] == max(means)
    q100 = R.grid_summary(R.find_cell(DATA, Q=100.0))
    assert q100["negative"] == sum(1 for v in R.find_cell(DATA, Q=100.0)["P"].values() if v["mean"] < 0) > 0


def test_negative_summary_counts_the_pairs_with_gamma_one_and_gamma_at_most_a_quarter():
    def fake(name, means):
        p = {R.p_key(h, g): dict(mean=m, se=0.1, median=m, q1=m, q3=m, share_loss=0.0) for (h, g), m in means.items()}
        return {"name": name, "P": p}
    d = {"cells": [fake("a", {(3, 0.25): -1.0, (3, 0.5): -2.0, (3, 1.0): -3.0, (12, 1.0): -4.0, (3, 0.1): 5.0}), fake("b", {(1, 0.1): -0.5, (2, 0.25): 1.0, (4, 0.5): -9.0})]}
    assert R.negative_summary(d) == (6, 8, 2, 2)                                                       # sechs negativ, zwei mit gamma 1,0, zwei mit gamma <= 0,25 (0,25 und 0,1)
    pairs = R.negative_pairs(d)
    assert [p[2] for p in pairs] == [-9.0, -4.0, -3.0, -2.0, -1.0, -0.5] and pairs[0][1] == "P_H4_g0.50"
    assert not any(p[2] == 0.0 for p in R.negative_pairs({"cells": [fake("z", {(3, 0.5): 0.0})]}))    # genau 0 ist nicht negativ


def test_oracle_summary_reports_the_slowest_instance():
    for wagon in (150, 250):
        assert R.oracle_summary(DATA, wagon)["time_max"] == max(r["time"] for r in R.oracle_rows(DATA, wagon))


def test_coverage_lists_only_exact_cells_at_distance_zero():
    cov = R.coverage(DATA)
    assert cov["by_distance"][0] == cov["exact"] == 19 and set(cov["by_distance"]) == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------------------------------
# Abnahmekriterien: genaue Texte
# ---------------------------------------------------------------------------------------------------
def test_criteria_texts_of_every_preset_word_for_word():
    t = {n: [x for _, x in ST.criteria(n, DATA)] for n in C.PRESETS}
    assert t["Standard"] == ["Bündeln P(3; 0,5) über 2 Standardfehler und mindestens 10 %: +17,0 ± 0,3 %", "Anteil der Instanzen mit Verlust 0 %: 0 %",
                             "früher liefern E(2) teurer als reaktiv (unter −2 Standardfehler): -16,3 ± 0,2 %"]
    assert t["Wagen fast voll"] == ["|Gewinn| unter 2 %: +0,4 ± 0,1 %"]
    assert t["Großer Wagen"] == ["Gewinn mindestens 25 % und über 2 Standardfehler: +29,9 ± 0,3 %", "größer als bei Wagen 300 (um mehr als 2 Standardfehler der Differenz): +29,9 gegen +17,0 %"]
    assert t["Zu großzügig"] == ["Gewinn unter −10 % und unter −2 Standardfehler: -26,3 ± 0,5 %", "mit γ = 0,25 dagegen positiv (über 2 Standardfehler): +9,0 ± 0,2 %"]
    assert t["Knappe Flotte"] == ["Gewinn mindestens 15 % und über 2 Standardfehler: +19,2 ± 0,3 %", "Fehlmenge sinkt: 386 auf 130 Einheiten"]
    d = copy.deepcopy(DATA)
    R.p_gain(R.find_cell(d), 3, 0.5)["share_loss"] = 0.03
    assert [x for _, x in ST.criteria("Standard", d)][1] == "Anteil der Instanzen mit Verlust 0 %: 3 %"


def fake_live(gp=0.0, ge=0.0, short_r=0.0, short_p=0.0):
    mk = lambda gain, short: {"gain": gain, "res": {"short": short}}
    return {"rules": {"R": mk(0.0, short_r), "P": mk(gp, short_p), "E": mk(ge, 0.0)}}


def test_day_criteria_texts_word_for_word():
    txt = lambda name, live, alt=None: [x for _, x in ST.day_criteria(name, live, alt)]
    assert txt("Standard", fake_live(gp=12.34, ge=-4.56)) == ["Bündeln billiger als reaktiv: +12,3 %", "früher liefern teurer als reaktiv: -4,6 %"]
    assert txt("Wagen fast voll", fake_live(gp=1.24)) == ["|Gewinn| auf der Instanz unter 3 %: +1,2 %"]
    assert txt("Großer Wagen", fake_live(gp=20.04)) == ["Gewinn auf der Instanz mindestens 15 %: +20,0 %"]
    assert txt("Zu großzügig", fake_live(gp=-5.55), fake_live(gp=2.46)) == ["Bündeln teurer als reaktiv: -5,5 %", "mit γ = 0,25 dagegen billiger: +2,5 %"]
    assert txt("Knappe Flotte", fake_live(gp=7.04, short_r=120.4, short_p=79.6)) == ["Gewinn auf der Instanz mindestens 5 %: +7,0 %", "Fehlmenge auf der Instanz sinkt: 120 auf 80 Einheiten"]


def test_the_standard_error_condition_treats_both_signs_and_the_sign_argument_exactly():
    assert ST._clear(3.0, 1.0) is True and ST._clear(2.0, 1.0) is False and ST._clear(-3.0, 1.0) is False
    assert ST._clear(-3.0, 1.0, -1) is True and ST._clear(-2.0, 1.0, -1) is False and ST._clear(3.0, 1.0, -1) is False


# ---------------------------------------------------------------------------------------------------
# Regler-Spezifikation und Konstanten
# ---------------------------------------------------------------------------------------------------
def test_setting_specs_are_immutable():
    for spec in PR.SETTING_SPECS.values():
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.default = 99


def test_the_five_presets_word_for_word_and_the_chart_height():
    base = dict(customers=20, capacity=300, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=211)
    assert C.PRESETS == {"Standard": base, "Wagen fast voll": dict(base, capacity=100), "Großer Wagen": dict(base, capacity=600), "Zu großzügig": dict(base, capacity=150, horizon=12, gamma=1.0),
                         "Knappe Flotte": dict(base, fleet=1)}
    assert C.CHART_HEIGHT == 380 and UI.CHART_TABLE_HEIGHT == 440 and C.CACHE_ENTRIES == 24


def test_format_helpers_signs_digits_and_thousands():
    assert F.fmt_num(1.25, 1) == "1,2" and F.fmt_num(1.25, 2) == "1,25" and F.fmt_num(1.0, 1, True) == "+1,0" and F.fmt_num(-1.0, 1, True) == "-1,0" and F.fmt_num(0.0, 1, True) == "+0,0"
    assert F.fmt_band(1.0, 0.25) == "+1,0 ± 0,2" and F.fmt_band(1.0, 0.25, 2, False) == "1,00 ± 0,25" and F.fmt_int(14694.4) == "14.694" and F.fmt_int(-1234.5, True) == "-1.234"
    assert F.fmt_cost(2675.4) == "2.675" and F.fmt_cost(2675.4, True) == "+2.675" and F.fmt_share(0.876) == "88 %" and F.fmt_share(None) == "–"
    assert [F.fmt_level(v) for v in (0, 0.1, 0.15, 0.25, 0.3, 0.5, 0.6, 1.0)] == ["0,0", "0,1", "0,15", "0,25", "0,3", "0,5", "0,6", "1,0"]


# ---------------------------------------------------------------------------------------------------
# Panel: Zahlenformate der Tabellen und Meldungen
# ---------------------------------------------------------------------------------------------------
def fake_live_full(j, gain, routes=10, util=0.5, short=0.0):
    res = {"routing": 100.0, "short": short, "n_routes": routes, "n_visits": routes * 2, "opt_visits": 0, "deferred": 0, "util_sum": util * routes}
    return {"J": j, "gain": gain, "res": res}


def test_metric_deltas_carry_the_sign_of_the_cost_difference_for_both_rules():
    class Col:
        def __init__(self):
            self.calls = []

        def metric(self, label, value, **kw):
            self.calls.append((label, value, kw))
    live = {"H": 3.0, "gamma": 0.5, "early": 2.0, "rules": {"R": fake_live_full(1000.0, 0.0), "P": fake_live_full(1100.0, -10.0), "E": fake_live_full(950.0, 5.0)}}
    cols = [Col() for _ in range(4)]
    UI.render_metrics(cols, live)
    assert cols[1].calls[0][2]["delta"] == "+100 (-10,0 %)" and cols[2].calls[0][2]["delta"] == "-50 (+5,0 %)"
    assert cols[3].calls[0][2]["delta"] == "Auslastung 0,50 / 0,50 / 0,50"


def test_message_texts_show_median_and_loss_share_of_the_set_rule():
    live = LV.solve_live(20, 300, 6, 0.3, 5, 6, 1.0, 2, 3)
    j = R.judge(BASE, 6, 1.0)
    assert j["state"] == C.STATE_WENIG and 0.1 < j["share_loss"] < 0.9
    _, text = UI.message(live, BASE)
    assert f"(Median {F.fmt_num(j['median'], 1, True)} %, in {F.fmt_num(100 * j['share_loss'], 0)} % der Instanzen ein Verlust)" in text and "(Median +" in text
    _, bad = UI.message(dict(live, H=12.0, gamma=1.0), BASE)
    assert "in 88 % der Instanzen ein Verlust" in bad and "5,1 ± 0,3 % mehr" in bad
    _, small = UI.message(dict(live, H=3.0, gamma=0.5), R.find_cell(DATA, Q=100.0))
    assert "(Median +0,0 %" in small


def test_frames_use_exact_number_formats():
    df = UI.oracle_instances_frame(DATA, 150)
    assert df.iloc[0]["Lücke reaktiv (%)"] == "37,5" and df.iloc[0]["Lücke P(3; 0,5) (%)"] == "12,9" and df.iloc[0]["Optimum"] == "516,4"
    rows = UI.rule_rows_frame(R.rule_rows(BASE, 8, 0.25, 3))
    assert list(rows["Gewinn gegenüber reaktiv (%)"]) == ["-7,0 ± 0,2", "-16,3 ± 0,2", "-27,1 ± 0,3", "Bezug", "+20,5 ± 0,3", "+20,5"]
    assert list(UI.rule_rows_frame([("x", 0.0, 0.3), ("y", 0.0, 0.0)])["Gewinn gegenüber reaktiv (%)"]) == ["+0,0 ± 0,3", "Bezug"]
    assert list(UI.shortage_frame(R.shortage_rows(DATA))["Instanzen mit mehr Fehlmenge"]) == ["1,0 %", "0,0 %", "15,0 %"]
    cap = UI.capacity_frame(R.capacity_rows(DATA, 3, 0.5), 300)
    assert list(cap["beste Zelle, kreuzvalidiert (%)"])[:3] == ["+1,3", "+9,0", "+13,7"]
    reg = UI.regime_frame(R.regime_rows(DATA, 3, 0.5))
    assert list(reg["Anteil Verlust"])[:2] == ["0 %", "0 %"] and list(reg["Q/q"])[:2] == ["3,0", "2,9"]
    assert UI.negative_frame(DATA).iloc[1]["Gewinn (Mittel ± SE, %)"] == "-26,3 ± 0,5"


def test_comparison_table_and_rule_table_signs():
    inst = FZ.frozen("base", 0)
    original = LV.make_live_instance
    LV.make_live_instance = lambda *a, **k: inst
    try:
        live = LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)
    finally:
        LV.make_live_instance = original
    tab = UI.rule_table(live, BASE)
    assert tab.iloc[0]["Gewinn dieser Instanz (%)"] == 0.0 and tab.iloc[1]["Gewinn dieser Instanz (%)"] > 5 and tab.iloc[2]["Gewinn dieser Instanz (%)"] < 0
    assert list(tab["Fehlmenge (Einheiten)"]) == [round(live["rules"][r]["res"]["short"]) for r in "RPE"]


def test_render_day_draws_two_columns_and_tabs_for_the_two_rules():
    from streamlit.testing.v1 import AppTest

    def script(live, day, right):
        import irp_ui_panel as UI_
        UI_.render_day("t", live, day, right)
    inst = FZ.frozen("base", 0)
    original = LV.make_live_instance
    LV.make_live_instance = lambda *a, **k: inst
    try:
        live = LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)
    finally:
        LV.make_live_instance = original
    at = AppTest.from_function(script, args=(live, 5, "P"), default_timeout=120)
    at.run()
    assert not at.exception and len(at.columns) == 2 and len(at.tabs) == 2
