"""Die vorgerechnete Messreihe (data/irp_results.json): Aufbau, Zahlen der Messreihe, Urteil in drei Zuständen, Zell-Zuordnung (nächstliegende gemessene Zelle),
Gitter, Kapazitätshebel, Regime-Tabelle, exakter Maßstab. Die Zahlen stammen aus messreihe_inventory_routing/ERGEBNIS.md; hier werden sie an der Ergebnisdatei
nachgerechnet (nichts wird live gewürfelt)."""
import copy
import itertools

import pytest

import irp_constants as C
import irp_results as R

DATA = R.load_results()
BASE = R.find_cell(DATA)


def dflt(cell):
    return R.p_gain(cell, 3, 0.5)


# ---------------------------------------------------------------------------------------------------
# Aufbau
# ---------------------------------------------------------------------------------------------------
def test_file_has_27_cells_with_the_full_grid_and_all_fields():
    cells = R.cells(DATA)
    assert len(cells) == 27 and len({c["name"] for c in cells}) == 27 and cells[0]["name"] == "base"
    grid = {R.p_key(h, g) for h in C.HORIZON_OPTIONS for g in C.GAMMA_OPTIONS}
    for c in cells:
        assert set(c["P"]) == grid and len(c["P"]) == 28 and set(c["E"]) == {"E_L1", "E_L2", "E_L3"}
        for g in list(c["P"].values()) + list(c["E"].values()) + [c["default"], c["best_insample_gain"]]:
            assert set(g) == {"mean", "se", "median", "q1", "q3", "share_loss"} and g["se"] > 0 and 0.0 <= g["share_loss"] <= 1.0 and g["q1"] <= g["median"] <= g["q3"]
        assert c["default"] == c["P"]["P_H3_g0.50"]
        assert set(c["cfg"]) == set(C.BASE_CFG) | {"penalty"} and c["Q_over_qL"] > 0 and c["tau"] > 0 and c["JR"] > 0 and set(c["short_cmp"]) == {"more", "less", "equal"}
        assert sum(c["short_cmp"].values()) == pytest.approx(1.0, abs=1e-3)
    assert DATA["_meta"]["n_inst"] == C.MEASURED_N == 200 and DATA["_meta"]["days"] == C.DAYS == 120 and DATA["_meta"]["default"] == {"H": 3, "gamma": 0.5}
    assert len(DATA["_meta"]["policies"]) == 32


def test_the_base_cell_matches_the_constants_and_every_other_cell_varies_a_documented_parameter():
    assert {k: BASE["cfg"][k] for k in C.BASE_CFG} == C.BASE_CFG and BASE["cfg"]["penalty"] == 5.0
    for c in R.cells(DATA):
        differing = [k for k in C.BASE_CFG if c["cfg"][k] != C.BASE_CFG[k]]
        assert len(differing) <= 2, (c["name"], differing)                                  # n40_q600 und die Flotten variieren zwei Parameter, sonst genau einen
    two = [c["name"] for c in R.cells(DATA) if len([k for k in C.BASE_CFG if c["cfg"][k] != C.BASE_CFG[k]]) == 2]
    assert two == ["n40_q600", "flotte_F1_Q250", "flotte_F2_Q150", "tank_kurz", "tank_lang"]           # Tank: untere und obere Grenze zusammen; flotte_F1_Q300 variiert nur die Flotte


def test_measured_numbers_of_the_findings():
    b = dflt(BASE)
    assert (round(b["mean"], 1), round(b["se"], 1), round(b["median"], 1), b["share_loss"]) == (17.0, 0.3, 16.9, 0.0)
    assert round(BASE["cv_gain"], 1) == 20.5 and BASE["cv_chosen"] == ["P_H8_g0.25", "P_H8_g0.25"] and BASE["best_insample"] == "P_H8_g0.25"
    assert (round(BASE["routes_R"], 1), round(BASE["routes_P"], 1), round(BASE["visits_R"], 1), round(BASE["visits_P"], 1), round(BASE["opt_visits_P"], 1)) == (132.8, 99.7, 238.1, 260.5, 101.7)
    assert (round(BASE["util_R"], 2), round(BASE["util_P"], 2), round(BASE["km_R"]), round(BASE["km_P"])) == (0.59, 0.79, 14694, 12220)
    assert round(BASE["Q_over_qL"], 2) == 3.02 and round(BASE["qL"]) == 99 and round(BASE["tau"], 3) == 0.621
    assert [round(R.e_gain(BASE, L)["mean"], 1) for L in (1, 2, 3)] == [-7.0, -16.3, -27.1]
    assert round(BASE["short_cmp"]["more"], 3) == 0.225 and round(BASE["short_R"], 1) == 14.4 and round(BASE["short_P"], 1) == 8.5
    assert [round(dflt(R.find_cell(DATA, sigma=s))["mean"], 1) for s in (0.0, 0.15, 0.6, 1.0)] == [16.5, 17.0, 17.1, 17.3]
    assert [round(dflt(R.find_cell(DATA, Q=q))["mean"], 1) for q in (100.0, 150.0, 200.0, 450.0, 600.0, 1000.0)] == [0.4, 6.3, 10.7, 25.4, 29.9, 33.2]
    assert [round(dflt(R.by_name(DATA, n))["mean"], 1) for n in ("pen001", "pen025", "pen100")] == [16.9, 17.5, 19.0]
    assert round(dflt(R.find_cell(DATA, F=1))["mean"], 1) == 19.2 and round(dflt(R.by_name(DATA, "flotte_F1_Q250"))["mean"], 1) == 30.4 and round(dflt(R.by_name(DATA, "flotte_F2_Q150"))["mean"], 1) == 7.9
    assert round(R.p_gain(R.find_cell(DATA, Q=150.0), 12, 1.0)["mean"], 1) == -26.3 and round(R.p_gain(R.by_name(DATA, "flotte_F2_Q150"), 12, 1.0)["mean"], 1) == -30.0
    assert DATA["_meta"]["spearman"]["n"] == 23 and round(DATA["_meta"]["spearman"]["rho"], 2) == 0.70
    d = DATA["_meta"]["base_distribution"]
    assert (round(d["min"], 1), round(d["q1"], 1), round(d["median"], 1), round(d["q3"], 1), round(d["p90"], 1), round(d["max"], 1), d["loss_share"]) == (10.5, 15.1, 16.9, 18.7, 20.6, 24.8, 0.0)


def test_findings_that_the_texts_rely_on_hold_in_the_data():
    assert R.negative_summary(DATA) == (46, 756, 44, 0) and DATA["_meta"]["n_negative_pairs"] == [46, 756] and DATA["_meta"]["E_negative_pairs"] == [75, 81]
    positive_e = [(c["name"], k) for c in R.cells(DATA) for k, v in c["E"].items() if v["mean"] >= 0]
    assert sorted(positive_e) == sorted([("flotte_F1_Q250", "E_L1"), ("flotte_F1_Q250", "E_L2"), ("flotte_F1_Q250", "E_L3"), ("flotte_F1_Q300", "E_L1"),
                                          ("flotte_F2_Q150", "E_L1"), ("pen100", "E_L1")])
    gs = R.grid_summary(BASE)
    assert (gs["n"], gs["positive"], gs["negative"], gs["near_best"], gs["best_key"]) == (28, 26, 2, 5, "P_H8_g0.25")
    assert BASE["deferred_R"] == 0.0 and R.by_name(DATA, "flotte_F1_Q300")["deferred_R"] > 100
    assert R.negative_pairs(DATA)[0][0]["name"] == "flotte_F2_Q150" and R.negative_pairs(DATA)[0][1] == "P_H12_g1.00"


def test_no_column_of_the_results_is_constant_or_zero():
    """Nullspalten-Signal: kein Kennwert ist in allen Zellen gleich."""
    for key in ("tau", "JR", "Q_over_qL", "short_R", "short_P", "routes_R", "routes_P", "visits_P", "opt_visits_P", "util_R", "util_P", "km_R", "km_P", "cv_gain"):
        assert len({round(c[key], 6) for c in R.cells(DATA)}) > 5, key
    for pol in ("P_H1_g0.10", "P_H12_g1.00", "E_L1"):
        vals = {round((c["P"] if pol.startswith("P") else c["E"])[pol]["mean"], 4) for c in R.cells(DATA)}
        assert len(vals) > 20, pol


# ---------------------------------------------------------------------------------------------------
# Urteil in drei Zuständen
# ---------------------------------------------------------------------------------------------------
def test_verdict_needs_more_than_two_standard_errors():
    assert R.verdict(2.01, 1.0) == "pos" and R.verdict(2.0, 1.0) == "none" and R.verdict(-2.0, 1.0) == "none" and R.verdict(-2.01, 1.0) == "neg" and R.verdict(0.0, 1.0) == "none"
    assert R.verdict(3.0, 1.0, factor=3.0) == "none" and R.verdict(3.01, 1.0, factor=3.0) == "pos"


def test_state_needs_five_percent_and_two_standard_errors_for_wins_and_minus_two_se_for_losses():
    assert R.state_of(5.0, 1.0) == C.STATE_LOHNT and R.state_of(4.99, 1.0) == C.STATE_WENIG
    assert R.state_of(5.0, 2.5) == C.STATE_WENIG and R.state_of(5.0, 2.49) == C.STATE_LOHNT           # genau 2 SE: nicht belastbar
    assert R.state_of(-2.01, 1.0) == C.STATE_TEURER and R.state_of(-2.0, 1.0) == C.STATE_WENIG and R.state_of(-0.5, 1.0) == C.STATE_WENIG and R.state_of(0.0, 0.0) == C.STATE_WENIG
    assert R.state_of(1.0, 0.1) == C.STATE_WENIG and R.state_of(-30.0, 0.7) == C.STATE_TEURER and R.state_of(100.0, 1.0) == C.STATE_LOHNT
    assert C.WIN_MIN_PCT == 5.0 and C.SE_FACTOR == 2.0


def test_judge_gives_the_state_and_the_measured_gain():
    j = R.judge(BASE, 3, 0.5)
    assert j["state"] == C.STATE_LOHNT and j["verdict"] == "pos" and j["mean"] == dflt(BASE)["mean"]
    assert R.judge(R.find_cell(DATA, Q=100.0), 3, 0.5)["state"] == C.STATE_WENIG
    assert R.judge(R.find_cell(DATA, Q=150.0), 12, 1.0)["state"] == C.STATE_TEURER and R.judge(R.find_cell(DATA, Q=150.0), 12, 0.25)["state"] == C.STATE_LOHNT


def test_all_three_states_occur_in_the_data_and_the_text_is_defined():
    states = {R.judge(c, h, g)["state"] for c in R.cells(DATA) for h in C.HORIZON_OPTIONS for g in C.GAMMA_OPTIONS}
    assert states == {C.STATE_LOHNT, C.STATE_WENIG, C.STATE_TEURER} and set(R.STATE_TEXT) == states
    assert R.STATE_TEXT[C.STATE_TEURER] == "Diese Mitnahmeregel ist hier teurer als reaktiv" and R.STATE_TEXT[C.STATE_LOHNT] == "Bündeln lohnt"


# ---------------------------------------------------------------------------------------------------
# Zellen finden, beschriften
# ---------------------------------------------------------------------------------------------------
def test_find_cell_and_by_name():
    assert R.find_cell(DATA)["name"] == "base" and R.find_cell(DATA, Q=600.0)["name"] == "q600" and R.find_cell(DATA, N=40, Q=600.0)["name"] == "n40_q600"
    assert R.find_cell(DATA, F=1)["name"] == "flotte_F1_Q300" and R.find_cell(DATA, sigma=0.15)["name"] == "sigma015" and R.find_cell(DATA, pfac=1.0)["name"] == "pen100"
    with pytest.raises(KeyError):
        R.find_cell(DATA, Q=123.0)
    with pytest.raises(KeyError):
        R.by_name(DATA, "gibt es nicht")


def test_keys_of_the_grid_and_the_early_rule():
    assert R.p_key(3, 0.5) == "P_H3_g0.50" and R.p_key(12, 1.0) == "P_H12_g1.00" and R.p_key(1, 0.1) == "P_H1_g0.10" and R.p_key(8, 0.25) == "P_H8_g0.25" and R.e_key(2) == "E_L2"
    assert R.parse_p_key("P_H8_g0.25") == (8, 0.25) and R.parse_p_key("P_H12_g1.00") == (12, 1.0)
    assert R.e_gain(BASE, 3) is BASE["E"]["E_L3"] and R.p_gain(BASE, 8, 0.25) is BASE["P"]["P_H8_g0.25"]


def test_cell_labels_groups_and_setting_texts():
    labels = {c["name"]: R.cell_label(c) for c in R.cells(DATA)}
    assert labels["base"] == "Basisfall" and labels["q100"] == "Wagen 100" and labels["n40"] == "40 Kunden" and labels["n40_q600"] == "40 Kunden, Wagen 600"
    assert labels["flotte_F1_Q250"] == "Wagen 250, 1 Tour je Tag" and labels["flotte_F2_Q150"] == "Wagen 150, 2 Touren je Tag" and labels["sigma015"] == "Schwankung 0,15"
    assert labels["pen025"] == "Strafe 25" and labels["kappa0"] == "Meldegrenze κ = 0" and labels["cluster"] == "geklumpte Kunden" and labels["depot_far"] == "Depot weit außen"
    assert labels["depot_corner"] == "Depot in der Ecke" and labels["tank_kurz"] == "Tank 4 bis 7 Tage" and labels["tank_lang"] == "Tank 14 bis 24 Tage"
    assert len(set(labels.values())) == 27
    groups = {c["name"]: R.cell_group(c) for c in R.cells(DATA)}
    assert groups["base"] == "Basisfall" and groups["sigma06"] == "Verbrauchsschwankung" and groups["q450"] == "Wagenkapazität" and groups["n10"] == "Dichte" and groups["n40_q600"] == "Dichte"
    assert groups["cluster"] == groups["depot_corner"] == "Geometrie" and groups["pen001"] == "Fehlmengenstrafe" and groups["flotte_F2_Q150"] == "Flotte"
    assert groups["kappa2"] == "Meldegrenze" and groups["tank_lang"] == "Tankgröße"
    assert R.cell_setting_text(BASE) == "20 Kunden, Wagen 300, 6 Touren je Tag, Verbrauchsschwankung 0,3, Strafe 5"
    assert R.cell_setting_text(R.by_name(DATA, "cluster")) == "geklumpte Kunden" and R.setting_text((10, 100, 1, 0.0, 100)) == "10 Kunden, Wagen 100, 1 Tour je Tag, Verbrauchsschwankung 0,0, Strafe 100"
    assert R.setting_text((10, 100, 2, 0.0, 100)).startswith("10 Kunden, Wagen 100, 2 Touren je Tag,")


def test_cfg_setting_marks_the_cells_that_lie_on_the_control_stages():
    assert R.cfg_setting(BASE["cfg"]) == (20, 300, 6, 0.3, 5) and R.cfg_setting(R.by_name(DATA, "n40_q600")["cfg"]) == (40, 600, 6, 0.3, 5)
    assert R.cfg_setting(R.by_name(DATA, "pen001")["cfg"]) == (20, 300, 6, 0.3, 1) and R.cfg_setting(R.by_name(DATA, "flotte_F2_Q150")["cfg"]) == (20, 150, 2, 0.3, 5)
    for name in ("cluster", "depot_far", "depot_corner", "kappa0", "kappa2", "tank_kurz", "tank_lang", "flotte_F1_Q250"):
        assert R.cfg_setting(R.by_name(DATA, name)["cfg"]) is None, name              # nicht auf den Reglerstufen (Geometrie, Meldegrenze, Tank, Wagen 250)
    fake = copy.deepcopy(BASE["cfg"])
    fake["penalty"] = 7.0
    assert R.cfg_setting(fake) is None
    fake["penalty"], fake["N"] = 5.0, 25
    assert R.cfg_setting(fake) is None


# ---------------------------------------------------------------------------------------------------
# Zell-Zuordnung
# ---------------------------------------------------------------------------------------------------
def test_assignable_cells_are_19_and_each_is_its_own_nearest_cell():
    pairs = R.assignable_cells(DATA)
    assert len(pairs) == 19 and [c["name"] for c, _ in pairs][0] == "base"
    for cell, setting in pairs:
        got, exact, diff = R.nearest_cell(DATA, *setting)
        assert got["name"] == cell["name"] and exact and diff == []


def test_nearest_cell_by_stage_distance_with_the_documented_tie_break():
    n = lambda *s: R.nearest_cell(DATA, *s)
    cell, exact, diff = n(20, 300, 2, 0.3, 5)                      # nur die Flotte weicht ab: Basis
    assert cell["name"] == "base" and not exact and diff == ["fleet"]
    cell, exact, diff = n(40, 100, 6, 0.3, 5)                      # Q = 100 (Abstand 1: Kunden) statt n40 (Abstand 3: Wagen)
    assert cell["name"] == "q100" and diff == ["customers"]
    cell, _, diff = n(10, 200, 6, 0.3, 5)                          # Gleichstand q200 (Kunden) gegen n10 (Wagen): der Wagen steht in der Rangfolge vorn
    assert cell["name"] == "q200" and diff == ["customers"]
    cell, _, diff = n(20, 300, 6, 0.6, 25)                         # sigma und Strafe weichen ab (Abstand 1 zu sigma06 und pen025): sigma steht vor der Strafe
    assert cell["name"] == "sigma06" and diff == ["penalty"]
    cell, _, diff = n(20, 1000, 1, 0.3, 5)                         # Q = 1000 und 1 Tour: q1000 (Abstand 1) vor flotte_F1_Q300 (Abstand 3)
    assert cell["name"] == "q1000" and diff == ["fleet"]
    assert R.PARAM_PRIORITY == ("capacity", "fleet", "customers", "sigma", "penalty")


def test_every_combination_of_the_controls_gets_a_measured_cell_and_the_coverage_is_known():
    cov = R.coverage(DATA)
    assert cov["total"] == 3 * 7 * 3 * 5 * 4 == 1260 and cov["exact"] == 19 and sum(cov["by_distance"].values()) == 1260 and cov["cells"] == 27
    assert cov["by_distance"] == {0: 19, 1: 127, 2: 374, 3: 500, 4: 240}
    assert len(cov["assignable"]) == 19 and sorted(cov["unreachable"]) == sorted(["cluster", "depot_far", "depot_corner", "flotte_F1_Q250", "kappa0", "kappa2", "tank_kurz", "tank_lang"])
    for combo in itertools.product(C.CUSTOMER_OPTIONS, C.CAPACITY_OPTIONS, C.FLEET_OPTIONS, C.SIGMA_OPTIONS, C.PENALTY_OPTIONS):
        cell, exact, diff = R.nearest_cell(DATA, *combo)
        assert (R.cfg_setting(cell["cfg"]) == combo) == exact and (exact == (diff == []))


def test_assignment_note_is_empty_for_a_measured_cell_and_explains_otherwise():
    cell, exact, diff = R.nearest_cell(DATA, 40, 100, 6, 0.3, 5)
    assert R.assignment_note(40, 100, 6, 0.3, 5, cell, []) == ""
    note = R.assignment_note(40, 100, 6, 0.3, 5, cell, diff)
    assert "keine eigene Messreihe" in note and "variiert je Zelle nur einen Parameter" in note and "nächstliegende gemessene Zelle" in note
    assert "40 Kunden, Wagen 100, 6 Touren je Tag" in note and "20 Kunden, Wagen 100" in note and "abweichend: Kunden." in note
    cell2, _, diff2 = R.nearest_cell(DATA, 10, 100, 2, 0.6, 25)
    assert "abweichend: Touren je Tag, Kunden, Verbrauchsschwankung, Fehlmengenstrafe." in R.assignment_note(10, 100, 2, 0.6, 25, cell2, diff2)


# ---------------------------------------------------------------------------------------------------
# Auswertungen der Kernabschnitte
# ---------------------------------------------------------------------------------------------------
def test_heat_matrix_has_gamma_rows_and_horizon_columns():
    z = R.heat_matrix(BASE)
    assert len(z) == 4 and all(len(r) == 7 for r in z)
    assert z[2][2] == dflt(BASE)["mean"] and z[1][5] == R.p_gain(BASE, 8, 0.25)["mean"] and z[3][6] == R.p_gain(BASE, 12, 1.0)["mean"] and z[0][0] == R.p_gain(BASE, 1, 0.1)["mean"]


def test_factor_cells_are_sorted_and_contain_the_base():
    assert [c["cfg"]["Q"] for c in R.factor_cells(DATA, "capacity")] == [100.0, 150.0, 200.0, 300.0, 450.0, 600.0, 1000.0]
    assert [c["cfg"]["sigma"] for c in R.factor_cells(DATA, "sigma")] == [0.0, 0.15, 0.3, 0.6, 1.0]
    assert [c["cfg"]["penalty"] for c in R.factor_cells(DATA, "penalty")] == [1.0, 5.0, 25.0, 100.0]
    with pytest.raises(KeyError):
        R.factor_cells(DATA, "gibt es nicht")


def test_capacity_rows_follow_the_lever_and_carry_the_state():
    rows = R.capacity_rows(DATA, 3, 0.5)
    assert [r["Q"] for r in rows] == [100, 150, 200, 300, 450, 600, 1000] and [r["gain"] for r in rows] == sorted(r["gain"] for r in rows)
    assert [round(r["Q_over_q"], 1) for r in rows] == [1.1, 1.5, 2.0, 3.0, 4.5, 6.0, 10.1] and rows[0]["state"] == C.STATE_WENIG and rows[3]["state"] == C.STATE_LOHNT
    assert rows[3]["gain"] == dflt(BASE)["mean"] and rows[3]["cv_gain"] == BASE["cv_gain"] and rows[3]["routes_R"] == BASE["routes_R"] and rows[0]["name"] == "q100"
    generous = R.capacity_rows(DATA, 12, 1.0)
    assert [r["state"] for r in generous][:3] == [C.STATE_TEURER] * 3 and generous[1]["gain"] == pytest.approx(-26.3, abs=0.05)


def test_uncertainty_rows_for_sigma_and_penalty():
    u = R.uncertainty_rows(DATA, 3, 0.5)
    assert [r["level"] for r in u["sigma"]] == [0.0, 0.15, 0.3, 0.6, 1.0] and [r["level"] for r in u["penalty"]] == [1.0, 5.0, 25.0, 100.0]
    assert max(r["gain"] for r in u["sigma"]) - min(r["gain"] for r in u["sigma"]) < 1.0 and all(r["state"] == C.STATE_LOHNT for r in u["sigma"] + u["penalty"])


def test_rule_rows_list_early_reactive_selected_and_best():
    rows = R.rule_rows(BASE, 8, 0.25, 3)
    assert [r[0] for r in rows] == ["Früher liefern, 1 Tag", "Früher liefern, 2 Tage", "Früher liefern, 3 Tage", "Reaktiv (Bezug)", "Bündeln P(8; 0,25)", "Beste Zelle, kreuzvalidiert"]
    assert [r[1] for r in rows][3] == 0.0 and rows[4][1] == R.p_gain(BASE, 8, 0.25)["mean"] and rows[5] == ("Beste Zelle, kreuzvalidiert", BASE["cv_gain"], None) and rows[0][1] < 0 < rows[4][1]


def test_mechanism_of_the_standard_rule():
    m = R.mechanism(BASE)
    assert m["routes"] == (BASE["routes_R"], BASE["routes_P"], BASE["routes_E"]) and m["util"][1] > m["util"][0] and m["km"][1] < m["km"][0] < m["km"][2] and m["visits"][1] > m["visits"][0]
    assert m["short"] == (BASE["short_R"], BASE["short_P"], BASE["short_E"]) and m["opt_visits"] == BASE["opt_visits_P"] and m["stockout_days"] == (BASE["stockout_R"], BASE["stockout_P"], BASE["stockout_E"])
    assert m["visits"][2] > m["visits"][0] and m["routes"][2] > m["routes"][0] > m["routes"][1] and m["short"][2] == 0.0 and R.by_name(DATA, "flotte_F1_Q300")["short_E"] > 0


def test_regime_rows_for_the_standard_rule_and_for_a_generous_one():
    rows = R.regime_rows(DATA, 3, 0.5)
    assert len(rows) == 27 and rows[0]["name"] == "base" and rows[0]["group"] == "Basisfall" and rows[0]["label"] == "Basisfall" and rows[0]["cell"] is BASE
    states = [r["state"] for r in rows]
    assert states.count(C.STATE_LOHNT) == 26 and states.count(C.STATE_WENIG) == 1 and [r["name"] for r in rows if r["state"] == C.STATE_WENIG] == ["q100"]
    gen = R.regime_rows(DATA, 12, 1.0)
    assert [r["state"] for r in gen].count(C.STATE_TEURER) >= 10 and next(r for r in gen if r["name"] == "q150")["gain"] == pytest.approx(-26.3, abs=0.05)
    r0 = rows[0]
    assert r0["median"] == dflt(BASE)["median"] and r0["q1"] == dflt(BASE)["q1"] and r0["q3"] == dflt(BASE)["q3"] and r0["share_loss"] == 0.0 and r0["verdict"] == "pos"


def test_negative_pairs_are_sorted_worst_first():
    neg = R.negative_pairs(DATA)
    assert len(neg) == 46 and [t[2] for t in neg] == sorted(t[2] for t in neg) and all(t[2] < 0 for t in neg) and neg[0][2] == pytest.approx(-30.0, abs=0.05)


def test_shortage_rows_of_the_tight_fleets():
    rows = R.shortage_rows(DATA)
    assert [r["cell"]["name"] for r in rows] == ["flotte_F1_Q300", "flotte_F1_Q250", "flotte_F2_Q150"]
    assert [round(r["short_R"]) for r in rows] == [386, 1973, 511] and [round(r["short_P"]) for r in rows] == [130, 639, 353]
    assert all(r["short_P"] < r["short_R"] and r["gain"] > 5 for r in rows) and rows[0]["cmp"]["more"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------------------------------
# Exakter Maßstab
# ---------------------------------------------------------------------------------------------------
def test_oracle_summary_numbers():
    s150, s250 = R.oracle_summary(DATA, 150), R.oracle_summary(DATA, 250)
    assert s150["n"] == s250["n"] == 30 and s150["n_optimal"] == s250["n_optimal"] == 30
    assert [round(s["gap"]["R"]["mean"], 1) for s in (s150, s250)] == [28.7, 32.3] and [round(s["gap"]["P_H3_g0.50"]["mean"], 1) for s in (s150, s250)] == [12.7, 11.5]
    assert [round(s["gap"]["P_H8_g0.25"]["mean"], 1) for s in (s150, s250)] == [14.7, 14.0] and [round(s["gap"]["E_L1"]["mean"], 0) for s in (s150, s250)] == [68.0, 75.0]
    assert [round(100 * s["closed_P3"]) for s in (s150, s250)] == [48, 61] and [round(s["gain_P3"]["mean"], 1) for s in (s150, s250)] == [11.8, 15.2]
    assert [round(s["oracle_gain"]["mean"], 1) for s in (s150, s250)] == [21.7, 23.8]
    assert {k: round(v, 2) for k, v in s150["routes"].items()} == {"optimum": 3.03, "R": 4.67, "P3": 3.70, "P8": 4.20} and round(s250["routes"]["optimum"], 2) == 2.53     # ERGEBNIS.md nennt 2,50: der Bericht der Messreihe und die Neurechnung ergeben 2,53
    assert round(s250["routes"]["R"], 2) == 4.67 and round(s250["routes"]["P3"], 2) == 3.37 and 0 < s150["time_max"] < 10
    assert [k for k, _ in R.ORACLE_POLICIES] == ["R", "P_H3_g0.50", "P_H8_g0.25", "E_L1"]


def test_oracle_rows_are_30_instances_with_proven_optima_and_no_regime_below_the_optimum():
    for wagon in (150, 250):
        rows = R.oracle_rows(DATA, wagon)
        assert [r["seed"] for r in rows] == list(range(30)) and all(r["status"] == "OPTIMAL" for r in rows) and R.oracle_agg(DATA, wagon)["n_optimal"] == 30
        for r in rows:
            assert r["oracle_short"] == 0.0
            for k in ("R", "P_H3_g0.50", "P_H8_g0.25", "E_L1"):
                if r[k + "_short"] == 0.0:
                    assert r[k] >= r["J_oracle"] - 1e-6, (wagon, r["seed"], k)          # keine Regel ohne Fehlmenge unter dem bewiesenen Optimum
    with pytest.raises(KeyError):
        R.oracle_rows(DATA, 200)


def test_load_results_is_cached_and_reads_the_committed_file():
    assert R.load_results() is DATA and R.DATA_PATH.name == "irp_results.json" and R.DATA_PATH.parent.name == "data"
