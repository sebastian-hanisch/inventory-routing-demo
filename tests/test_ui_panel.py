"""Panel (irp_ui_panel): Kennzahlen, Meldung in drei Zuständen, Tabellen der Kernabschnitte. Die Tabellen sind reine Funktionen (pandas) und werden hier direkt geprüft; das
Zeichnen (Streamlit) prüft tests/test_app.py. Die Live-Instanz ist die eingefrorene Basisinstanz."""
import pytest

import irp_constants as C
import irp_frozen as FZ
import irp_live as LV
import irp_results as R
import irp_ui_panel as UI

DATA = R.load_results()
BASE = R.find_cell(DATA)


@pytest.fixture(scope="module")
def live():
    inst = FZ.frozen("base", 0)
    original = LV.make_live_instance
    LV.make_live_instance = lambda *a, **k: inst
    try:
        return LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)
    finally:
        LV.make_live_instance = original


class Col:
    """Ersatz für eine Spalte: merkt sich die Aufrufe von metric()."""
    def __init__(self):
        self.calls = []

    def metric(self, label, value, **kw):
        self.calls.append((label, value, kw))


def test_rule_label_and_utilization():
    assert UI.rule_label("R", 3, 0.5, 2) == "Reaktiv" and UI.rule_label("P", 3, 0.5, 2) == "Bündeln P(3; 0,5)" and UI.rule_label("E", 3, 0.5, 2) == "Früher liefern E(2)"
    assert UI.rule_label("P", 12.0, 0.25, 1.0) == "Bündeln P(12; 0,25)" and UI.rule_label("P", 1, 1.0, 1) == "Bündeln P(1; 1,0)"
    assert UI.utilization({"util_sum": 1.5, "n_routes": 3}) == 0.5 and UI.utilization({"util_sum": 0.0, "n_routes": 0}) == 0.0


def test_live_numbers_per_rule(live):
    x = UI.live_numbers(live)
    assert set(x) == {"R", "P", "E"} and x["R"]["diff"] == 0.0 and x["R"]["gain"] == 0.0
    for name in ("P", "E"):
        assert x[name]["diff"] == pytest.approx(live["rules"][name]["J"] - live["rules"]["R"]["J"]) and x[name]["gain"] == live["rules"][name]["gain"]
    r = live["rules"]["P"]["res"]
    assert x["P"]["routes"] == r["n_routes"] and x["P"]["visits"] == r["n_visits"] and x["P"]["picks"] == r["opt_visits"] and x["P"]["km"] == r["routing"] and x["P"]["short"] == r["short"]
    assert x["P"]["util"] == pytest.approx(r["util_sum"] / r["n_routes"]) and x["P"]["diff"] < 0 < x["E"]["diff"]


def test_metrics_are_four_labelled_cards_and_the_deltas_read_as_my_value_minus_the_reference(live):
    cols = [Col() for _ in range(4)]
    UI.render_metrics(cols, live)
    (l0, v0, k0), (l1, v1, k1), (l2, v2, k2), (l3, v3, k3) = (c.calls[0] for c in cols)
    x = UI.live_numbers(live)
    assert l0 == "Kosten reaktiv (R)" and l1 == "Kosten Bündeln P(3; 0,5)" and l2 == "Kosten Früher liefern E(2)" and l3 == "Touren in 120 Tagen (R / P / E)"
    fmt = lambda v: f"{round(v):,}".replace(",", ".")
    assert v0 == fmt(x["R"]["J"]) and v1 == fmt(x["P"]["J"]) and v2 == fmt(x["E"]["J"])
    assert v3 == f"{x['R']['routes']} / {x['P']['routes']} / {x['E']['routes']}"
    assert k1["delta"].startswith("-") and "(+" in k1["delta"] and k1["delta_color"] == "inverse"                                                  # Bündeln billiger: negatives Delta, grün
    assert k2["delta"].startswith("+") and "(-" in k2["delta"] and k2["delta_color"] == "inverse"                                                  # früher liefern teurer: positives Delta, rot
    assert k1["delta"].split(" ")[0] == f"{x['P']['diff']:+,.0f}".replace(",", ".") and k2["delta"].split(" ")[0] == f"{x['E']['diff']:+,.0f}".replace(",", ".")
    assert k0["delta_color"] == "off" and k3["delta_color"] == "off" and k0["delta_arrow"] == "off" and k3["delta_arrow"] == "off"
    assert k3["delta"] == f"Auslastung {x['R']['util']:.2f} / {x['P']['util']:.2f} / {x['E']['util']:.2f}".replace(".", ",") and all("help" in k and len(k["help"]) > 40 for k in (k0, k1, k2, k3))
    assert f"({x['P']['gain']:+.1f} %)".replace(".", ",") in k1["delta"] and f"({x['E']['gain']:+.1f} %)".replace(".", ",") in k2["delta"] and "Fehlmenge" in k3["help"]


def test_shortage_caption_states_shortage_visits_and_pickups(live):
    x = UI.live_numbers(live)
    text = UI.shortage_caption(live)
    assert text == (f"Fehlmenge in 120 Tagen (Einheiten, R / P / E): {x['R']['short']:.0f} / {x['P']['short']:.0f} / {x['E']['short']:.0f}; "
                    f"Besuche: {x['R']['visits']} / {x['P']['visits']} / {x['E']['visits']}, davon Mitnahmen beim Bündeln: {x['P']['picks']}.")


def test_metric_labels_follow_the_selected_rule_parameters(live):
    cols = [Col() for _ in range(4)]
    UI.render_metrics(cols, dict(live, H=12.0, gamma=0.25, early=3.0))
    assert cols[1].calls[0][0] == "Kosten Bündeln P(12; 0,25)" and cols[2].calls[0][0] == "Kosten Früher liefern E(3)"


def test_distribution_sentence_gives_mean_median_quartiles_and_loss_share():
    s = UI.distribution_sentence(BASE, 3, 0.5)
    assert "P(3; 0,5)" in s and "Mittel +17,0 %" in s and "Median +16,9 %" in s and "[+15,1; +18,7]" in s and "in 0 % der Instanzen ein Verlust" in s and "200 Instanzen" in s
    s2 = UI.distribution_sentence(R.find_cell(DATA, Q=150.0), 12, 1.0)
    assert "P(12; 1,0)" in s2 and "Mittel -26,3 %" in s2 and "in 100 % der Instanzen ein Verlust" in s2


def test_message_lohnt(live):
    state, text = UI.message(live, BASE)
    assert state == C.STATE_LOHNT and text.startswith("✅ Bündeln lohnt: P(3; 0,5) spart laut Messreihe (") and "20 Kunden, Wagen 300, 6 Touren je Tag" in text
    assert "200 Instanzen) 17,0 ± 0,3 % Fahrkosten gegenüber reaktivem Nachliefern (Median 16,9 %, in 0 % der Instanzen ein Verlust)" in text
    assert (f"Auf dieser einen Instanz: {live['rules']['P']['gain']:+.1f} %".replace(".", ",") + ".") in text and text.endswith("%.")


def test_message_wenig(live):
    q100 = R.find_cell(DATA, Q=100.0)
    state, text = UI.message(live, q100)
    assert state == C.STATE_WENIG and text.startswith("ℹ️ Bündeln bringt hier wenig: P(3; 0,5) liegt laut Messreihe (") and "Wagen 100" in text
    assert "bei +0,4 ± 0,1 %" in text and "unter 5 % oder nicht von 0 zu unterscheiden" in text and "in 1 % der Instanzen ein Verlust" in text


def test_message_teurer(live):
    state, text = UI.message(dict(live, H=12.0, gamma=1.0), R.find_cell(DATA, Q=150.0))
    assert state == C.STATE_TEURER and text.startswith("⚠️ Diese Mitnahmeregel ist hier teurer als reaktiv: P(12; 1,0) kostet laut Messreihe (")
    assert "26,3 ± 0,5 % mehr als reaktives Nachliefern" in text and "in 100 % der Instanzen ein Verlust" in text and "kleinere Mitnahmeschwelle γ" in text and "-26" not in text


def test_message_appends_the_assignment_note(live):
    _, text = UI.message(live, BASE, "Hinweis: nächstliegende Zelle.")
    assert text.endswith(" Hinweis: nächstliegende Zelle.")
    assert "Hinweis" not in UI.message(live, BASE)[1]


def test_comparison_table_puts_the_instance_next_to_the_measurement(live):
    df = UI.comparison_table(live, BASE)
    assert list(df.columns) == ["Kennzahl", "Diese Instanz (Seed 0, 1 Instanz)", "Messreihe (200 Instanzen)"] and len(df) == 3
    assert df.iloc[0]["Kennzahl"] == "Gewinn Bündeln P(3; 0,5) (%)" and df.iloc[1]["Kennzahl"] == "Median [Q1; Q3] der Einzelgewinne Bündeln P(3; 0,5) (%)" and df.iloc[0]["Messreihe (200 Instanzen)"] == "+17,0 ± 0,3"
    assert df.iloc[0]["Diese Instanz (Seed 0, 1 Instanz)"] == f"{live['rules']['P']['gain']:+.1f}".replace(".", ",")
    assert df.iloc[1]["Diese Instanz (Seed 0, 1 Instanz)"] == "eine Instanz" and df.iloc[1]["Messreihe (200 Instanzen)"] == "+16,9 [+15,1; +18,7]"
    assert df.iloc[2]["Messreihe (200 Instanzen)"] == "-16,3 ± 0,2" and df.iloc[2]["Kennzahl"] == "Gewinn Früher liefern E(2) (%)"


def test_rule_table_has_one_row_per_rule(live):
    df = UI.rule_table(live, BASE)
    assert list(df["Regel"]) == ["Reaktiv", "Bündeln P(3; 0,5)", "Früher liefern E(2)"] and list(df["Gewinn Messreihe (%)"]) == ["Bezug", "+17,0 ± 0,3", "-16,3 ± 0,2"]
    x = UI.live_numbers(live)
    assert list(df["Touren"]) == [x[r]["routes"] for r in "RPE"] and list(df["Mitnahmen"]) == [0, x["P"]["picks"], 0] and df.iloc[0]["Gewinn dieser Instanz (%)"] == 0.0
    assert list(df["Kosten"]) == [round(x[r]["J"]) for r in "RPE"] and list(df["Besuche"]) == [x[r]["visits"] for r in "RPE"]
    assert df.iloc[1]["Auslastung"] == round(x["P"]["util"], 2) and df.iloc[1]["Fehlmenge (Einheiten)"] == round(x["P"]["short"]) and df.iloc[1]["Fahr-km"] == round(x["P"]["km"])


def test_mechanism_frame_of_the_base_cell():
    df = UI.mechanism_frame(BASE)
    rows = {r["Kennzahl (je 120 Tage)"]: (r["Reaktiv"], r["Bündeln P(3; 0,5)"], r["Früher liefern E(2)"]) for _, r in df.iterrows()}
    assert list(df.columns) == ["Kennzahl (je 120 Tage)", "Reaktiv", "Bündeln P(3; 0,5)", "Früher liefern E(2)"]
    assert rows["Touren"] == ("132,8", "99,7", "137,0") and rows["Besuche"] == ("238,1", "260,5", "303,6") and rows["davon Mitnahmen"] == ("0", "101,7", "0")
    assert rows["mittlere Wagenauslastung"] == ("0,59", "0,79", "0,58") and rows["Fahr-km"] == ("14.694", "12.220", "17.341") and rows["Fehlmenge (Einheiten)"] == ("14,4", "8,5", "0,0")
    assert rows["Kunden-Tage mit Fehlmenge"] == ("7,4", "4,5", "0,0")


def test_halteliste_lists_all_stops_of_all_tours(live):
    d = next(d for d in LV.pickup_days(live, 20) if len(LV.day_view(live, "P", d)["tours"]) >= 1)
    view = LV.day_view(live, "P", d)
    df = UI.halteliste_frame(view)
    assert list(df.columns) == ["Tour", "Nr.", "Kunde", "Art", "Bestand vorher", "Füllstand vorher", "Reichweite (Tage)", "Liefermenge"] and len(df) == view["n_stops"]
    assert set(df["Art"]) <= {"fällig", "Mitnahme"} and (df["Art"] == "Mitnahme").sum() == view["n_pickups"] and df["Nr."].iloc[0] == 1
    assert list(df["Liefermenge"])[0] == round(view["tours"][0]["stops"][0]["qty"]) and df["Füllstand vorher"].iloc[0].endswith(" %")
    empty = next(dd for dd in range(1, 121) if not LV.day_view(live, "R", dd)["tours"])
    e = UI.halteliste_frame(LV.day_view(live, "R", empty))
    assert len(e) == 0 and list(e.columns) == list(df.columns)


def test_capacity_frame_marks_the_set_wagon():
    df = UI.capacity_frame(R.capacity_rows(DATA, 3, 0.5), 300)
    assert len(df) == 7 and list(df["Wagenkapazität Q"])[3] == "300  ◀ eingestellt" and list(df["Wagenkapazität Q"])[0] == "100" and df.iloc[0]["Q/q (Lieferungen je Wagen)"] == "1,1"
    assert df.iloc[3]["Gewinn Bündeln (Mittel ± SE, %)"] == "+17,0 ± 0,3" and df.iloc[3]["Touren R → P (Standardregel)"] == "132,8 → 99,7" and df.iloc[3]["beste Zelle, kreuzvalidiert (%)"] == "+20,5"
    assert df.iloc[3]["Auslastung R → P"] == "0,59 → 0,79"


def test_uncertainty_and_rule_frames():
    u = R.uncertainty_rows(DATA, 3, 0.5)
    df = UI.uncertainty_frame(u["sigma"], "Verbrauchsschwankung", fmt=lambda v: f"{v}")
    assert list(df.columns) == ["Verbrauchsschwankung", "Gewinn Bündeln (Mittel ± SE, %)"] and list(df["Verbrauchsschwankung"]) == ["0.0", "0.15", "0.3", "0.6", "1.0"] and df.iloc[0]["Gewinn Bündeln (Mittel ± SE, %)"] == "+16,5 ± 0,3"
    rules = UI.rule_rows_frame(R.rule_rows(BASE, 8, 0.25, 3))
    assert list(rules["Gewinn gegenüber reaktiv (%)"])[3] == "Bezug" and list(rules["Gewinn gegenüber reaktiv (%)"])[0] == "-7,0 ± 0,2" and list(rules["Gewinn gegenüber reaktiv (%)"])[5].startswith("+20,")
    assert "±" not in list(rules["Gewinn gegenüber reaktiv (%)"])[5]                                                # die beste Zelle hat keinen Standardfehler


def test_shortage_frame_of_the_tight_fleets():
    df = UI.shortage_frame(R.shortage_rows(DATA))
    assert list(df["Fehlmenge reaktiv → Bündeln (Einheiten)"]) == ["386 → 130", "1973 → 639", "511 → 353"] and list(df["Instanzen mit mehr Fehlmenge"])[0] == "1,0 %"
    assert df.iloc[0]["Gewinn Standardregel (%)"] == "+19,2 ± 0,3"


def test_regime_frame_covers_all_cells_and_marks_the_chosen_one():
    rows = R.regime_rows(DATA, 3, 0.5)
    df = UI.regime_frame(rows, "q600")
    assert len(df) == 27 and list(df["Gruppe"])[0] == "Basisfall" and sum(1 for z in df["Zelle"] if z.startswith("▶ ")) == 1 and next(z for z in df["Zelle"] if z.startswith("▶ ")) == "▶ Wagen 600"
    assert df[df["Zelle"] == "Wagen 100"]["Urteil"].iloc[0] == "Bündeln bringt hier wenig" and df[df["Zelle"] == "Basisfall"]["Urteil"].iloc[0] == "Bündeln lohnt"
    assert df.iloc[0]["Median [Q1; Q3]"] == "+16,9 [+15,1; +18,7]" and df.iloc[0]["Anteil Verlust"] == "0 %" and df.iloc[0]["Q/q"] == "3,0"
    gen = UI.regime_frame(R.regime_rows(DATA, 12, 1.0))
    assert "Diese Mitnahmeregel ist hier teurer als reaktiv" in set(gen["Urteil"]) and not any(z.startswith("▶") for z in gen["Zelle"])


def test_negative_frame_lists_the_worst_cells_first():
    df = UI.negative_frame(DATA)
    assert len(df) == 12 and df.iloc[0]["Zelle"] == "Wagen 150, 2 Touren je Tag" and df.iloc[0]["Regel"] == "P(12; 1,0)" and df.iloc[0]["Gewinn (Mittel ± SE, %)"] == "-30,0 ± 0,7"
    assert len(UI.negative_frame(DATA, limit=3)) == 3 and all(r.startswith("P(") for r in df["Regel"])


def test_oracle_frames_summarise_both_wagons_and_list_the_instances():
    sums = {150: R.oracle_summary(DATA, 150), 250: R.oracle_summary(DATA, 250)}
    df = UI.oracle_frame(sums)
    assert list(df["Wagen"]) == [150, 250] and list(df["Instanzen (bewiesen optimal)"]) == ["30 (30)", "30 (30)"]
    assert df.iloc[0]["Lücke Reaktiv (%)"] == "28,7 ± 2,1" and df.iloc[1]["Lücke Bündeln P(3; 0,5) (%)"] == "11,5 ± 1,4" and list(df["P(3; 0,5) schließt Anteil der reaktiven Lücke"]) == ["48 %", "61 %"]
    assert df.iloc[0]["Touren in 6 Tagen: Optimum / R / P(3; 0,5)"] == "3,03 / 4,67 / 3,70" and df.iloc[1]["Touren in 6 Tagen: Optimum / R / P(3; 0,5)"] == "2,53 / 4,67 / 3,37"
    inst = UI.oracle_instances_frame(DATA, 150)
    assert len(inst) == 30 and list(inst["Seed"]) == list(range(30)) and set(inst["Status"]) == {"OPTIMAL"} and inst.iloc[0]["Optimum"] == "516,4"
    assert float(inst.iloc[0]["Lücke reaktiv (%)"].replace(",", ".")) == pytest.approx(100 * (710.0 - 516.4) / 516.4, abs=0.06)


def test_small_plan_and_compare_frames():
    plans = [[[(1, 60.0), (3, 20.0)]], [], [[(2, 40.0)]]]
    df = UI.small_plan_frame(plans)
    assert list(df["Tag"]) == [1, 3] and list(df["Tour"]) == ["1 (60) → 3 (20)", "2 (40)"] and list(UI.small_plan_frame([[], []]).columns) == ["Tag", "Tour"]
    cmp_ = {"optimum": {"J": 500.0, "routes": 3, "routing": 480.0, "short": 0.0}, "R": {"J": 640.0, "gap": 28.0, "routes": 5, "routing": 600.0, "short": 3.0},
            "P": {"J": 550.0, "gap": 10.0, "routes": 4, "routing": 520.0, "short": 0.0}, "E": {"J": 800.0, "gap": 60.0, "routes": 6, "routing": 790.0, "short": 0.0}}
    out = UI.compare_small_frame(cmp_)
    assert list(out["Plan"]) == ["Optimum (CP-SAT)", "Reaktiv", "Bündeln", "Früher liefern"] and list(out["Lücke zum Optimum (%)"]) == ["0,0", "+28,0", "+10,0", "+60,0"]
    assert list(out["Touren"]) == [3, 5, 4, 6] and list(out["Fehlmenge"]) == ["0", "3", "0", "0"] and out.iloc[0]["Kosten J"] == "500,0"


def test_day_frame_has_one_row_per_day(live):
    df = UI.day_frame(live)
    assert len(df) == 120 and list(df.columns)[0] == "Tag" and df["Mitnehmer"].sum() == live["rules"]["P"]["res"]["opt_visits"]


# ---------------------------------------------------------------------------------------------------
# Zeichnen (Streamlit) mit einem kleinen Skript: render_day und render_message
# ---------------------------------------------------------------------------------------------------
def _day_script(live, day, right):
    import irp_ui_panel as UI_
    UI_.render_day("t", live, day, right)


def _message_script(live, cell, note):
    import irp_ui_panel as UI_
    UI_.render_message(live, cell, note)


def _run(fn, *args):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_function(fn, args=args, default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def test_render_day_draws_two_maps_captions_and_hallisten(live):
    d = LV.pickup_days(live, 1)[0]
    at = _run(_day_script, live, d, "P")
    md = [m.value for m in at.markdown]
    assert md == [f"**Reaktiv** – Tag {d}", f"**Bündeln P(3; 0,5)** – Tag {d}"]
    keys = [c.proto.id for c in at.get("plotly_chart")]
    assert len(keys) == 2 and len(set(keys)) == 2
    rp, pp = LV.day_view(live, "R", d), LV.day_view(live, "P", d)
    caps = [c.value for c in at.caption]
    assert caps[0] == f"{len(rp['tours'])} Tour{'en' if len(rp['tours']) != 1 else ''}, {rp['n_stops']} Stopp{'s' if rp['n_stops'] != 1 else ''}, {round(rp['km'])} km."
    assert caps[1] == (f"{len(pp['tours'])} Tour{'en' if len(pp['tours']) != 1 else ''}, {pp['n_stops']} Stopp{'s' if pp['n_stops'] != 1 else ''}, davon {pp['n_pickups']} "
                       f"Mitnahme{'n' if pp['n_pickups'] != 1 else ''}, {round(pp['km'])} km.")
    assert [t.label for t in at.tabs] == ["Halteliste Reaktiv", "Halteliste Bündeln P(3; 0,5)"] and len(at.dataframe) == 2 and len(at.dataframe[1].value) == pp["n_stops"]


def test_render_day_with_the_early_rule_and_an_empty_day(live):
    d = next(d for d in range(1, 121) if LV.day_view(live, "E", d)["tours"])
    at = _run(_day_script, live, d, "E")
    assert [m.value for m in at.markdown][1] == f"**Früher liefern E(2)** – Tag {d}" and "davon" not in " ".join(c.value for c in at.caption)
    empty = next(d for d in range(1, 121) if not LV.day_view(live, "R", d)["tours"] and not LV.day_view(live, "P", d)["tours"])
    at = _run(_day_script, live, empty, "P")
    assert [c.value for c in at.caption if c.value.startswith("An diesem Tag")] == ["An diesem Tag fährt diese Regel keine Tour."] * 2
    assert len(at.dataframe) == 0 and [c.value for c in at.caption].count("Keine Tour an diesem Tag.") == 2


def test_render_message_uses_the_box_of_the_state(live):
    at = _run(_message_script, live, BASE, "")
    assert len(at.success) == 1 and not at.info and not at.warning and at.success[0].value.startswith("Bündeln lohnt") and at.success[0].icon == "✅"          # Streamlit macht das führende Emoji zum Symbol des Kastens
    at = _run(_message_script, live, R.find_cell(DATA, Q=100.0), "Hinweis.")
    assert len(at.info) == 1 and not at.success and at.info[0].value.endswith(" Hinweis.") and at.info[0].icon == "ℹ️"
    at = _run(_message_script, dict(live, H=12.0, gamma=1.0), R.find_cell(DATA, Q=150.0), "")
    assert len(at.warning) == 1 and at.warning[0].icon == "⚠️" and at.warning[0].value.startswith("Diese Mitnahmeregel")
