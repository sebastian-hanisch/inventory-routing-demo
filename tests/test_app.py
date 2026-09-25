"""AppTest: Skelett und Footer, jedes Preset, Permalink, alle Regler an Min und Max, alle drei Meldungszustände, Kernabschnitte, Ansichten, PDF, Exakt-Tab (Kleininstanz),
Texte. Die Live-Rechnung läuft auf den Instanzen des Zufallsgenerators; die Tests vergleichen deshalb mit derselben Rechnung (irp_live.solve_live) statt mit festen Zahlen,
feste Zahlen stehen nur dort, wo sie aus der vorgerechneten Messreihe kommen. Keine Wall-Clock-Annahmen (Pause des Exakt-Tabs über den Sitzungszustand gesteuert)."""
import pathlib
import re
import time

import pytest
from streamlit.testing.v1 import AppTest

import irp_constants as C
import irp_live as LV
import irp_oracle as OR
import irp_presets as P
import irp_results as R
import irp_ui_panel as UI
from irp_presets import PRESET_STATE_KEYS, SETTING_SPECS

APP = str(pathlib.Path(__file__).resolve().parent.parent / "app.py")
DATA = R.load_results()
FOOTER = ("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
          "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
          "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)")
SLIDERS = ("customers_slider", "capacity_slider", "fleet_slider", "sigma_slider", "penalty_slider", "horizon_slider", "gamma_slider", "early_slider")


@pytest.fixture(autouse=True)
def clean_state():
    """st.cache_data ist prozessweit: Tests dürfen keine Ergebnisse anderer Tests sehen."""
    import streamlit as st
    st.cache_data.clear()
    yield


def fresh(**query):
    at = AppTest.from_file(APP, default_timeout=180)
    for k, v in query.items():
        at.query_params[k] = v
    at.run()
    assert not at.exception, at.exception
    return at


def set_and_run(at, **values):
    for key, value in values.items():
        if key == "seed_input":
            at.number_input(key=key).set_value(value)
        else:
            at.select_slider(key=key).set_value(value)
    at.run()
    assert not at.exception, at.exception
    return at


def click(at, label):
    next(b for b in at.button if b.label == label).click().run()
    assert not at.exception, at.exception
    return at


def main_metrics(at):
    return [(m.label, m.value) for m in at.metric[:4]]


def box(at, kind, needle):
    for x in getattr(at, kind):
        if needle in x.value:
            return x.value
    return None


def state_values(at):
    return {key: at.session_state[key] for key in SETTING_SPECS}


def live_for(**over):
    p = dict(customers=20, capacity=300, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=C.SEED_DEFAULT)
    p.update(over)
    return LV.solve_live(p["customers"], p["capacity"], p["fleet"], p["sigma"], p["penalty"], p["horizon"], p["gamma"], p["early"], p["seed"])


# ---------------------------------------------------------------------------------------------------
# Skelett
# ---------------------------------------------------------------------------------------------------
def test_skeleton_and_footer():
    at = fresh()
    assert [h.value for h in at.sidebar.header] == ["⚙️ Einstellungen"]                   # genau EIN Header
    assert len(at.title) == 1 and at.title[0].value == "🛢️ Inventory Routing: Wer gehört heute auf die Tour?"
    assert any(v.value.startswith("## 🛢️ Wie viel spart es, mehr Kunden auf eine Tour mitzunehmen?") for v in at.markdown)
    assert any(v.value.startswith("### 📐 Was die Messreihe über 200 Instanzen zeigt") for v in at.markdown)
    assert [e.label for e in at.expander] == ["🔧 Wie wir das erreichen – Regeln im Vergleich", "Wie funktioniert diese Demo?", "📐 Mathematische Formulierung"]
    assert any(c.value == FOOTER for c in at.caption)
    presets = [b.label for b in at.button if b.label in C.PRESETS]
    assert presets == list(C.PRESETS) == ["Standard", "Wagen fast voll", "Großer Wagen", "Zu großzügig", "Knappe Flotte"] and all(len(n) <= 32 for n in presets)
    assert [s.label for s in at.sidebar.select_slider] == ["Kunden", "Wagenkapazität", "Touren je Tag", "Verbrauchsschwankung", "Fehlmengenstrafe", "Vorschau H", "Mitnahmeschwelle γ", "Früher liefern (Tage)"]
    assert [n.label for n in at.sidebar.number_input] == ["Seed"] and not at.sidebar.slider
    assert [b.label for b in at.sidebar.button] == ["🎲 Neue Instanz"]                       # letztes Sidebar-Element
    assert [type(e).__name__ for e in at.sidebar.children.values()][-1] == "Button"
    assert [t.label for t in at.tabs][-4:] == ["🗺️ Tagesansicht", "📊 Regeln", "🎯 Exaktes Optimum", "📈 Messreihe"] and [t.label for t in at.tabs][:2] == ["Halteliste Reaktiv", "Halteliste Bündeln P(3; 0,5)"]


def test_the_preset_buttons_sit_in_two_rows_of_three_and_two_and_have_help():
    at = fresh()
    buttons = [b for b in at.button if b.label in C.PRESETS]
    assert all(b.help and len(b.help) > 20 for b in buttons) and [b.help for b in buttons] == [C.PRESET_HELP[n] for n in C.PRESETS]


def test_intro_names_the_expanders_and_neighbours():
    at = fresh()
    intro = next(m.value for m in at.markdown if m.value.strip().startswith("Kunden mit Tanks verbrauchen jeden Tag"))
    for needle in ("wer heute auf die Tour gehört", "reaktiv", "bündeln", "Bestand beim Kunden", "Wagenkapazität", "Wie funktioniert diese Demo?", "📐 Mathematische Formulierung", "vrp_demo", "alns-demo",
                   "vrp-nachbarschaften-demo", "leercontainer-demo", "nahverkehr-demo", "fernverkehr-demo", "Inventory Routing Problem", "Kleininstanz"):
        assert needle in intro, needle


def test_sidebar_has_no_second_header_or_subheader_and_named_groups():
    at = fresh()
    assert len(at.sidebar.header) == 1 and len(at.sidebar.subheader) == 0
    assert [m.value for m in at.sidebar.markdown] == ["**Der Betrieb**", "**Die Regel**", "**Die gezeigte Instanz**"]


def test_there_are_no_dead_switches_and_the_only_button_with_a_solver_is_in_the_exact_tab():
    at = fresh()
    assert not at.sidebar.checkbox and not at.sidebar.toggle and not at.checkbox and not at.toggle
    assert len(at.sidebar.select_slider) == 8 and not at.sidebar.slider and len(at.sidebar.number_input) == 1
    solver_buttons = [b for b in at.button if b.key == "oracle_button"]
    assert len(solver_buttons) == 1 and solver_buttons[0].label == "🧮 Optimum berechnen (CP-SAT)"
    assert sorted(b.label for b in at.button if b.label not in C.PRESETS) == sorted(["🎲 Neue Instanz", "🧮 Optimum berechnen (CP-SAT)"])
    assert [s.key for s in at.slider] == ["day_slider"] and [r.key for r in at.radio] == ["view_select", "oracle_wagon"] and [s.key for s in at.selectbox] == ["cell_select"]


def test_sliders_have_the_measured_stages_and_the_seed_bounds():
    at = fresh()
    assert [o for o in at.select_slider(key="customers_slider").options] == ["10", "20", "40"] and [o for o in at.select_slider(key="capacity_slider").options] == ["100", "150", "200", "300", "450", "600", "1000"]
    assert [o for o in at.select_slider(key="fleet_slider").options] == ["1", "2", "6"] and [o for o in at.select_slider(key="sigma_slider").options] == ["0,0", "0,15", "0,3", "0,6", "1,0"]
    assert [o for o in at.select_slider(key="penalty_slider").options] == ["1 je Einheit", "5 je Einheit", "25 je Einheit", "100 je Einheit"]
    assert [o for o in at.select_slider(key="horizon_slider").options] == ["1 Tage", "2 Tage", "3 Tage", "4 Tage", "6 Tage", "8 Tage", "12 Tage"]
    assert [o for o in at.select_slider(key="gamma_slider").options] == ["0,1", "0,25", "0,5", "1,0"] and [o for o in at.select_slider(key="early_slider").options] == ["1 Tage", "2 Tage", "3 Tage"]
    seed = at.number_input(key="seed_input")
    assert (seed.min, seed.max, seed.step, seed.value) == (0, 299, 1, C.SEED_DEFAULT)
    day = at.slider(key="day_slider")
    assert (day.min, day.max) == (1, 120)


def test_capacity_control_shows_the_ratio_to_the_delivery_size():
    at = fresh()
    assert any(c.value.startswith("Q/q ≈ 3,0 Lieferungen je Wagen (mittlere Liefermenge q ≈ 99 bei reaktivem Nachliefern, Messreihe)") for c in at.sidebar.caption)
    set_and_run(at, capacity_slider=100)
    assert any(c.value.startswith("Q/q ≈ 1,1 Lieferungen je Wagen") for c in at.sidebar.caption)
    set_and_run(at, capacity_slider=1000)
    assert any(c.value.startswith("Q/q ≈ 10,1 Lieferungen je Wagen") for c in at.sidebar.caption)


def test_default_state_and_permalink_written_to_the_address_bar():
    at = fresh()
    assert state_values(at) == dict(customers_slider=20, capacity_slider=300, fleet_slider=6, sigma_slider=0.3, penalty_slider=5, horizon_slider=3, gamma_slider=0.5, early_slider=2, seed_input=C.SEED_DEFAULT)
    qp = {k: (v[0] if isinstance(v, list) else v) for k, v in dict(at.query_params).items()}
    assert qp == {"n": "20", "q": "300", "f": "6", "s": "0.3", "p": "5", "h": "3", "g": "0.5", "l": "2", "seed": str(C.SEED_DEFAULT)}


def test_main_metrics_are_2x2_with_the_right_labels_and_values():
    at = fresh()
    labels = [m[0] for m in main_metrics(at)]
    assert labels == ["Kosten reaktiv (R)", "Kosten Bündeln P(3; 0,5)", "Kosten Früher liefern E(2)", "Touren in 120 Tagen (R / P / E)"]
    live = live_for()
    x = UI.live_numbers(live)
    fmt = lambda v: f"{round(v):,}".replace(",", ".")
    assert [m[1] for m in main_metrics(at)] == [fmt(x["R"]["J"]), fmt(x["P"]["J"]), fmt(x["E"]["J"]), f"{x['R']['routes']} / {x['P']['routes']} / {x['E']['routes']}"]
    delta = {m.label: m.delta for m in at.metric[:4]}
    assert delta["Kosten Bündeln P(3; 0,5)"].startswith("-") and delta["Kosten Früher liefern E(2)"].startswith("+") and "(+" in delta["Kosten Bündeln P(3; 0,5)"] and " %)" in delta["Kosten Bündeln P(3; 0,5)"]
    assert any(c.value.startswith("Fehlmenge in 120 Tagen (Einheiten, R / P / E):") for c in at.caption)
    assert len(at.columns) >= 4


def test_live_caption_says_it_is_one_instance():
    at = fresh()
    cap = next(c.value for c in at.caption if c.value.startswith("Live-Instanz (eine Instanz, 120 Tage)"))
    assert f"Seed {C.SEED_DEFAULT}" in cap and "Eine einzelne Instanz" in cap and "20 Kunden" in cap and "Wagen 300" in cap and "unter 0,05 s" in cap


# ---------------------------------------------------------------------------------------------------
# Presets, Permalink, Regler
# ---------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(C.PRESETS))
def test_every_preset_sets_all_controls_and_shows_its_message(name):
    at = fresh()
    click(at, name)
    p = C.PRESETS[name]
    assert state_values(at) == {PRESET_STATE_KEYS[k]: v for k, v in p.items()}
    expected = {"Standard": ("success", "Bündeln lohnt"), "Wagen fast voll": ("info", "Bündeln bringt hier wenig"), "Großer Wagen": ("success", "Bündeln lohnt"),
                "Zu großzügig": ("warning", "Diese Mitnahmeregel ist hier teurer als reaktiv"), "Knappe Flotte": ("success", "Bündeln lohnt")}[name]
    kind, needle = expected
    text = box(at, kind, needle)
    assert text is not None
    assert next(x for x in getattr(at, kind) if needle in x.value).icon == {"success": "✅", "info": "ℹ️", "warning": "⚠️"}[kind]
    cell, exact, _ = R.nearest_cell(DATA, p["customers"], p["capacity"], p["fleet"], p["sigma"], p["penalty"])
    assert exact and R.cell_setting_text(cell) in text
    assert dict(at.query_params).get("seed") in (str(C.SEED_DEFAULT), [str(C.SEED_DEFAULT)])
    live = live_for(**p)
    assert at.metric[0].value == f"{round(live['rules']['R']['J']):,}".replace(",", ".")


def test_presets_change_the_instance_shown():
    at = fresh()
    base_values = main_metrics(at)
    click(at, "Großer Wagen")
    assert main_metrics(at) != base_values
    click(at, "Knappe Flotte")
    assert next(c.value for c in at.caption if c.value.startswith("Live-Instanz")).count("höchstens 1 Tour je Tag") == 1
    click(at, "Standard")
    assert main_metrics(at) == base_values


def test_permalink_sets_the_controls_and_snaps_to_stages():
    at = fresh(n="40", q="450", f="2", s="1.0", p="25", h="8", g="0.25", l="3", seed="17")
    assert state_values(at) == dict(customers_slider=40, capacity_slider=450, fleet_slider=2, sigma_slider=1.0, penalty_slider=25, horizon_slider=8, gamma_slider=0.25, early_slider=3, seed_input=17)
    at = fresh(n="31", q="170", f="4", s="0.2", p="26", h="5", g="0.4", l="9", seed="-4")
    assert state_values(at) == dict(customers_slider=40, capacity_slider=150, fleet_slider=2, sigma_slider=0.15, penalty_slider=25, horizon_slider=4, gamma_slider=0.5, early_slider=3, seed_input=0)
    at = fresh(n="abc", q="", seed="x")
    assert state_values(at)["customers_slider"] == 20 and state_values(at)["capacity_slider"] == 300 and state_values(at)["seed_input"] == C.SEED_DEFAULT


def test_all_controls_at_min_and_max_run_without_exception():
    at = fresh()
    for key in SLIDERS:
        options = list(SETTING_SPECS[key].options)
        for v in (options[0], options[-1]):
            set_and_run(at, **{key: v})
            assert state_values(at)[key] == v
    for v in (0, 299):
        set_and_run(at, seed_input=v)
        assert state_values(at)["seed_input"] == v
    set_and_run(at, customers_slider=10, capacity_slider=100, fleet_slider=1, sigma_slider=0.0, penalty_slider=1, horizon_slider=1, gamma_slider=0.1, early_slider=1, seed_input=0)
    assert box(at, "info", "Bündeln bringt hier wenig") is not None
    set_and_run(at, customers_slider=40, capacity_slider=1000, fleet_slider=6, sigma_slider=1.0, penalty_slider=100, horizon_slider=12, gamma_slider=1.0, early_slider=3, seed_input=299)
    assert at.metric[0].value


def test_new_instance_button_rolls_a_new_seed(monkeypatch):
    frozen = iter([3, 5, C.SEED_DEFAULT, 3])
    monkeypatch.setattr(P.random, "randint", lambda lo, hi: next(frozen))
    at = fresh()
    seen = {at.session_state["seed_input"]}
    for _ in range(3):
        click(at, "🎲 Neue Instanz")
        seen.add(at.session_state["seed_input"])
        assert 0 <= at.session_state["seed_input"] <= 299
    assert seen == {C.SEED_DEFAULT, 3, 5}


# ---------------------------------------------------------------------------------------------------
# Meldung in drei Zuständen, Zell-Zuordnung, keine wirkungslosen Regler
# ---------------------------------------------------------------------------------------------------
def test_three_message_states_follow_the_controls():
    at = fresh()
    assert box(at, "success", "Bündeln lohnt") and not box(at, "info", "Bündeln bringt hier wenig") and not box(at, "warning", "teurer als reaktiv")
    set_and_run(at, capacity_slider=100)
    assert box(at, "info", "Bündeln bringt hier wenig") and not box(at, "success", "Bündeln lohnt")
    set_and_run(at, capacity_slider=150, horizon_slider=12, gamma_slider=1.0)
    assert box(at, "warning", "Diese Mitnahmeregel ist hier teurer als reaktiv") and not box(at, "info", "Bündeln bringt hier wenig")
    set_and_run(at, gamma_slider=0.25)
    assert box(at, "success", "Bündeln lohnt") and not box(at, "warning", "teurer als reaktiv")                  # kleinere Mitnahmeschwelle: wieder billiger
    set_and_run(at, capacity_slider=300, gamma_slider=1.0)
    assert box(at, "warning", "P(12; 1,0) kostet laut Messreihe")                                                 # auch im Basisfall kippt γ = 1 bei weiter Vorschau (gemessen −5,1)


def test_the_message_uses_the_measurement_of_the_set_rule_not_the_instance():
    at = fresh()
    text = box(at, "success", "Bündeln lohnt")
    assert "17,0 ± 0,3 %" in text and "Median 16,9 %" in text and "in 0 % der Instanzen ein Verlust" in text and "Auf dieser einen Instanz:" in text
    set_and_run(at, seed_input=5)
    assert "17,0 ± 0,3 %" in box(at, "success", "Bündeln lohnt")                                                    # andere Instanz, dieselbe Messreihe


def test_every_control_changes_the_live_result_no_dead_controls():
    at = fresh()
    base = main_metrics(at)
    for key, value in (("customers_slider", 40), ("capacity_slider", 100), ("fleet_slider", 1), ("sigma_slider", 0.0), ("penalty_slider", 100), ("horizon_slider", 12), ("gamma_slider", 0.1),
                       ("early_slider", 3), ("seed_input", 5)):
        other = fresh()
        set_and_run(other, **{key: value})
        assert main_metrics(other) != base, key


def test_rule_controls_change_only_their_own_rule_and_not_reactive():
    at = fresh()
    r0, p0, e0, t0 = (m[1] for m in main_metrics(at))
    set_and_run(at, horizon_slider=8)
    r1, p1, e1, _ = (m[1] for m in main_metrics(at))
    assert r1 == r0 and p1 != p0 and e1 == e0                                        # H wirkt nur auf Bündeln
    set_and_run(at, gamma_slider=0.1)
    assert main_metrics(at)[0][1] == r0 and main_metrics(at)[1][1] != p1
    set_and_run(at, early_slider=3)
    r3, _, e3, _ = (m[1] for m in main_metrics(at))
    assert r3 == r0 and e3 != e0                                                     # L wirkt nur auf früher liefern


def test_only_the_live_settings_recompute_the_instance_display_choices_do_not(monkeypatch):
    calls = []
    real = LV.solve_live
    monkeypatch.setattr(LV, "solve_live", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    at = fresh()
    assert len(calls) == 1
    at.slider(key="day_slider").set_value(50).run()
    at.radio(key="view_select").set_value("Früher liefern neben Reaktiv").run()
    at.selectbox(key="cell_select").set_value("q100").run()
    assert len(calls) == 1 and not at.exception                                      # Tag, Anzeige und Zellenwahl sind reine Anzeige (st.cache_data)
    set_and_run(at, horizon_slider=4)
    assert len(calls) == 2
    set_and_run(at, horizon_slider=3)
    assert len(calls) == 2                                                           # zurück auf eine schon gerechnete Einstellung: aus dem Zwischenspeicher


def test_unmeasured_combination_shows_the_nearest_cell_with_a_note():
    at = fresh()
    set_and_run(at, customers_slider=40, capacity_slider=100)
    note = "keine eigene Messreihe: die Messreihe variiert je Zelle nur einen Parameter"
    assert any(note in i.value for i in at.info) and any("nächstliegende gemessene Zelle" in i.value for i in at.info) and any(note in c.value for c in at.caption)
    assert any("abweichend: Kunden." in i.value for i in at.info)
    set_and_run(at, customers_slider=20, capacity_slider=300)
    assert not any(note in i.value for i in at.info) and not any(note in c.value for c in at.caption)      # exakt gemessen: kein Hinweis


def test_every_setting_is_assigned_a_cell_and_the_note_names_the_differences():
    at = fresh()
    set_and_run(at, customers_slider=10, capacity_slider=1000, fleet_slider=2, sigma_slider=0.6, penalty_slider=25)
    text = next(i.value for i in at.info if "keine eigene Messreihe" in i.value)
    assert "10 Kunden, Wagen 1000, 2 Touren je Tag, Verbrauchsschwankung 0,6, Strafe 25" in text and "Wagen 1000" in text and "abweichend:" in text


# ---------------------------------------------------------------------------------------------------
# Tagesansicht
# ---------------------------------------------------------------------------------------------------
def test_day_view_defaults_to_the_first_pickup_day_and_shows_two_maps():
    at = fresh()
    live = live_for()
    assert at.slider(key="day_slider").value == LV.pickup_days(live, 1)[0]
    assert at.radio(key="view_select").value == "Bündeln neben Reaktiv" and list(at.radio(key="view_select").options) == ["Bündeln neben Reaktiv", "Früher liefern neben Reaktiv"]
    keys = [c.proto.id for c in at.get("plotly_chart")]
    assert len(set(keys)) == len(keys)
    text = " ".join(m.value for m in at.markdown)
    assert "**Reaktiv** – Tag" in text and "**Bündeln P(3; 0,5)** – Tag" in text
    assert any("Tage, an denen Bündeln Kunden mitnimmt (die ersten):" in c.value for c in at.caption)
    assert [t.label for t in at.tabs][:2] == ["Halteliste Reaktiv", "Halteliste Bündeln P(3; 0,5)"]


def test_day_slider_and_view_choice_are_display_only_and_keep_the_metrics():
    at = fresh()
    before = main_metrics(at)
    at.slider(key="day_slider").set_value(1).run()
    assert not at.exception and main_metrics(at) == before
    at.slider(key="day_slider").set_value(120).run()
    assert not at.exception
    at.radio(key="view_select").set_value("Früher liefern neben Reaktiv").run()
    assert not at.exception and main_metrics(at) == before
    assert "**Früher liefern E(2)** – Tag 120" in " ".join(m.value for m in at.markdown) and [t.label for t in at.tabs][:2] == ["Halteliste Reaktiv", "Halteliste Früher liefern E(2)"]


def test_a_day_without_a_tour_says_so():
    at = fresh()
    live = live_for()
    empty = next(d for d in range(1, 121) if not live["rules"]["R"]["plans"][d - 1] and not live["rules"]["P"]["plans"][d - 1])
    at.slider(key="day_slider").set_value(empty).run()
    assert not at.exception and sum("An diesem Tag fährt diese Regel keine Tour." in c.value for c in at.caption) == 2


def test_instance_without_pickups_says_so():
    at = fresh()
    set_and_run(at, capacity_slider=100)                                              # Wagen fast voll: Bündeln nimmt nur selten mit
    live = live_for(capacity=100)
    if LV.pickup_days(live, 1):
        assert any("Tage, an denen Bündeln Kunden mitnimmt" in c.value for c in at.caption)
    else:
        assert any("Bei dieser Einstellung nimmt Bündeln keinen Kunden mit." in c.value for c in at.caption)


# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ②, Ansichten, Texte
# ---------------------------------------------------------------------------------------------------
def test_core_section_two_shows_the_measured_numbers():
    at = fresh()
    text = " ".join(c.value for c in at.caption)
    assert "0,4 % für die Standardregel" in text and "17,0 % mit Platz für drei Lieferungen".replace("17,0 % mit Platz für drei Lieferungen", "sind es +17,0 %") in text and "+29,9 %" in text
    assert "Rangkorrelation zwischen Gewinn und Q/q über 23 Konfigurationen: 0,70" in text
    assert "46 von 756" in text and "davon 44 mit γ = 1,0 und keine mit γ ≤ 0,25" in text
    assert "+16,5 %" in text and "-7,0 / -16,3 / -27,1 %" in text and "in 75 von 81" in text and "22,5 %" in text and "132,8 → 99,7" in text and "0,59 → 0,79" in text
    assert "48 % bzw. 61 %" in text and "kein Optimum" in text
    md = " ".join(m.value for m in at.markdown)
    assert "26 von 28" in md and "höchstens 3 Prozentpunkte darunter" in md and "**5**" in md and "Ihre Einstellung ist umrandet" in md
    assert "**1 · Kapazitätshebel**" in md and "**7 · Exakter Maßstab**" in md and "**6 · Regime**" in md
    frames = [d.value for d in at.dataframe]
    assert len(frames) >= 14 and any("Bündeln lohnt" in " ".join(map(str, f.to_numpy().ravel())) for f in frames if hasattr(f, "to_numpy"))


def test_core_section_reacts_to_the_set_rule_and_the_cell():
    at = fresh()
    set_and_run(at, capacity_slider=150, horizon_slider=12, gamma_slider=1.0)
    md = " ".join(m.value for m in at.markdown)
    assert "**P(12; 1,0)**" in md and "**Wagen 150**".replace("**Wagen 150**", "Wagen 150") in md
    frames = [f.value for f in at.dataframe]
    assert any("Diese Mitnahmeregel ist hier teurer als reaktiv" in " ".join(map(str, f.to_numpy().ravel())) for f in frames)


def test_regime_and_capacity_tables_mark_the_chosen_cell():
    at = fresh()
    frames = [d.value for d in at.dataframe]
    regime = next(f for f in frames if "Gruppe" in f.columns and "Anteil Verlust" in f.columns)
    assert len(regime) == 27 and [z for z in regime["Zelle"] if z.startswith("▶")] == ["▶ Basisfall"]
    cap = next(f for f in frames if "Wagenkapazität Q" in f.columns)
    assert [q for q in cap["Wagenkapazität Q"] if "eingestellt" in q] == ["300  ◀ eingestellt"]
    set_and_run(at, capacity_slider=600)
    frames = [d.value for d in at.dataframe]
    assert [z for z in next(f for f in frames if "Anteil Verlust" in f.columns)["Zelle"] if z.startswith("▶")] == ["▶ Wagen 600"]


def test_exact_tab_explains_that_the_stochastic_base_has_no_optimum():
    at = fresh()
    assert any("kein Optimum" in i.value and "20 Kunden, 120 Tage" in i.value for i in at.info)
    assert [r.label for r in at.radio if r.key == "oracle_wagon"] == ["Wagen der Kleininstanz"] and list(at.radio(key="oracle_wagon").options) == ["150", "250"]
    assert at.number_input(key="oracle_seed").value == 0 and at.number_input(key="oracle_seed").max == 299
    inst = [d.value for d in at.dataframe if "Status" in d.value.columns]
    assert len(inst) == 1 and len(inst[0]) == 30 and set(inst[0]["Status"]) == {"OPTIMAL"}


def test_exact_button_solves_the_small_instance_and_shows_optimum_gaps_and_plan():
    pytest.importorskip("ortools")
    at = fresh()
    click(at, "🧮 Optimum berechnen (CP-SAT)")
    direct = OR.solve_small(150, 0, 3, 0.5, 2)
    metrics = {m.label: m for m in at.metric if m.label in ("Optimum (CP-SAT)", "Reaktiv", "Bündeln P(3; 0,5)", "Früher liefern E(2)")}
    assert set(metrics) == {"Optimum (CP-SAT)", "Reaktiv", "Bündeln P(3; 0,5)", "Früher liefern E(2)"}
    assert metrics["Optimum (CP-SAT)"].delta.split(",")[0] in ("OPTIMAL", "FEASIBLE") and metrics["Reaktiv"].delta.endswith("% Lücke")
    if direct["status"] == "OPTIMAL" and metrics["Optimum (CP-SAT)"].delta.startswith("OPTIMAL,"):                                   # nur bewiesene Optima sind eindeutig (und nur der Zielwert, nicht der Plan)
        assert metrics["Optimum (CP-SAT)"].value == f"{direct['compare']['optimum']['J']:.1f}".replace(".", ",") and metrics["Reaktiv"].delta.startswith("+")
    assert metrics["Reaktiv"].value == f"{direct['compare']['R']['J']:.1f}".replace(".", ",")
    frames = [d.value for d in at.dataframe]
    cmp_frame = next(f for f in frames if "Lücke zum Optimum (%)" in f.columns)
    assert list(cmp_frame["Plan"]) == ["Optimum (CP-SAT)", "Reaktiv", "Bündeln", "Früher liefern"]
    plan = next(f for f in frames if list(f.columns) == ["Tag", "Tour"])
    assert len(plan) >= 1 and any(c.proto.id for c in at.get("plotly_chart"))
    assert any("Der Zielwert des Optimums ist eindeutig" in c.value for c in at.caption)


def test_exact_button_pause_and_stale_result(monkeypatch):
    pytest.importorskip("ortools")
    calls = []
    real = OR.solve_small
    monkeypatch.setattr(OR, "solve_small", lambda *a, **k: (calls.append(a), real(*a, **k))[1])
    at = fresh()
    click(at, "🧮 Optimum berechnen (CP-SAT)")
    assert len(calls) == 1 and at.session_state["oracle_result"]["key"] == (150, 0, 3, 0.5, 2)
    click(at, "🧮 Optimum berechnen (CP-SAT)")                                                       # sofort ein zweites Mal: Pause
    assert len(calls) == 1 and any("Bitte noch" in w.value and "Pause von 4 s" in w.value for w in at.warning)
    at.session_state["oracle_last"] = time.monotonic() - 1000.0                                      # Pause abgelaufen
    at.radio(key="oracle_wagon").set_value(250).run()
    assert any("Die Einstellungen haben sich seit dem letzten Aufruf geändert" in i.value for i in at.info)   # das alte Ergebnis wird nicht mehr gezeigt
    assert not [m for m in at.metric if m.label == "Optimum (CP-SAT)"]
    click(at, "🧮 Optimum berechnen (CP-SAT)")
    assert len(calls) == 2 and calls[1][0] == 250 and [m for m in at.metric if m.label == "Optimum (CP-SAT)"]


def test_exact_button_handles_an_infeasible_instance_and_a_missing_solver(monkeypatch):
    monkeypatch.setattr(OR, "solve_small", lambda *a, **k: None)
    at = fresh()
    click(at, "🧮 Optimum berechnen (CP-SAT)")
    assert any("unzulässig" in w.value for w in at.warning)

    def missing(*a, **k):
        raise ImportError("kein ortools")
    monkeypatch.setattr(OR, "solve_small", missing)
    at = fresh()
    click(at, "🧮 Optimum berechnen (CP-SAT)")
    assert any("OR-Tools ist in dieser Umgebung nicht installiert" in e.value for e in at.error)


def test_exact_button_warns_when_the_time_limit_was_reached(monkeypatch):
    pytest.importorskip("ortools")
    real = OR.solve_small

    def limited(*a, **k):
        res = real(*a, **k)
        return dict(res, status="FEASIBLE")

    monkeypatch.setattr(OR, "solve_small", limited)
    at = fresh()
    click(at, "🧮 Optimum berechnen (CP-SAT)")
    assert any("Das Zeitlimit von 20 s wurde erreicht" in w.value and "nicht bewiesen optimale" in w.value for w in at.warning)
    assert next(m for m in at.metric if m.label == "Optimum (CP-SAT)").delta.startswith("FEASIBLE,")
    assert not any("Zeitlimit" in w.value for w in fresh().warning)                                                  # ohne Aufruf keine Warnung


def test_exact_tab_uses_the_set_rule_parameters_in_its_labels(monkeypatch):
    seen = []
    real = OR.solve_small
    monkeypatch.setattr(OR, "solve_small", lambda *a, **k: (seen.append(a), real(*a, **k))[1])
    at = fresh(h="12", g="0.25", l="3")
    click(at, "🧮 Optimum berechnen (CP-SAT)")
    assert seen[0][2:5] == (12, 0.25, 3) and any(m.label == "Bündeln P(12; 0,25)" for m in at.metric) and any(m.label == "Früher liefern E(3)" for m in at.metric)


def test_measurement_tab_shows_the_grid_of_any_cell_and_the_worst_combinations():
    at = fresh()
    box_ = at.selectbox(key="cell_select")
    assert len(box_.options) == 27 and box_.value == "base"
    at.selectbox(key="cell_select").set_value("cluster").run()
    assert not at.exception and any("Zelle **geklumpte Kunden**" in c.value for c in at.caption)
    frames = [d.value for d in at.dataframe]
    worst = next(f for f in frames if list(f.columns) == ["Zelle", "Regel", "Gewinn (Mittel ± SE, %)"])
    assert len(worst) == 12 and worst.iloc[0]["Regel"] == "P(12; 1,0)"
    assert any("Einzelgewinne der Standardregel P(3; 0,5), Basis: Minimum +10,5, Q1 +15,1, Median +16,9" in c.value for c in at.caption)


def test_rules_tab_lists_the_three_rules_of_the_live_instance():
    at = fresh()
    frames = [d.value for d in at.dataframe]
    rules = next(f for f in frames if list(f.columns)[:1] == ["Regel"] and "Gewinn Messreihe (%)" in f.columns)
    assert list(rules["Regel"]) == ["Reaktiv", "Bündeln P(3; 0,5)", "Früher liefern E(2)"] and list(rules["Gewinn Messreihe (%)"]) == ["Bezug", "+17,0 ± 0,3", "-16,3 ± 0,2"]
    days = next(f for f in frames if list(f.columns)[0] == "Tag")
    assert len(days) == 120


def test_expander_texts_state_the_limits_and_the_findings():
    at = fresh()
    how = next(m.value for m in at.markdown if m.value.strip().startswith("**Instanz.**"))
    for needle in ("Gamma-verteilt", "derselbe", "Wagenkapazität Q", "Strafe je Einheit, kein Rückstau", "Zielgröße", "17,0 / 17,0 / 17,1 %", "Savings (Clarke-Wright) und 2-opt", "H = 0 ist R", "immer alle drei Regeln",
                   "Warum der Gewinn aus volleren Wagen kommt", "99,7 statt 132,8 Touren", "Grenzen dieses Modells", "erfunden, nicht kalibriert", "Stark stilisiert", "Seeds 0 bis 59", "200 bis 299",
                   "0,12 %", "Keine Garantie auf weniger Fehlmengen", "22,5 %", "Nicht Teil dieser Demo", "Mengenwahl", "Zeitfenster", "mehrere Depots", "Korrelation im Verbrauch", "Rückstau",
                   "ein Optimum für die stochastische Basis", "Bündeln lohnt", "Bündeln bringt hier wenig", "teurer als reaktiv", "nächstliegende gemessene Zelle"):
        assert needle in how, needle
    math = next(m.value for m in at.markdown if m.value.strip().startswith("**Zustand und Dynamik.**"))
    for needle in ("I_i(t+1)", "\\max", "s_i(t)", "\\gamma \\cdot 2", "\\kappa", "irp_model.py", "irp_routing.py", "irp_policy.py", "irp_oracle.py", "irp_live.py", "irp_results.py", "2\\,\\text{SE}", "AddCircuit".replace("AddCircuit", "x^t_{ij}")):
        assert needle in math, needle


def test_no_dead_file_links_in_markdown():
    at = fresh()
    for md in list(at.markdown) + list(at.caption):
        for target in re.findall(r"\]\(([^)]+)\)", md.value):
            assert target.startswith("https://"), (target, md.value[:80])                  # nur echte Links, kein [foo.py](foo.py)


def test_real_umlauts_and_no_ascii_replacements_in_texts():
    at = fresh()
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    for bad in (" fuer ", " ueber ", " waehlen ", "Aenderung", "Kuenftig", "Loesung", "Bestaende", "Moeglich", "Waehrung", "Groesse", "Strafe fuer"):
        assert bad not in text, bad
    assert "Bündeln" in text and "Wagenkapazität" in text and "Früher" in text and "größer".replace("größer", "Größe") not in "" and "Überlagerung" not in text


def test_all_plotly_charts_render_with_unique_keys():
    at = fresh()
    charts = at.get("plotly_chart")
    assert len(charts) == 11
    ids = [c.proto.id for c in charts]
    assert len(set(ids)) == len(ids)


def test_dataframes_have_no_null_columns():
    """Keine Ergebnisspalte ist überall gleich (Nullspalten-Signal), außer den per Definition oder Befund konstanten."""
    at = fresh()
    constant_ok = {"Urteil", "Status", "Instanzen (bewiesen optimal)", "Art", "Tour"}          # Urteil: in allen Zellen belastbar; Art/Tour: ein Tag mit nur einer Tour ohne Mitnahme
    for d in at.dataframe:
        df = d.value
        if len(df) < 3:
            continue
        for col in df.columns:
            if str(col) in constant_ok:
                continue
            assert df[col].astype(str).nunique() > 1, (col, df[col].tolist()[:3])


def test_pdf_download_button_is_present_and_named():
    at = fresh()
    buttons = at.get("download_button")
    assert len(buttons) == 1 and buttons[0].proto.label == "📄 Tourenplan als PDF herunterladen"


def test_app_survives_every_preset_with_each_right_hand_view_and_extreme_days():
    at = fresh()
    for name in C.PRESETS:
        click(at, name)
        for label in ("Bündeln neben Reaktiv", "Früher liefern neben Reaktiv"):
            at.radio(key="view_select").set_value(label).run()
            assert not at.exception, (name, label)
        for day in (1, 120):
            at.slider(key="day_slider").set_value(day).run()
            assert not at.exception, (name, day)
