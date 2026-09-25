"""Abnahme der ECHTEN Presets: die Kriterien der Messreihe (an data/irp_results.json, nicht an frisch gewürfelten Instanzen) und die qualitativen Kriterien am gezeigten
Seed. Die Abnahme gegen die Ergebnisdatei ist unabhängig vom Zufallsgenerator (die CI installiert immer das neueste NumPy). Die qualitativen Tageskriterien laufen auf der
Instanz des Preset-Seeds: dort sind sie eine Eigenschaft der Instanz, die sich mit einem anderen Zufallsstrom ändern kann; deshalb prüft die CI nur die Kriterien der Messreihe
und die Struktur, die Tageskriterien nur mit der NumPy-Version, mit der der Seed abgestimmt wurde (tools/tune_presets.py)."""
import numpy as np
import pytest

import irp_constants as C
import irp_live as LV
import irp_results as R
import irp_stories as ST
from toolload import load_tool

TP = load_tool("tune_presets")
DATA = R.load_results()
TUNED_WITH = "2.5.3"


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_measured_criteria_hold_for_every_preset(name):
    result = ST.criteria(name, DATA)
    assert result and all(ok for ok, _ in result), result


def test_preset_settings_match_the_cells_the_criteria_read():
    p = C.PRESETS
    assert (p["Standard"]["capacity"], p["Standard"]["fleet"], p["Standard"]["customers"], p["Standard"]["sigma"], p["Standard"]["penalty"]) == (300, 6, 20, 0.3, 5)
    assert (p["Standard"]["horizon"], p["Standard"]["gamma"], p["Standard"]["early"]) == (3, 0.5, ST.STANDARD_EARLY)
    assert p["Wagen fast voll"]["capacity"] == 100 and p["Großer Wagen"]["capacity"] == 600 and p["Knappe Flotte"]["fleet"] == 1 and p["Knappe Flotte"]["capacity"] == 300
    assert (p["Zu großzügig"]["capacity"], p["Zu großzügig"]["horizon"], p["Zu großzügig"]["gamma"]) == (150, ST.GENEROUS_H, 1.0)
    for name in ("Wagen fast voll", "Großer Wagen", "Knappe Flotte"):
        assert (p[name]["horizon"], p[name]["gamma"]) == (C.DEFAULT_H, C.DEFAULT_GAMMA)                # die Kriterien lesen die Standardregel
    for name, preset in p.items():
        cell, exact, _ = R.nearest_cell(DATA, preset["customers"], preset["capacity"], preset["fleet"], preset["sigma"], preset["penalty"])
        assert exact, name                                                                                # jedes Preset liegt auf einer gemessenen Zelle


def test_every_preset_shows_its_message_state():
    expected = {"Standard": C.STATE_LOHNT, "Wagen fast voll": C.STATE_WENIG, "Großer Wagen": C.STATE_LOHNT, "Zu großzügig": C.STATE_TEURER, "Knappe Flotte": C.STATE_LOHNT}
    for name, preset in C.PRESETS.items():
        cell, _, _ = R.nearest_cell(DATA, preset["customers"], preset["capacity"], preset["fleet"], preset["sigma"], preset["penalty"])
        assert R.judge(cell, preset["horizon"], preset["gamma"])["state"] == expected[name], name


def test_preset_seeds_lie_outside_the_measured_seeds():
    for name, p in C.PRESETS.items():
        assert 200 <= p["seed"] <= C.SEED_RANGE[1] and p["seed"] > 199, name


@pytest.mark.skipif(np.__version__ != TUNED_WITH, reason="der Anzeige-Seed wurde mit dieser NumPy-Version abgestimmt (Zufallsstrom kann sich ändern)")
@pytest.mark.parametrize("name", list(C.PRESETS))
def test_day_criteria_hold_on_the_shown_instance_with_the_tuned_numpy(name):
    res = TP.evaluate(C.PRESETS[name]["seed"], {name: C.PRESETS[name]})
    ok, crit, _ = res[name]
    assert ok, crit


def test_the_shown_instance_is_a_valid_live_instance_for_every_preset():
    for name, p in C.PRESETS.items():
        live = LV.solve_live(p["customers"], p["capacity"], p["fleet"], p["sigma"], p["penalty"], p["horizon"], p["gamma"], p["early"], p["seed"])
        assert live["days"] == 120 and set(live["rules"]) == {"R", "P", "E"} and live["rules"]["R"]["gain"] == 0.0
