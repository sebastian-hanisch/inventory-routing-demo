"""Regler-Spezifikation, Permalink (Begrenzen und Einrasten), Presets, Seed-Knopf und Reglergrenzen."""
import random

import pytest
import streamlit as st

import irp_constants as C
import irp_presets as P
from irp_presets import SETTING_SPECS, PRESET_STATE_KEYS


@pytest.fixture
def fake_state(monkeypatch):
    """st.session_state und st.query_params durch einfache dicts ersetzen (ohne laufende App)."""
    state, params = {}, {}
    monkeypatch.setattr(st, "session_state", state)
    monkeypatch.setattr(st, "query_params", params)
    return state, params


def test_setting_specs_cover_every_control_once():
    assert list(SETTING_SPECS) == ["customers_slider", "capacity_slider", "fleet_slider", "sigma_slider", "penalty_slider", "horizon_slider", "gamma_slider", "early_slider", "seed_input"]
    assert len({s.url_param for s in SETTING_SPECS.values()}) == 9 and {s.url_param for s in SETTING_SPECS.values()} == {"n", "q", "f", "s", "p", "h", "g", "l", "seed"}
    assert set(PRESET_STATE_KEYS.values()) == set(SETTING_SPECS) and list(PRESET_STATE_KEYS) == ["customers", "capacity", "fleet", "sigma", "penalty", "horizon", "gamma", "early", "seed"]
    assert SETTING_SPECS["customers_slider"].options == C.CUSTOMER_OPTIONS == (10, 20, 40) and SETTING_SPECS["capacity_slider"].options == (100, 150, 200, 300, 450, 600, 1000)
    assert SETTING_SPECS["fleet_slider"].options == (1, 2, 6) and SETTING_SPECS["sigma_slider"].options == (0.0, 0.15, 0.3, 0.6, 1.0) and SETTING_SPECS["penalty_slider"].options == (1, 5, 25, 100)
    assert SETTING_SPECS["horizon_slider"].options == (1, 2, 3, 4, 6, 8, 12) and SETTING_SPECS["gamma_slider"].options == (0.1, 0.25, 0.5, 1.0) and SETTING_SPECS["early_slider"].options == (1, 2, 3)
    assert [SETTING_SPECS[k].default for k in SETTING_SPECS] == [20, 300, 6, 0.3, 5, 3, 0.5, 2, C.SEED_DEFAULT]
    assert P.bounds("seed_input") == (0, 299) and P.bounds("customers_slider") == (None, None) and C.PENALTY_PFAC == {1: 0.01, 5: 0.05, 25: 0.25, 100: 1.0}


def test_parse_setting_snaps_to_the_nearest_stage_and_the_smaller_one_on_a_tie():
    spec = SETTING_SPECS["customers_slider"]
    assert P.parse_setting(spec, "10") == 10 and P.parse_setting(spec, "40") == 40 and P.parse_setting(spec, "24") == 20 and P.parse_setting(spec, "31") == 40
    assert P.parse_setting(spec, "15") == 10 and P.parse_setting(spec, "15.0001") == 20 and P.parse_setting(spec, "30") == 20 and P.parse_setting(spec, "-100") == 10 and P.parse_setting(spec, "1e9") == 40
    cap = SETTING_SPECS["capacity_slider"]
    assert P.parse_setting(cap, "170") == 150 and P.parse_setting(cap, "180") == 200 and P.parse_setting(cap, "375") == 300 and P.parse_setting(cap, "376") == 450
    sig = SETTING_SPECS["sigma_slider"]
    assert P.parse_setting(sig, "0.2") == 0.15 and P.parse_setting(sig, "0.9") == 1.0 and P.parse_setting(sig, "5") == 1.0 and P.parse_setting(sig, "0") == 0.0
    assert P.parse_setting(SETTING_SPECS["gamma_slider"], "0.4") == 0.5 and P.parse_setting(SETTING_SPECS["horizon_slider"], "5") == 4       # 5 liegt mittig zwischen 4 und 6: die kleinere


def test_parse_setting_clamps_the_seed_and_rejects_garbage():
    seed = SETTING_SPECS["seed_input"]
    assert P.parse_setting(seed, "17") == 17 and P.parse_setting(seed, "999") == 299 and P.parse_setting(seed, "-4") == 0 and P.parse_setting(seed, "12.6") == 13 and P.parse_setting(seed, "299") == 299
    assert isinstance(P.parse_setting(seed, "12.6"), int)
    for spec in SETTING_SPECS.values():
        for junk in ("abc", "", "nan", "inf", "-inf", None):
            assert P.parse_setting(spec, junk) is None, (spec.url_param, junk)


def test_encoders_write_short_numbers():
    assert P._num(0.15) == "0.15" and P._num(0.3) == "0.3" and P._num(1.0) == "1" and P._text(300) == "300"
    assert SETTING_SPECS["sigma_slider"].encoder(0.15) == "0.15" and SETTING_SPECS["customers_slider"].encoder(20) == "20"


def test_permalink_is_loaded_once_and_snaps(fake_state):
    state, params = fake_state
    params.update({"n": "31", "q": "170", "f": "1", "s": "0.2", "p": "26", "h": "5", "g": "0.4", "l": "3", "seed": "999"})
    P.load_permalink_settings()
    assert state["customers_slider"] == 40 and state["capacity_slider"] == 150 and state["fleet_slider"] == 1 and state["sigma_slider"] == 0.15 and state["penalty_slider"] == 25
    assert state["horizon_slider"] == 4 and state["gamma_slider"] == 0.5 and state["early_slider"] == 3 and state["seed_input"] == 299 and state["permalink_loaded"] is True
    state["customers_slider"] = 10
    params["n"] = "40"
    P.load_permalink_settings()
    assert state["customers_slider"] == 10                                                            # nur beim ersten Lauf


def test_permalink_ignores_unusable_values_and_missing_keys(fake_state):
    state, params = fake_state
    params.update({"n": "viele", "q": "600"})
    P.load_permalink_settings()
    assert "customers_slider" not in state and state["capacity_slider"] == 600 and "fleet_slider" not in state


def test_defaults_fill_only_missing_state(fake_state):
    state, _ = fake_state
    state["capacity_slider"] = 600
    P.init_session_state_defaults()
    assert state["capacity_slider"] == 600 and state["customers_slider"] == 20 and state["seed_input"] == C.SEED_DEFAULT and set(state) == set(SETTING_SPECS)


def test_sync_query_params_writes_the_encoded_values(fake_state):
    _, params = fake_state
    P.sync_query_params({"sigma_slider": 0.15, "capacity_slider": 450, "gamma_slider": 1.0, "seed_input": 211})
    assert params == {"s": "0.15", "q": "450", "g": "1", "seed": "211"}


def test_sync_query_params_survives_a_broken_query_string_api(monkeypatch):
    class Broken:
        def __setitem__(self, k, v):
            raise RuntimeError("kein Zugriff")
    monkeypatch.setattr(st, "query_params", Broken())
    P.sync_query_params({"seed_input": 3})                                                             # darf nicht abstürzen


def test_apply_preset_sets_every_control(fake_state):
    state, _ = fake_state
    for name, preset in C.PRESETS.items():
        P.apply_preset(name)
        assert {k: state[k] for k in SETTING_SPECS} == {PRESET_STATE_KEYS[f]: v for f, v in preset.items()}
    P.apply_preset("Zu großzügig")
    assert (state["capacity_slider"], state["horizon_slider"], state["gamma_slider"]) == (150, 12, 1.0)
    with pytest.raises(KeyError):
        P.apply_preset("gibt es nicht")


def test_randomize_seed_stays_in_range(fake_state, monkeypatch):
    state, _ = fake_state
    seen = []
    monkeypatch.setattr(random, "randint", lambda lo, hi: (seen.append((lo, hi)), 123)[1])
    P.randomize_seed()
    assert state["seed_input"] == 123 and seen == [(0, 299)]
    monkeypatch.undo()
    for _ in range(50):
        P.randomize_seed()
        assert 0 <= state["seed_input"] <= 299


def test_presets_are_five_short_named_measured_stages_with_help():
    assert list(C.PRESETS) == ["Standard", "Wagen fast voll", "Großer Wagen", "Zu großzügig", "Knappe Flotte"] and list(C.PRESET_HELP) == list(C.PRESETS)
    for name, p in C.PRESETS.items():
        assert len(name) <= 32 and len(C.PRESET_HELP[name]) > 20 and set(p) == set(PRESET_STATE_KEYS)
        for field, spec_key in PRESET_STATE_KEYS.items():
            spec = SETTING_SPECS[spec_key]
            assert (p[field] in spec.options) if spec.options else (spec.lo <= p[field] <= spec.hi), (name, field)
    assert C.PRESETS["Standard"] == dict(customers=20, capacity=300, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=C.SEED_DEFAULT)
    assert C.SEED_DEFAULT == 211 and len({p["seed"] for p in C.PRESETS.values()}) == 1
