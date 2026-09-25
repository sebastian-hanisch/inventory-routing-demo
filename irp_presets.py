"""Regler-Spezifikation, Permalink, Presets und Seed-Knopf (Standardmuster aus dem OR-Demo-Portfolio, siehe nv_presets.py in nahverkehr-demo).
Acht Regler sind feste Stufen (Kunden, Wagenkapazität, Touren je Tag, Verbrauchsschwankung, Fehlmengenstrafe, Vorschau H, Mitnahmeschwelle gamma,
früher liefern L), einer ist ein Zahlenbereich (Seed): beim Permalink wird jeder Wert auf den Bereich begrenzt bzw. auf die nächste Stufe eingerastet,
damit die Adresszeile nie einen Wert außerhalb des Rasters in den Regler schreibt."""
import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import irp_constants as C


def _text(value):
    return str(value)


def _num(value):
    return f"{value:g}"


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None
    options: Optional[tuple] = None          # feste Stufen: auf die nächste Stufe einrasten
    encoder: Callable = _text


SETTING_SPECS = {
    "customers_slider": SettingSpec("n", int, C.CUSTOMER_DEFAULT, options=C.CUSTOMER_OPTIONS),
    "capacity_slider": SettingSpec("q", int, C.CAPACITY_DEFAULT, options=C.CAPACITY_OPTIONS),
    "fleet_slider": SettingSpec("f", int, C.FLEET_DEFAULT, options=C.FLEET_OPTIONS),
    "sigma_slider": SettingSpec("s", float, C.SIGMA_DEFAULT, options=C.SIGMA_OPTIONS, encoder=_num),
    "penalty_slider": SettingSpec("p", int, C.PENALTY_DEFAULT, options=C.PENALTY_OPTIONS),
    "horizon_slider": SettingSpec("h", int, C.HORIZON_DEFAULT, options=C.HORIZON_OPTIONS),
    "gamma_slider": SettingSpec("g", float, C.GAMMA_DEFAULT, options=C.GAMMA_OPTIONS, encoder=_num),
    "early_slider": SettingSpec("l", int, C.EARLY_DEFAULT, options=C.EARLY_OPTIONS),
    "seed_input": SettingSpec("seed", int, C.SEED_DEFAULT, *C.SEED_RANGE),
}

PRESET_STATE_KEYS = {
    "customers": "customers_slider", "capacity": "capacity_slider", "fleet": "fleet_slider", "sigma": "sigma_slider", "penalty": "penalty_slider",
    "horizon": "horizon_slider", "gamma": "gamma_slider", "early": "early_slider", "seed": "seed_input",
}


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def parse_setting(spec, raw):
    """Wert aus der Adresszeile: umwandeln, auf den Bereich begrenzen bzw. auf die nächste Stufe einrasten. None, wenn er sich nicht auswerten
    lässt (kein Zahlenwert oder nicht endlich)."""
    try:
        value = float(raw)
    except (ValueError, TypeError):
        return None
    if not math.isfinite(value):
        return None
    if spec.options:                                        # Zahlenstufen: die nächste Stufe, bei Gleichstand die kleinere
        return min(spec.options, key=lambda o: (abs(o - value), o))
    return int(round(max(spec.lo, min(spec.hi, value))))


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            value = parse_setting(spec, qp[spec.url_param])
            if value is not None:
                st.session_state[state_key] = value
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """values: dict state_key -> aktueller Wert (aus den Widgets)."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = SETTING_SPECS[state_key].encoder(value)
    except Exception:
        pass


def apply_preset(name):
    for field, state_key in PRESET_STATE_KEYS.items():
        st.session_state[state_key] = C.PRESETS[name][field]


def randomize_seed():
    """Würfelt einen neuen Seed für die gezeigte Instanz."""
    st.session_state["seed_input"] = random.randint(*C.SEED_RANGE)
