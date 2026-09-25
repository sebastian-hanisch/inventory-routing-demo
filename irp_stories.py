"""Abnahmekriterien der Presets (Detailplan plan_inventory_routing.html, Abschnitt 7): welche Geschichte erzählt jedes Beispielszenario, und woran erkennt
man, dass sie trägt?

Einzige Quelle für tests/test_stories.py (Schwellen einzeln kippen) und tests/test_preset_stories.py (Abnahme der echten Presets). Die Kriterien der
Messreihe (`criteria`) stehen auf data/irp_results.json, Vorzeichen-Kriterien IMMER zusammen mit der Standardfehler-Bedingung (Betrag > 2 SE). Die gezeigte
Instanz (`day_criteria`) bleibt qualitativ, weil eine Instanz streut; ihr Seed liegt bewusst außerhalb der Stichprobe-Seeds der Messreihe (0 bis 199).
Kein Löser mit Wall-Clock-Grenze in den Kriterien (CP-SAT läuft nur im Exakt-Tab).

`criteria(name, data)` liefert für jedes Preset eine Liste (erfüllt, Text). Die Schwellen stehen als Konstanten oben, damit ein Test jede einzeln an ihre
Grenze schieben kann."""
import math

import irp_constants as C
import irp_results as R

# Standard: Bündeln spart mindestens 10 %, in keiner Instanz ein Verlust, früher liefern ist teurer als reaktiv
STANDARD_MIN_GAIN = 10.0
STANDARD_MAX_LOSS_SHARE = 0.0
STANDARD_EARLY = 2                     # E(2) als Gegenprobe
# Wagen fast voll: es gibt nichts zu bündeln
FULL_MAX_ABS_GAIN = 2.0
# Großer Wagen: großer Gewinn, größer als beim Basiswagen
BIG_MIN_GAIN = 25.0
# Zu großzügig: Wagen 150, H = 12, gamma = 1,0 kippt ins Negative, mit gamma = 0,25 nicht
GENEROUS_H = 12
GENEROUS_MAX_GAIN = -10.0
GENEROUS_GAMMA_OK = 0.25
# Knappe Flotte: hoher Gewinn und weniger Fehlmenge
FLEET_MIN_GAIN = 15.0

FACTOR = C.SE_FACTOR

# --- gezeigte Instanz (qualitativ) ---------------------------------------------------------------------------------
DAY_FULL_MAX_ABS_GAIN = 3.0            # Wagen fast voll: |Gewinn| auf der Instanz unter 3 % (Messreihe im Mittel +0,4)
DAY_BIG_MIN_GAIN = 15.0                # Großer Wagen: Gewinn auf der Instanz mindestens 15 % (Messreihe im Mittel +29,9)
DAY_FLEET_MIN_GAIN = 5.0               # Knappe Flotte: Gewinn auf der Instanz mindestens 5 % (Messreihe im Mittel +19,2)


def _de(v, digits=1, sign=False):
    text = f"{v:+.{digits}f}" if sign else f"{v:.{digits}f}"
    return text.replace(".", ",")


def _band(mean, se):
    return f"{_de(mean, 1, True)} ± {_de(se)}"


def _clear(mean, se, sign=+1):
    """Vorzeichen-Bedingung mit Standardfehler: sign +1 (Mittel > 2 SE) oder −1 (Mittel < −2 SE)."""
    return mean > FACTOR * se if sign > 0 else mean < -FACTOR * se


def _default(cell):
    return R.p_gain(cell, C.DEFAULT_H, C.DEFAULT_GAMMA)


def criteria(name, data):
    """Abnahmekriterien des Presets `name` auf der Messreihe `data`: Liste (erfüllt, Text)."""
    if name == "Standard":
        cell = R.find_cell(data)
        g, e = _default(cell), R.e_gain(cell, STANDARD_EARLY)
        return [
            (_clear(g["mean"], g["se"]) and g["mean"] >= STANDARD_MIN_GAIN, f"Bündeln P(3; 0,5) über 2 Standardfehler und mindestens 10 %: {_band(g['mean'], g['se'])} %"),
            (g["share_loss"] <= STANDARD_MAX_LOSS_SHARE, f"Anteil der Instanzen mit Verlust 0 %: {_de(100 * g['share_loss'], 0)} %"),
            (_clear(e["mean"], e["se"], -1), f"früher liefern E(2) teurer als reaktiv (unter −2 Standardfehler): {_band(e['mean'], e['se'])} %"),
        ]
    if name == "Wagen fast voll":
        cell = R.find_cell(data, Q=100.0)
        g = _default(cell)
        return [(abs(g["mean"]) < FULL_MAX_ABS_GAIN, f"|Gewinn| unter 2 %: {_band(g['mean'], g['se'])} %")]
    if name == "Großer Wagen":
        cell, base = R.find_cell(data, Q=600.0), R.find_cell(data)
        g, b = _default(cell), _default(base)
        margin = FACTOR * math.hypot(g["se"], b["se"])
        return [
            (g["mean"] >= BIG_MIN_GAIN and _clear(g["mean"], g["se"]), f"Gewinn mindestens 25 % und über 2 Standardfehler: {_band(g['mean'], g['se'])} %"),
            (g["mean"] - b["mean"] > margin, f"größer als bei Wagen 300 (um mehr als 2 Standardfehler der Differenz): {_de(g['mean'], 1, True)} gegen {_de(b['mean'], 1, True)} %"),
        ]
    if name == "Zu großzügig":
        cell = R.find_cell(data, Q=150.0)
        g, ok = R.p_gain(cell, GENEROUS_H, 1.0), R.p_gain(cell, GENEROUS_H, GENEROUS_GAMMA_OK)
        return [
            (g["mean"] < GENEROUS_MAX_GAIN and _clear(g["mean"], g["se"], -1), f"Gewinn unter −10 % und unter −2 Standardfehler: {_band(g['mean'], g['se'])} %"),
            (_clear(ok["mean"], ok["se"]), f"mit γ = 0,25 dagegen positiv (über 2 Standardfehler): {_band(ok['mean'], ok['se'])} %"),
        ]
    if name == "Knappe Flotte":
        cell = R.find_cell(data, F=1)
        g = _default(cell)
        return [
            (g["mean"] >= FLEET_MIN_GAIN and _clear(g["mean"], g["se"]), f"Gewinn mindestens 15 % und über 2 Standardfehler: {_band(g['mean'], g['se'])} %"),
            (cell["short_P"] < cell["short_R"], f"Fehlmenge sinkt: {_de(cell['short_R'], 0)} auf {_de(cell['short_P'], 0)} Einheiten"),
        ]
    raise KeyError(name)


def day_criteria(name, live, alt=None):
    """Qualitative Kriterien an der gezeigten Instanz `live` (irp_live.solve_live): eine Instanz streut, deshalb nur die Richtung der Geschichte.
    `alt` ist für „Zu großzügig“ dieselbe Instanz mit gamma = 0,25 (sonst nicht gebraucht)."""
    rules = live["rules"]
    gp, ge = rules["P"]["gain"], rules["E"]["gain"]
    if name == "Standard":
        return [(gp > 0, f"Bündeln billiger als reaktiv: {_de(gp, 1, True)} %"), (ge < 0, f"früher liefern teurer als reaktiv: {_de(ge, 1, True)} %")]
    if name == "Wagen fast voll":
        return [(abs(gp) < DAY_FULL_MAX_ABS_GAIN, f"|Gewinn| auf der Instanz unter 3 %: {_de(gp, 1, True)} %")]
    if name == "Großer Wagen":
        return [(gp >= DAY_BIG_MIN_GAIN, f"Gewinn auf der Instanz mindestens 15 %: {_de(gp, 1, True)} %")]
    if name == "Zu großzügig":
        if alt is None:
            raise ValueError("Zu großzügig braucht die Vergleichsinstanz mit gamma = 0,25 (alt)")
        ga = alt["rules"]["P"]["gain"]
        return [(gp < 0, f"Bündeln teurer als reaktiv: {_de(gp, 1, True)} %"), (ga > 0, f"mit γ = 0,25 dagegen billiger: {_de(ga, 1, True)} %")]
    if name == "Knappe Flotte":
        sr, sp = rules["R"]["res"]["short"], rules["P"]["res"]["short"]
        return [(gp >= DAY_FLEET_MIN_GAIN, f"Gewinn auf der Instanz mindestens 5 %: {_de(gp, 1, True)} %"),
                (sp < sr, f"Fehlmenge auf der Instanz sinkt: {_de(sr, 0)} auf {_de(sp, 0)} Einheiten")]
    raise KeyError(name)
