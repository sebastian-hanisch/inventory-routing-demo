"""Abnahmekriterien der Presets an KÜNSTLICHEN Werten: jedes Kriterium kippt einzeln an seiner Schwelle (Kopie der Ergebnisdatei mit verstellten Werten),
Vorzeichen-Kriterien nur zusammen mit der Standardfehler-Bedingung. Die Abnahme der echten Presets steht in tests/test_preset_stories.py."""
import copy

import pytest

import irp_constants as C
import irp_results as R
import irp_stories as ST

DATA = R.load_results()


def crit(name, mutate=None):
    d = copy.deepcopy(DATA)
    if mutate:
        mutate(d)
    return [ok for ok, _ in ST.criteria(name, d)]


def texts(name):
    return [t for _, t in ST.criteria(name, DATA)]


def set_p(H, g, mean=None, se=None, loss=None, **cell_overrides):
    def m(d):
        p = R.p_gain(R.find_cell(d, **cell_overrides), H, g)
        if mean is not None:
            p["mean"] = mean
        if se is not None:
            p["se"] = se
        if loss is not None:
            p["share_loss"] = loss
    return m


def set_default(mean=None, se=None, loss=None, **cell_overrides):
    """P(3; 0,5) einer Zelle; `default` ist ein eigenes Objekt in der Datei, deshalb beide setzen."""
    inner = set_p(3, 0.5, mean, se, loss, **cell_overrides)

    def m(d):
        inner(d)
    return m


def set_e(L, mean, se, **cell_overrides):
    def m(d):
        e = R.e_gain(R.find_cell(d, **cell_overrides), L)
        e["mean"], e["se"] = mean, se
    return m


def test_all_real_criteria_hold_and_have_texts():
    for name in C.PRESETS:
        result = ST.criteria(name, DATA)
        assert result and all(ok for ok, _ in result), (name, result)
        assert all(isinstance(t, str) and len(t) > 10 for _, t in result)
    assert [len(ST.criteria(n, DATA)) for n in C.PRESETS] == [3, 1, 2, 2, 2]
    with pytest.raises(KeyError):
        ST.criteria("gibt es nicht", DATA)


# --- Standard ---------------------------------------------------------------------------------------------
def test_standard_gain_threshold_and_two_standard_errors():
    assert crit("Standard", set_default(10.0, 1.0))[0] is True and crit("Standard", set_default(9.99, 1.0))[0] is False
    assert crit("Standard", set_default(20.0, 10.0))[0] is False and crit("Standard", set_default(20.0, 9.99))[0] is True          # genau 2 SE: nicht belastbar
    assert crit("Standard", set_default(-20.0, 1.0))[0] is False and crit("Standard", set_default(0.0, 0.0))[0] is False


def test_standard_needs_a_loss_share_of_zero():
    assert crit("Standard", set_default(loss=0.0))[1] is True and crit("Standard", set_default(loss=0.005))[1] is False
    assert ST.STANDARD_MAX_LOSS_SHARE == 0.0 and ST.STANDARD_MIN_GAIN == 10.0


def test_standard_early_delivery_must_be_dearer_by_more_than_two_standard_errors():
    assert crit("Standard", set_e(2, -20.01, 10.0))[2] is True and crit("Standard", set_e(2, -20.0, 10.0))[2] is False
    assert crit("Standard", set_e(2, 5.0, 1.0))[2] is False and crit("Standard", set_e(1, -50.0, 0.1))[2] is True               # E(2) zählt, nicht E(1)
    assert ST.STANDARD_EARLY == 2


# --- Wagen fast voll -------------------------------------------------------------------------------------------------
def test_full_wagon_gain_must_be_within_two_percent_in_both_directions():
    for v, ok in ((1.99, True), (2.0, False), (-1.99, True), (-2.0, False), (0.0, True), (7.0, False)):
        assert crit("Wagen fast voll", set_default(v, Q=100.0)) == [ok], v
    assert ST.FULL_MAX_ABS_GAIN == 2.0


# --- Großer Wagen ------------------------------------------------------------------------------------------------------
def test_big_wagon_gain_threshold_and_two_standard_errors():
    assert crit("Großer Wagen", set_default(25.0, 1.0, Q=600.0))[0] is True and crit("Großer Wagen", set_default(24.99, 1.0, Q=600.0))[0] is False
    assert crit("Großer Wagen", set_default(30.0, 15.0, Q=600.0))[0] is False and crit("Großer Wagen", set_default(30.0, 14.9, Q=600.0))[0] is True


def test_big_wagon_must_beat_the_base_wagon_by_two_standard_errors_of_the_difference():
    base = R.p_gain(R.find_cell(DATA), 3, 0.5)
    margin = 2 * (base["se"] ** 2 + 0.3 ** 2) ** 0.5
    ok = base["mean"] + margin + 0.01
    assert crit("Großer Wagen", set_default(ok, 0.3, Q=600.0))[1] is True
    assert crit("Großer Wagen", set_default(base["mean"] + margin - 0.01, 0.3, Q=600.0))[1] is False
    assert crit("Großer Wagen", set_default(base["mean"] - 5, 0.3, Q=600.0))[1] is False


# --- Zu großzügig ----------------------------------------------------------------------------------------------------------
def test_generous_rule_must_be_dearer_by_ten_percent_and_two_standard_errors():
    assert crit("Zu großzügig", set_p(12, 1.0, -10.01, 0.1, Q=150.0))[0] is True and crit("Zu großzügig", set_p(12, 1.0, -10.0, 0.1, Q=150.0))[0] is False
    assert crit("Zu großzügig", set_p(12, 1.0, -10.5, 5.3, Q=150.0))[0] is False and crit("Zu großzügig", set_p(12, 1.0, -10.5, 5.2, Q=150.0))[0] is True
    assert crit("Zu großzügig", set_p(12, 1.0, 5.0, 0.1, Q=150.0))[0] is False and ST.GENEROUS_H == 12


def test_generous_rule_with_small_gamma_must_be_clearly_positive():
    assert crit("Zu großzügig", set_p(12, 0.25, 5.0, 2.4, Q=150.0))[1] is True and crit("Zu großzügig", set_p(12, 0.25, 5.0, 2.5, Q=150.0))[1] is False
    assert crit("Zu großzügig", set_p(12, 0.25, -1.0, 0.1, Q=150.0))[1] is False and ST.GENEROUS_GAMMA_OK == 0.25


# --- Knappe Flotte -----------------------------------------------------------------------------------------------------------
def test_tight_fleet_gain_threshold_and_two_standard_errors():
    assert crit("Knappe Flotte", set_default(15.0, 1.0, F=1))[0] is True and crit("Knappe Flotte", set_default(14.99, 1.0, F=1))[0] is False
    assert crit("Knappe Flotte", set_default(20.0, 10.0, F=1))[0] is False and crit("Knappe Flotte", set_default(20.0, 9.9, F=1))[0] is True


def test_tight_fleet_shortage_must_fall():
    def short(delta):
        def m(d):
            c = R.find_cell(d, F=1)
            c["short_P"] = c["short_R"] + delta
        return m
    assert crit("Knappe Flotte", short(-0.01))[1] is True and crit("Knappe Flotte", short(0.0))[1] is False and crit("Knappe Flotte", short(5.0))[1] is False


def test_criteria_texts_carry_the_measured_numbers():
    t = texts("Standard")
    assert "+17,0 ± 0,3 %" in t[0] and "0 %" in t[1] and "−16,3" not in t[2] and "-16,3 ± 0,2" in t[2]
    assert "+0,4 ± 0,1" in texts("Wagen fast voll")[0]
    b = texts("Großer Wagen")
    assert "+29,9 ± 0,3" in b[0] and "+29,9 gegen +17,0" in b[1]
    z = texts("Zu großzügig")
    assert "-26,3 ± 0,5" in z[0] and "+9,0 ± 0,2" in z[1]
    f = texts("Knappe Flotte")
    assert "+19,2 ± 0,3" in f[0] and "386 auf 130" in f[1]


# --- gezeigte Instanz ---------------------------------------------------------------------------------------------------------
def fake_live(gp=0.0, ge=0.0, short_r=0.0, short_p=0.0):
    """Künstliche Live-Instanz: nur die Felder, die day_criteria liest."""
    mk = lambda gain, short: {"gain": gain, "res": {"short": short}}
    return {"rules": {"R": mk(0.0, short_r), "P": mk(gp, short_p), "E": mk(ge, 0.0)}}


def dcrit(name, live, alt=None):
    return [ok for ok, _ in ST.day_criteria(name, live, alt)]


def test_day_criteria_standard():
    assert dcrit("Standard", fake_live(gp=0.1, ge=-0.1)) == [True, True]
    assert dcrit("Standard", fake_live(gp=0.0, ge=-0.1)) == [False, True] and dcrit("Standard", fake_live(gp=0.1, ge=0.0)) == [True, False]


def test_day_criteria_full_wagon():
    assert dcrit("Wagen fast voll", fake_live(gp=2.99)) == [True] and dcrit("Wagen fast voll", fake_live(gp=3.0)) == [False]
    assert dcrit("Wagen fast voll", fake_live(gp=-2.99)) == [True] and dcrit("Wagen fast voll", fake_live(gp=-3.0)) == [False]


def test_day_criteria_big_wagon():
    assert dcrit("Großer Wagen", fake_live(gp=15.0)) == [True] and dcrit("Großer Wagen", fake_live(gp=14.99)) == [False]


def test_day_criteria_generous():
    assert dcrit("Zu großzügig", fake_live(gp=-0.1), fake_live(gp=0.1)) == [True, True]
    assert dcrit("Zu großzügig", fake_live(gp=0.0), fake_live(gp=0.1)) == [False, True] and dcrit("Zu großzügig", fake_live(gp=-0.1), fake_live(gp=0.0)) == [True, False]
    with pytest.raises(ValueError, match="alt"):
        ST.day_criteria("Zu großzügig", fake_live(gp=-1.0))


def test_day_criteria_tight_fleet():
    assert dcrit("Knappe Flotte", fake_live(gp=5.0, short_r=10, short_p=9)) == [True, True]
    assert dcrit("Knappe Flotte", fake_live(gp=4.99, short_r=10, short_p=9)) == [False, True]
    assert dcrit("Knappe Flotte", fake_live(gp=5.0, short_r=10, short_p=10)) == [True, False]
    with pytest.raises(KeyError):
        ST.day_criteria("gibt es nicht", fake_live())


def test_day_criteria_texts_show_the_numbers():
    txt = [t for _, t in ST.day_criteria("Standard", fake_live(gp=12.3, ge=-4.5))]
    assert "+12,3 %" in txt[0] and "-4,5 %" in txt[1]
