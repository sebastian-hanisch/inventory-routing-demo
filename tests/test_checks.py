"""Die Korrektheits-Checks der Messreihe (check.py, check_oracle.py) in verkleinerter Fassung; das volle Bau-Gate ist tools/check_full.py. Die Checks mit Zufallsinstanzen
prüfen nur Invarianten, die für jede Instanz gelten (die Ströme von default_rng dürfen sich zwischen NumPy-Versionen ändern); Zahlen stehen auf den eingefrorenen Instanzen."""
import numpy as np
import pytest

import irp_checks as K
import irp_frozen as FZ
import irp_policy as P


def test_no_consumption_means_no_tours():
    assert K.check_no_consumption(12)


def test_h_zero_is_reactive_for_every_gamma():
    assert K.check_h0_equals_reactive(n_seeds=6, days=30) == 24


def test_independent_reevaluation_and_exact_inventory_balance():
    assert K.check_balance(n_seeds=4, days=30) == 4 * 5 * 4


def test_hand_instance_tour_costs_14_24_18():
    assert K.check_hand_instance() == pytest.approx(18.0)


def test_savings_and_two_opt_are_never_better_than_the_exact_cvrp():
    mean, median, p90, worst, share_opt = K.check_savings_vs_brute_force(n_inst=30)
    assert mean >= -1e-9 and worst >= median >= -1e-9 and 0.0 <= share_opt <= 100.0 and worst < 30.0


def test_every_rule_branch_fires_on_random_instances():
    out = K.check_branches(n_seeds=4, days=60)
    assert out["short_sigma0"] == 0.0 and out["short_sigma1"] > 0.0


def test_shortage_share_is_a_measurement_not_a_rule():
    rows = K.check_shortage_share(n_seeds=5, days=60, configs=[dict(sigma=0.0), dict(sigma=0.6)])
    assert rows[0][1] == 0 and 0 <= rows[1][1] <= 5                     # sigma = 0: nie mehr Fehlmenge (Fehlmenge ist dort exakt 0), sigma = 0,6: keine harte Regel


def test_determinism_and_stream_independent_of_the_rule():
    assert K.check_determinism()


def test_cp_sat_model_equals_the_independent_exact_dp():
    pytest.importorskip("ortools")
    n = K.check_oracle_vs_dp(seeds=range(16), min_feasible=6)
    assert n >= 6


# --- auf den eingefrorenen Instanzen (unabhängig vom Zufallsgenerator) -----------------------------------------------------------
@pytest.mark.parametrize("config", ["base", "sigma0", "q100", "q600", "n40", "flotte_F1_Q300", "cluster", "depot_far", "kappa2"])
def test_independent_reevaluation_on_frozen_instances(config):
    inst = FZ.frozen(config, 0)
    for H, g, e in [(0, 0.5, 0.0), (3, 0.5, 0.0), (12, 1.0, 0.0), (0, 0.5, 3.0)]:
        r = P.simulate(inst, H, g, early=e, record=True)
        rt, sh, dl, sv, iend = K.indep_eval(inst, r["plans"])
        assert rt == pytest.approx(r["routing"], rel=1e-9, abs=1e-9) and sh == pytest.approx(r["short"], abs=1e-6)
        assert dl == pytest.approx(r["delivered"], abs=1e-6) and sv == pytest.approx(r["served"], abs=1e-6) and iend == pytest.approx(r["I_end"], abs=1e-6)
        assert r["I_end"] == pytest.approx(r["I_start"] + r["delivered"] - r["served"], abs=1e-6)
        assert r["served"] + r["short"] == pytest.approx(inst.cons.sum(), abs=1e-6)


def test_no_shortage_at_sigma_zero_is_correct_not_a_bug():
    """sigma = 0: die Meldegrenze mu*(1 + kappa*0) deckt den Tagesverbrauch mu exakt (der Kunde wird beliefert, sobald der Bestand unter mu fällt), und der Wagen
    (Q = 300) fasst jede Menge (C <= 196): Fehlmenge exakt 0 und Kunden-Tage mit Fehlmenge 0. Bei sigma = 0,3 dagegen entstehen Fehlmengen: die Spalte ist keine tote Spalte."""
    for s in (0,):
        r0 = P.simulate(FZ.frozen("sigma0", s), 0.0)
        assert r0["short"] == 0.0 and r0["stockout_days"] == 0
    shorts = [P.simulate(FZ.frozen("base", s), 0.0)["short"] for s in (0, 1, 2)]
    assert sum(shorts) > 0.0
    assert P.simulate(FZ.frozen("base", 0), 3.0, 0.5)["short"] >= 0.0


def test_tight_fleet_defers_customers_and_shortage_arises():
    inst = FZ.frozen("flotte_F1_Q300", 0)
    r = P.simulate(inst, 0.0)
    assert r["deferred"] > 0 and r["short"] > 0 and r["stockout_days"] > 0
    assert max(len(p) for p in P.simulate(inst, 0.0, record=True)["plans"]) <= inst.F == 1
    free = P.simulate(FZ.frozen("base", 0), 0.0)
    assert free["deferred"] == 0


def test_gamma_steers_the_number_of_pickups_on_a_frozen_instance():
    inst = FZ.frozen("base", 0)
    picks = [P.simulate(inst, 3.0, g)["opt_visits"] for g in (0.1, 0.25, 0.5, 1.0)]
    assert picks == sorted(picks) and picks[0] < picks[-1]
    assert np.all(np.diff([P.simulate(inst, h, 0.5)["opt_visits"] for h in (1, 2, 3, 4, 6, 8, 12)]) >= 0)      # mehr Vorschau, mehr Mitnahmen
