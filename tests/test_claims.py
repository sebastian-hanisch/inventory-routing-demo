"""Jede Zahl der README (Abschnitte „Befunde", „Modell", „Ehrliche Grenzen", „Befunde und Korrekturen") wird hier aus data/irp_results.json nachgerechnet, dazu Aufbau der README
(Reihenfolge der Abschnitte, Nachbardemos, Dateistruktur gegen die echten Dateien). Die Vorzeichen stehen in der README als Minuszeichen (U+2212); die Formate der App nutzen den
Bindestrich, deshalb übersetzt `mn`. Zahlen, die auf frisch gewürfelten Instanzen beruhen (Bestätigung auf den Seeds 200 bis 299, Sensitivität von τ, Preset-Seeds), stehen nur mit der
NumPy-Version exakt fest, mit der sie gemessen wurden; mit einer anderen Version gelten Bänder (die CI installiert immer das neueste NumPy)."""
import os
import pathlib
import re

import numpy as np
import pytest

import irp_constants as C
import irp_format as F
import irp_frozen as FZ
import irp_live as LV
import irp_model as M
import irp_policy as P
import irp_results as R
from toolload import load_tool

ROOT = pathlib.Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")
FLAT = re.sub(r"\s+", " ", README)                                              # Absätze der README sind hart umbrochen
DATA = R.load_results()
BASE = R.find_cell(DATA)
CD = load_tool("confirm_default")
TP = load_tool("tune_presets")
SAME_NUMPY = np.__version__ == FZ.reference()["numpy"]
MUTATION_RUN = os.environ.get("IRP_MUTATION_RUN") == "1"


def mn(text):
    """ASCII-Bindestrich vor Ziffern als Minuszeichen (U+2212), wie in der README."""
    return re.sub(r"-(?=\d)", "−", text)


def has(text):
    assert mn(text) in FLAT, mn(text)


def dflt(cell):
    return R.p_gain(cell, 3, 0.5)


def band(mean, se, digits=1, signed=True):
    return F.fmt_band(mean, se, digits, signed)


def pct(v):
    return f"{100 * v:.0f}"


# ---------------------------------------------------------------------------------------------------
# Aufbau
# ---------------------------------------------------------------------------------------------------
def test_readme_sections_come_in_the_portfolio_order():
    heads = re.findall(r"^## (.+)$", README, flags=re.M)
    order = ["Warum dieses Problem", "Befunde und Korrekturen gegenüber dem Plan", "Modell", "Methodik", "Befunde (gemessen, keine Behauptungen)", "Ehrliche Grenzen", "Tests", "Dateistruktur",
             "Bewusst nicht umgesetzt", "Reproduktion der Messreihe", "Lokal ausführen", "Verwandte Demos mit demselben mathematischen Modell"]
    assert heads == order
    lines = README.splitlines()
    assert lines[0] == "# Inventory Routing: Wer gehört heute auf die Tour? – Streamlit-Demo" and lines[2] == "*(noch nicht deployed)*" and "sebastianhanisch-inventory-routing-demo.streamlit.app" not in README
    assert README.rstrip().endswith("Gebaut mit Streamlit, Plotly, OR-Tools und fpdf2.")


def test_readme_lists_the_neighbour_demos_of_the_model_register_entry():
    section = README.split("## Verwandte Demos mit demselben mathematischen Modell")[1]
    for needle in ("Stand 2026-09-24", "`vrp_demo`", "`alns-demo`", "`vrp-nachbarschaften-demo`", "`leercontainer-demo`", "erstes Exemplar", "`nahverkehr-demo`", "`fernverkehr-demo`", "Kein „starr gegen reaktiv“".replace("“", '"'),
                   "Tourenplanungs-Zweig", "Vorschau-Fenster-Muster"):
        assert needle in section or needle.replace('"', "“") in section, needle


def test_every_file_in_the_structure_table_exists_and_every_module_is_listed():
    table = README.split("## Dateistruktur")[1].split("## Bewusst nicht umgesetzt")[0]
    for name in re.findall(r"`([A-Za-z0-9_./-]+\.(?:py|json|txt|yml))`", table):
        assert (ROOT / name).exists() or (ROOT / "tools" / name).exists() or (ROOT / "tests" / name).exists() or name in ("raw_sweep.npz", "ir.py") or (ROOT / ".github" / "workflows" / name).exists(), name
    for f in sorted(ROOT.glob("irp_*.py")) + [ROOT / "app.py"] + sorted((ROOT / "tools").glob("*.py")):
        assert f.name in table, f.name


def test_readme_has_no_markdown_links_to_local_files():
    for target in re.findall(r"\]\(([^)]+)\)", README):
        assert target.startswith("https://"), target


def test_readme_uses_real_umlauts():
    for bad in (" fuer ", " ueber ", "Aenderung", "Loesung", "Groesse", "Bestaende"):
        assert bad not in README, bad


# ---------------------------------------------------------------------------------------------------
# Befunde und Korrekturen (AP 0)
# ---------------------------------------------------------------------------------------------------
def test_ap0_coverage_numbers():
    cov = R.coverage(DATA)
    has(f"**{cov['exact']}** der 27 Zellen".replace("**19** der 27 Zellen", "**19** der 27 Zellen"))
    has(f"{3} · {7} · {3} · {5} · {4} = {cov['total']} Kombinationen")
    has(f"deshalb sind **{cov['exact']} von {cov['total']}** Kombinationen exakt gemessen ({cov['by_distance'][1]} weichen in einem Parameter ab, {cov['by_distance'][2]} in zweien, "
        f"{cov['by_distance'][3]} in dreien, {cov['by_distance'][4]} in vieren)")
    assert len(cov["unreachable"]) == 8 and "**Acht Zellen sind nicht über die Regler erreichbar**" in FLAT
    has(f"nur {cov['exact']} von {cov['total']} Kombinationen sind exakt gemessen")


def test_ap0_oracle_correction_and_uniqueness_of_the_plan():
    s250 = R.oracle_summary(DATA, 250)
    has(f"ergeben **{F.fmt_num(s250['routes']['optimum'], 2)}**")
    assert "nennt für Wagen 250 im Optimum 2,50 Touren" in FLAT and "Seed 19" in FLAT and "11 statt 12 Besuche" in FLAT and "**der Zielwert ist eindeutig, der Plan nicht**" in FLAT
    assert "4 bis 25 s" in FLAT and "0,3 bis 1,2 s" in FLAT and "8 Arbeitern" in FLAT and "20 s" in FLAT


def test_ap0_seed_range_and_display_choices_are_documented_as_deviations():
    assert "Seed-Bereich der Instanz ist 0 bis 299" in FLAT and C.SEED_RANGE == (0, 299)
    assert "Bündeln neben Reaktiv" in FLAT and "Früher liefern neben Reaktiv" in FLAT
    assert "17,0 / 17,0 / 17,1 %" in FLAT


@pytest.mark.skipif(MUTATION_RUN, reason="rechnet 100 frische Instanzen")
def test_ap0_default_rule_on_fresh_seeds():
    res = CD.confirm({}, seeds=range(200, 300))
    if SAME_NUMPY:
        has(f"**{band(res['mean'], res['se'])} %** (Median {F.fmt_num(res['median'])} %, Quartile [{F.fmt_num(res['q1'])}; {F.fmt_num(res['q3'])}], Minimum {F.fmt_num(res['min'])}, Maximum {F.fmt_num(res['max'])}, "
            f"in {F.fmt_num(100 * res['share_loss'], 0)} % der Instanzen ein Verlust) gegen +17,0 ± 0,3 % im Sweep")
        sweep = dflt(BASE)
        diff = res["mean"] - sweep["mean"]
        se_diff = (sweep["se"] ** 2 + res["se"] ** 2) ** 0.5
        has(f"die Differenz der Mittel ist {F.fmt_num(diff, 1, True)} % ({F.fmt_num(diff / se_diff, 1, True)} Standardfehler der Differenz)")
        q100, q600, f1 = (CD.confirm(o, seeds=range(200, 300)) for o in ({"Q": 100.0}, {"Q": 600.0}, {"F": 1, "Q": 300.0}))
        has(f"Wagen 100 ({band(q100['mean'], q100['se'])} %)")
        has(f"Wagen 600 ({band(q600['mean'], q600['se'])} %)")
        has(f"die knappe Flotte ({band(f1['mean'], f1['se'])} %)")
    else:
        assert 15.5 < res["mean"] < 18.5 and res["share_loss"] == 0.0 and abs(res["mean"] - dflt(BASE)["mean"]) < 3 * (res["se"] ** 2 + dflt(BASE)["se"] ** 2) ** 0.5


@pytest.mark.skipif(MUTATION_RUN or not SAME_NUMPY, reason="Preset-Seeds sind mit dieser NumPy-Version abgestimmt (100 Seeds x 5 Presets)")
def test_ap0_preset_seed_search_numbers():
    ok = 0
    for seed in range(200, 300):
        if all(v[0] for v in TP.evaluate(seed).values()):
            ok += 1
    assert f"{ok} von 100 Seeds erfüllen" in FLAT and "**Seed 211**" in FLAT and C.SEED_DEFAULT == 211
    live = LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 211)
    assert f"{live['rules']['P']['gain']:+.1f}".replace(".", ",") == "+17,0" and "mit +17,0 % genau auf dem Mittel der Messreihe" in FLAT


# ---------------------------------------------------------------------------------------------------
# Befunde
# ---------------------------------------------------------------------------------------------------
def test_finding_1_size_and_mechanism():
    b, d = dflt(BASE), DATA["_meta"]["base_distribution"]
    has(f"**{band(b['mean'], b['se'])} %** Fahrkosten (Median {F.fmt_num(b['median'])} %, Quartile [{F.fmt_num(b['q1'])}; {F.fmt_num(b['q3'])}], Minimum {F.fmt_num(d['min'])}, Maximum {F.fmt_num(d['max'])}; in **0 %** der 200 Instanzen ein Verlust")
    assert b["share_loss"] == 0.0 and d["loss_share"] == 0.0
    has(f"beste Zelle des Gitters (H = 8, γ = 0,25) bringt {F.fmt_num(BASE['cv_gain'], 1, True)} %")
    m = R.mechanism(BASE)
    has(f"Touren {F.fmt_num(m['routes'][0])} → {F.fmt_num(m['routes'][1])} (−{100 * (1 - m['routes'][1] / m['routes'][0]):.0f} %)")
    has(f"Besuche {F.fmt_num(m['visits'][0])} → {F.fmt_num(m['visits'][1])} (+{100 * (m['visits'][1] / m['visits'][0] - 1):.0f} %, davon {F.fmt_num(m['opt_visits'])} Mitnahmen)")
    has(f"Auslastung {F.fmt_num(m['util'][0], 2)} → {F.fmt_num(m['util'][1], 2)}")
    has(f"Fahr-km {F.fmt_int(m['km'][0])} → {F.fmt_int(m['km'][1])} (−{F.fmt_num(100 * (1 - m['km'][1] / m['km'][0]))} %)")


def test_finding_2_capacity_lever():
    rows = R.capacity_rows(DATA, 3, 0.5)
    has("Q = 100 / 150 / 200 / 300 / 450 / 600 / 1000 (Q/q = " + " / ".join(F.fmt_num(r["Q_over_q"]) for r in rows) + ")")
    has("**" + " / ".join(F.fmt_num(r["gain"], 1, True) for r in rows) + " %**")
    has("beste Zelle kreuzvalidiert " + " / ".join(F.fmt_num(r["cv_gain"], 1, True) for r in rows) + " %")
    rho = DATA["_meta"]["spearman"]
    has(f"über {rho['n']} Konfigurationen: {F.fmt_num(rho['rho'], 2)}")


def test_finding_3_uncertainty_does_not_matter():
    u = R.uncertainty_rows(DATA, 3, 0.5)
    has("σ = 0 / 0,15 / 0,3 / 0,6 / 1,0: **" + " / ".join(F.fmt_num(r["gain"], 1, True) for r in u["sigma"]) + " %**")
    has("Fehlmengenstrafe 1 / 5 / 25 / 100: " + " / ".join(F.fmt_num(r["gain"], 1, True) for r in u["penalty"]) + " %")


def test_finding_4_tipping_zone():
    gs = R.grid_summary(BASE)
    has(f"{gs['positive']} von {gs['n']} Gitterkombinationen billiger als reaktiv, {gs['near_best']} liegen höchstens 3 Prozentpunkte unter der besten")
    neg = R.negative_summary(DATA)
    has(f"**{neg[0]} von {neg[1]}** proaktiven Kombinationen im Mittel teurer als reaktiv, davon **{neg[2]} mit γ = 1,0** und keine mit γ ≤ 0,25")
    assert neg[3] == 0
    w = R.negative_pairs(DATA)[0]
    f2 = R.by_name(DATA, "flotte_F2_Q150")
    has(f"Wagen 150 mit zwei Touren, H = 12, γ = 1: {band(w[2], w[3])} %")
    has(f"Wagen 150 (H = 12, γ = 1): {F.fmt_num(R.p_gain(R.find_cell(DATA, Q=150.0), 12, 1.0)['mean'])} %")
    has(f"Wagen 100: {F.fmt_num(R.p_gain(R.find_cell(DATA, Q=100.0), 12, 1.0)['mean'])} %")
    assert w[0] is f2 and w[1] == "P_H12_g1.00"


def test_finding_5_early_delivery_is_no_alternative():
    e = [R.e_gain(BASE, L)["mean"] for L in (1, 2, 3)]
    has("L = 1 / 2 / 3 Tage: **" + " / ".join(F.fmt_num(v, 1, True) for v in e) + " %**")
    has(f"in {DATA['_meta']['E_negative_pairs'][0]} von {DATA['_meta']['E_negative_pairs'][1]} Regel-Zellen-Paaren teurer")
    m = R.mechanism(BASE)
    has(f"mehr Besuche ohne Tourenbezug ({F.fmt_num(m['visits'][2])} gegen {F.fmt_num(m['visits'][0])}), keine eingesparte Tour ({F.fmt_num(m['routes'][2])} gegen {F.fmt_num(m['routes'][0])}), Fahr-km {F.fmt_int(m['km'][2])} gegen {F.fmt_int(m['km'][0])}")
    has(f"vermeidet dafür in der Basis jede Fehlmenge ({F.fmt_num(m['short'][2])} gegen {F.fmt_num(m['short'][0])} Einheiten)")


def test_finding_6_tight_fleet():
    rows = {r["cell"]["name"]: r for r in R.shortage_rows(DATA)}
    f1, f250, f2 = rows["flotte_F1_Q300"], rows["flotte_F1_Q250"], rows["flotte_F2_Q150"]
    has(f"Wagen 300): **{band(f1['gain'], f1['se'])} %**, Fehlmenge {F.fmt_num(f1['short_R'], 0)} → {F.fmt_num(f1['short_P'], 0)} Einheiten")
    has(f"Wagen 250: {band(f250['gain'], f250['se'])} %, Fehlmenge {F.fmt_num(f250['short_R'], 0)} → {F.fmt_num(f250['short_P'], 0)}")
    has(f"zwei Touren mit Wagen 150: {band(f2['gain'], f2['se'])} % ({F.fmt_num(f2['short_R'], 0)} → {F.fmt_num(f2['short_P'], 0)})")
    has(f"(14,4 → 8,5 Einheiten), **aber in {F.fmt_num(100 * BASE['short_cmp']['more'])} % der Instanzen")
    has(f"in {F.fmt_num(100 * BASE['short_cmp']['more'])} % der Basis-Instanzen ist sie höher")


def test_finding_7_exact_benchmark():
    s150, s250 = R.oracle_summary(DATA, 150), R.oracle_summary(DATA, 250)
    g = lambda s, k: band(s["gap"][k]["mean"], s["gap"][k]["se"], 1, False)
    has(f"reaktiv liegt **{g(s150, 'R')} %** (Wagen 150) bzw. **{g(s250, 'R')} %** (Wagen 250) über dem Optimum")
    has(f"P(3; 0,5) {g(s150, 'P_H3_g0.50')} % bzw. {g(s250, 'P_H3_g0.50')} %")
    has(f"die Regel schließt **{pct(s150['closed_P3'])} % bzw. {pct(s250['closed_P3'])} %** der reaktiven Lücke")
    has(f"{F.fmt_num(s150['routes']['optimum'], 2)} / {F.fmt_num(s150['routes']['R'], 2)} / {F.fmt_num(s150['routes']['P3'], 2)} bzw. {F.fmt_num(s250['routes']['optimum'], 2)} / {F.fmt_num(s250['routes']['R'], 2)} / {F.fmt_num(s250['routes']['P3'], 2)}")
    assert s150["n_optimal"] == s250["n_optimal"] == 30 and "30 Instanzen je Wagen, alle bewiesen optimal" in FLAT


# ---------------------------------------------------------------------------------------------------
# Modell, Grenzen, Methodik
# ---------------------------------------------------------------------------------------------------
def test_model_numbers_match_the_constants():
    assert C.BASE_CFG["N"] == 20 and C.BASE_CFG["Q"] == 300.0 and C.BASE_CFG["F"] == 6 and C.BASE_CFG["sigma"] == 0.3 and C.PENALTY_DEFAULT == 5 and C.DAYS == 120 and C.BASE_CFG["kappa"] == 1.0
    for text in ("20 Kunden gleichverteilt in 100 × 100", "Wagenkapazität Q = 300, bis zu F = 6 Touren je Tag", "Strafe 5 je Einheit", "über 120 Tage", "κ = 1", "Gitter H ∈ {1, 2, 3, 4, 6, 8, 12} Tage, γ ∈ {0,1; 0,25; 0,5; 1,0}, Standard P(3; 0,5)",
                 "L ∈ {1, 2, 3} Tage", "7 Kunden, 6 Tage", "Wagen 150 bzw. 250", "27 Konfigurationen × 32 Regeln × 200 Instanzen × 120 Tage", "mindestens 5 % und über 2 Standardfehler", "unter −2 Standardfehler"):
        assert text in FLAT, text
    assert tuple(C.HORIZON_OPTIONS) == (1, 2, 3, 4, 6, 8, 12) and tuple(C.GAMMA_OPTIONS) == (0.1, 0.25, 0.5, 1.0) and tuple(C.EARLY_OPTIONS) == (1, 2, 3) and len(R.cells(DATA)) == 27


def test_limits_numbers():
    assert "mittlere Lücke 0,12 %, Median 0, Maximum 6,13 %, in 93 % der Instanzen exakt optimal" in FLAT
    b = dflt(BASE)
    has(f"Basis: Quartile {F.fmt_num(b['q1'])} bis {F.fmt_num(b['q3'])} %")
    assert "nur 19 von 1260 Kombinationen sind exakt gemessen" in FLAT and "12 bis 13 %" in FLAT
    assert "τ = 0,6" in FLAT and "22,5 %" in FLAT and "keine Garantie".lower() in FLAT.lower()


@pytest.mark.skipif(MUTATION_RUN, reason="rechnet 200 Instanzen")
def test_tau_sensitivity_of_the_base_gain():
    """17,0 / 17,0 / 17,1 % bei tau = 0 / tau_ref / 2 · tau_ref auf den Seeds 0 bis 199."""
    reactive, bundling = [], []
    for s in range(200):
        inst = M.make_instance(s, D=120)
        reactive.append(P.simulate(inst, 0.0, 0.5))
        bundling.append(P.simulate(inst, 3.0, 0.5))
    tau_ref = sum(r["routing"] for r in reactive) / sum(r["delivered"] for r in reactive)
    out = []
    for f in (0.0, 1.0, 2.0):
        jr = np.array([r["routing"] + 5.0 * r["short"] - f * tau_ref * r["dI"] for r in reactive])
        jp = np.array([r["routing"] + 5.0 * r["short"] - f * tau_ref * r["dI"] for r in bundling])
        out.append(100 * (jr - jp).mean() / jr.mean())
    if SAME_NUMPY:
        assert [F.fmt_num(v) for v in out] == ["17,0", "17,0", "17,1"] and abs(tau_ref - BASE["tau"]) < 1e-3
    assert max(out) - min(out) < 0.6 and all(15.5 < v < 18.5 for v in out)


def test_readme_test_section_names_the_bit_identity_evidence():
    ref = FZ.reference()
    assert f"{len(ref['cases'])} eingefrorene Instanzen" in FLAT and "12 Konfigurationen × alle 32 Regeln × alle 12 Kennzahlen" in FLAT and "2.073.600 Werte (27 × 32 × 12 × 200)" in FLAT
    assert "60 Kleininstanzen" in FLAT and "1e-13" in FLAT and "Fehlmenge bei σ = 0 ist exakt 0 und richtig" in FLAT.replace("**", "")
    assert len(ref["policies"]) == 32 and len(ref["metrics"]) == 12 and len({(c["config"]) for c in ref["cases"]}) == 12
