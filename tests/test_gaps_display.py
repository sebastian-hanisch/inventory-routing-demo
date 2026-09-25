"""Tests, die im ersten vollständigen Fehler-Einbau-Lauf überlebende Mutanten der Darstellung (Figuren, PDF) schließen, soweit sie etwas Prüfbares festlegen: Sichtbarkeit der Fehlerbalken,
Legenden, Achsenbereiche, gemeinsame Achsen, Tabellenköpfe und Vorzeichen im PDF."""
import numpy as np
import pytest

import irp_constants as C
import irp_frozen as FZ
import irp_live as LV
import irp_pdf_export as PDF
import irp_results as R
import irp_ui_panel as UI
import irp_visualization as V

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


def test_error_bars_are_visible_in_every_measured_chart():
    cap = V.capacity_figure(R.capacity_rows(DATA, 3, 0.5), 300, 3, 0.5)
    assert cap.data[0].error_y.visible is True
    lvl = V.level_figure(R.uncertainty_rows(DATA, 3, 0.5)["sigma"], "s", 0.3, fmt=str)
    assert all(t.error_y.visible is True for t in lvl.data)
    rules = V.rules_figure(R.rule_rows(BASE, 3, 0.5, 2))
    assert all(t.error_x.visible is True for t in rules.data if t.error_x is not None and t.error_x.array is not None and any(t.error_x.array))
    assert list(rules.data[1].error_x.array) == [R.e_gain(BASE, l)["se"] for l in (1, 2, 3)] and list(rules.data[0].error_x.array)[0] == R.p_gain(BASE, 3, 0.5)["se"] and rules.data[0].error_x.array[1] == 0.0
    reg = V.regime_figure(R.regime_rows(DATA, 3, 0.5), "base")
    assert all(t.error_x.visible is True for t in reg.data)
    orc = V.oracle_figure({150: R.oracle_summary(DATA, 150)})
    assert orc.data[0].error_y.visible is True


def test_shared_axes_ranges_legends_and_scales(live):
    fig = V.trajectory_figure(live, 5)
    assert fig.layout.xaxis.matches == "x2" and fig.layout.height == C.CHART_HEIGHT + 40
    pm = V.plan_matrix_figure([("a", "#123456", [[[(1, 10.0)]]]), ("b", "#123456", [[]])], 7, 6)
    assert pm.layout.xaxis.matches == "x2" and list(pm.layout.xaxis2.range) == [0.4, 6.6] and list(pm.layout.yaxis.range) == [0.4, 7.6] and pm.data[0].showlegend is False
    m = V.day_map_figure(live, "P", LV.pickup_days(live, 1)[0])
    assert all(t.showlegend is False for t in m.data if t.mode == "lines") and len(next(t for t in m.data if t.name == "Kunde (Füllstand als Farbe)").customdata) == 20
    assert V.heatmap_figure(BASE, 3, 0.5).data[0].showscale is False
    assert V.rules_figure(R.rule_rows(BASE, 3, 0.5, 2)).layout.height == C.CHART_HEIGHT - 40 and V.oracle_figure({150: R.oracle_summary(DATA, 150)}).layout.height == C.CHART_HEIGHT - 40
    assert V.level_figure(R.uncertainty_rows(DATA, 3, 0.5)["sigma"], "s", 0.3).layout.height == C.CHART_HEIGHT - 80 and V.capacity_figure(R.capacity_rows(DATA, 3, 0.5), 300, 3, 0.5).layout.height == C.CHART_HEIGHT
    assert V.day_map_figure(live, "R", 3).layout.height == C.CHART_HEIGHT + 60 and UI.CHART_TABLE_HEIGHT == C.CHART_HEIGHT + 60


def test_pdf_table_headers_are_filled_and_the_reference_and_signs_are_printed(live):
    settings = dict(customers=20, capacity=300, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=0)
    day = 5
    data = PDF.generate_irp_pdf(settings, live, day, "P", "m", "", "c", compress=False)
    tours = len(LV.day_view(live, "R", day)["tours"]) + len(LV.day_view(live, "P", day)["tours"])
    assert data.count(b" re B") == 7 * (1 + tours)                                                # jeder Tabellenkopf (Kennzahlen und je Tour) ist gefüllt
    x = UI.live_numbers(live)

    def has(text):
        return text.encode("latin-1").replace(b"(", b"\\(").replace(b")", b"\\)") in data
    assert has("Bezug") and has(f"{x['P']['gain']:+.1f}".replace(".", ",")) and has(f"{x['E']['gain']:+.1f}".replace(".", ","))
    assert len(PDF.generate_irp_pdf(settings, live, day, "P", "m", "", "c")) < len(data)              # Vorgabe: komprimiert
