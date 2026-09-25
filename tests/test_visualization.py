"""Plotly-Figuren: Inhalt (Spuren, Werte, Hervorhebung der eingestellten Stufe), Konventionen (alle Achsen fixedrange, keine Farblisten, keine Warnungen), Randfälle."""
import warnings

import numpy as np
import pytest

import irp_constants as C
import irp_frozen as FZ
import irp_live as LV
import irp_results as R
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


def assert_locked(fig):
    """Alle Achsen fest (Touch-Scrollen)."""
    axes = [v for k, v in fig.layout.to_plotly_json().items() if k.startswith(("xaxis", "yaxis"))]
    assert axes and all(a.get("fixedrange") is True for a in axes), axes


def no_color_lists(fig):
    """Keine Farblisten in Marker-Eigenschaften: Farben sind Text oder Zahlen mit Farbskala, nie eine Liste von Farbnamen."""
    for tr in fig.data:
        marker = tr.to_plotly_json().get("marker", {})
        colors = marker.get("color")
        if isinstance(colors, (list, tuple, np.ndarray)):
            assert all(isinstance(c, (int, float)) for c in colors), tr.name
            assert marker.get("colorscale") is not None
        line = marker.get("line", {}).get("color")
        assert not isinstance(line, (list, tuple, np.ndarray)), tr.name


def strict(fn, *a, **k):
    """Figur ohne jede Warnung bauen und in ein dict überführen (Plotly warnt bei ungültigen Eigenschaften erst dort)."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fig = fn(*a, **k)
        fig.to_dict()
    return fig


def test_day_map_shows_depot_customers_tours_and_pickups(live):
    d = LV.pickup_days(live, 1)[0]
    view = LV.day_view(live, "P", d)
    fig = strict(V.day_map_figure, live, "P", d)
    names = [t.name for t in fig.data]
    assert names.count("Depot") == 1 and names.count("Kunde (Füllstand als Farbe)") == 1 and names.count("mitgenommen (nicht fällig)") == 1
    cust = next(t for t in fig.data if t.name == "Kunde (Füllstand als Farbe)")
    assert len(cust.x) == 20 == len(cust.y) == len(cust.text) and list(cust.text) == [str(i) for i in range(1, 21)]
    assert np.allclose(cust.marker.color, view["I"][1:] / live["C"][1:]) and cust.marker.cmin == 0.0 and cust.marker.cmax == 1.0
    assert np.allclose(cust.x, live["xy"][1:, 0]) and np.allclose(cust.y, live["xy"][1:, 1]) and cust.marker.colorbar.thickness == 9
    depot = next(t for t in fig.data if t.name == "Depot")
    assert (depot.x[0], depot.y[0]) == (float(live["xy"][0, 0]), float(live["xy"][0, 1]))
    picks = next(t for t in fig.data if t.name == "mitgenommen (nicht fällig)")
    want = [s["customer"] for t in view["tours"] for s in t["stops"] if s["kind"] == "Mitnahme"]
    assert len(picks.x) == len(want) and np.allclose(picks.x, [live["xy"][c, 0] for c in want])
    lines = [t for t in fig.data if t.mode == "lines"]
    assert len(lines) == len(view["tours"]) and all(len(t.x) == len(view["tours"][i]["stops"]) + 2 for i, t in enumerate(lines)) and all(t.x[0] == t.x[-1] for t in lines)
    assert lines[0].line.color == C.TOUR_COLORS[0] and fig.layout.yaxis.scaleanchor == "x" and fig.layout.height == C.CHART_HEIGHT + 60
    assert_locked(fig)
    no_color_lists(fig)


def test_day_map_without_a_tour_and_without_pickups_and_for_other_rules(live):
    empty = next(d for d in range(1, 121) if not LV.day_view(live, "R", d)["tours"])
    fig = strict(V.day_map_figure, live, "R", empty)
    assert [t.name for t in fig.data] == ["Kunde (Füllstand als Farbe)", "Depot"]
    d = next(d for d in range(1, 121) if LV.day_view(live, "R", d)["tours"])
    fig_r = strict(V.day_map_figure, live, "R", d)
    assert "mitgenommen (nicht fällig)" not in [t.name for t in fig_r.data]                              # reaktiv nimmt nie mit
    fig_e = strict(V.day_map_figure, live, "E", d)
    assert "mitgenommen (nicht fällig)" not in [t.name for t in fig_e.data]
    rings = [t for t in fig_r.data if t.mode == "markers" and t.marker.symbol == "circle-open"]
    view = LV.day_view(live, "R", d)
    assert len(rings) == len(view["tours"]) and sum(len(t.x) for t in rings) == view["n_stops"]         # jeder Stopp der reaktiven Regel ist fällig: Ring


def test_day_map_ranges_cover_the_area_and_the_far_depot():
    inst = FZ.frozen("depot_far", 0)
    orig = LV.make_live_instance
    LV.make_live_instance = lambda *a, **k: inst
    try:
        far = LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)
    finally:
        LV.make_live_instance = orig
    fig = strict(V.day_map_figure, far, "P", 5)
    lo = float(far["xy"][:, 1].min())
    assert lo < 0 and fig.layout.yaxis.range[0] <= lo - 3.99 and fig.layout.xaxis.range[1] >= 100 and fig.layout.xaxis.range == fig.layout.yaxis.range


def test_trajectory_shows_cumulative_km_and_the_mean_fill(live):
    fig = strict(V.trajectory_figure, live, 17)
    assert len(fig.data) == 6 and [t.name for t in fig.data][::2] == ["Reaktiv", "Bündeln P(3; 0,5)", "Früher liefern E(2)"]
    for i, name in enumerate(("R", "P", "E")):
        km = fig.data[2 * i]
        assert km.x[0] == 1 and km.x[-1] == 120 and km.y[-1] == pytest.approx(live["rules"][name]["res"]["routing"]) and np.all(np.diff(km.y) >= 0)
        fill = fig.data[2 * i + 1]
        assert 0.0 < min(fill.y) and max(fill.y) <= 1.0 and fill.showlegend is False
    assert fig.data[2].y[-1] < fig.data[0].y[-1] < fig.data[4].y[-1]                                    # Bündeln fährt am wenigsten, früher liefern am meisten
    vlines = [s for s in fig.layout.shapes if s.type == "line"]
    assert vlines and all(s.x0 == 17 for s in vlines)
    assert_locked(fig)


def test_capacity_figure_marks_the_set_wagon_and_shows_both_bars():
    rows = R.capacity_rows(DATA, 3, 0.5)
    fig = strict(V.capacity_figure, rows, 300, 3, 0.5)
    assert [t.name for t in fig.data] == ["Bündeln P(3; 0,5)", "beste Zelle des Gitters (kreuzvalidiert)"] and len(fig.data[0].x) == 7
    assert sum("◀ eingestellt" in x for x in fig.data[0].x) == 1 and "◀ eingestellt" in fig.data[0].x[3] and "Q = 300" in fig.data[0].x[3] and "Q/q = 3,0" in fig.data[0].x[3]
    assert list(fig.data[0].y) == [r["gain"] for r in rows] and list(fig.data[0].error_y.array) == [r["se"] for r in rows] and list(fig.data[1].y) == [r["cv_gain"] for r in rows]
    assert not any("eingestellt" in x for x in V.capacity_figure(rows, 250, 3, 0.5).data[0].x)         # Q = 250 gibt es nicht
    assert_locked(fig)


def test_heatmap_matches_the_grid_and_outlines_the_chosen_combination():
    fig = strict(V.heatmap_figure, BASE, 8, 0.25)
    z = np.array(fig.data[0].z)
    assert z.shape == (4, 7) and np.allclose(z, R.heat_matrix(BASE)) and fig.data[0].zmid == 0
    (rect,) = fig.layout.shapes
    assert (rect.x0, rect.x1, rect.y0, rect.y1) == (4.5, 5.5, 0.5, 1.5)                                  # H = 8 ist Spalte 5, gamma = 0,25 Zeile 1
    rect2 = V.heatmap_figure(BASE, 1, 1.0).layout.shapes[0]
    assert (rect2.x0, rect2.y0) == (-0.5, 2.5) and list(fig.layout.xaxis.ticktext) == [f"H = {h}" for h in C.HORIZON_OPTIONS]
    assert list(fig.layout.yaxis.ticktext) == ["γ = 0,1", "γ = 0,25", "γ = 0,5", "γ = 1,0"] and fig.layout.yaxis.autorange == "reversed"
    assert_locked(fig)


def test_level_figure_highlights_the_set_level_in_its_own_trace():
    rows = R.uncertainty_rows(DATA, 3, 0.5)["sigma"]
    fig = strict(V.level_figure, rows, "Verbrauchsschwankung σ", 0.3, fmt=lambda v: f"{v}")
    assert [t.name for t in fig.data] == ["gemessene Stufe", "eingestellt"] and list(fig.data[1].x) == ["0.3"] and len(fig.data[0].x) == 4
    assert fig.data[1].marker.color == C.COLOR_P and fig.data[0].marker.color == C.COLOR_LIGHT and fig.layout.xaxis.type == "category"
    none = strict(V.level_figure, rows, "x", 0.45, fmt=lambda v: f"{v}")
    assert [t.name for t in none.data] == ["gemessene Stufe"]                                            # keine Stufe eingestellt: keine leere Spur
    assert_locked(fig)


def test_rules_figure_splits_by_sign():
    rows = R.rule_rows(BASE, 3, 0.5, 2)
    fig = strict(V.rules_figure, rows)
    assert [t.name for t in fig.data] == ["billiger als reaktiv", "teurer als reaktiv", "Bezug: reaktiv"]
    assert list(fig.data[1].y) == ["Früher liefern, 1 Tag", "Früher liefern, 2 Tage", "Früher liefern, 3 Tage"] and list(fig.data[0].y) == ["Bündeln P(3; 0,5)", "Beste Zelle, kreuzvalidiert"]
    assert list(fig.data[2].x) == [0.0] and fig.data[0].marker.color == C.COLOR_GOOD and fig.data[1].marker.color == C.COLOR_BAD and fig.layout.yaxis.autorange == "reversed"
    assert_locked(fig)
    only_bad = strict(V.rules_figure, [("a", -1.0, 0.1)])
    assert [t.name for t in only_bad.data] == ["teurer als reaktiv"]


def test_regime_figure_has_one_bar_per_cell_and_highlights_the_chosen():
    rows = R.regime_rows(DATA, 3, 0.5)
    fig = strict(V.regime_figure, rows, "q600")
    assert [t.name for t in fig.data] == ["gemessene Zelle", "Zelle Ihrer Einstellung"] and len(fig.data[0].y) == 26 and list(fig.data[1].y) == ["Wagen 600"]
    assert fig.data[1].x[0] == pytest.approx(R.p_gain(R.find_cell(DATA, Q=600.0), 3, 0.5)["mean"]) and fig.layout.height >= 24 * 27 + 130
    assert [t.name for t in strict(V.regime_figure, rows, "gibt es nicht").data] == ["gemessene Zelle"]
    assert len(fig.layout.yaxis.ticktext) == 27 and all(len(t) <= 30 for t in fig.layout.yaxis.ticktext)
    assert_locked(fig)


def test_short_label_cuts_brackets_and_length():
    assert V.short_label("Basisfall (20 Kunden)") == "Basisfall" and V.short_label("x" * 40) == "x" * 29 + "…" and V.short_label("x" * 30) == "x" * 30 and V.short_label("kurz", 2) == "k…"


def test_oracle_figure_groups_by_wagon():
    sums = {150: R.oracle_summary(DATA, 150), 250: R.oracle_summary(DATA, 250)}
    fig = strict(V.oracle_figure, sums)
    assert [t.name for t in fig.data] == ["Wagen 150", "Wagen 250"] and list(fig.data[0].x) == ["Reaktiv", "Bündeln P(3; 0,5)", "Bündeln P(8; 0,25)", "Früher liefern E(1)"]
    assert fig.data[0].y[0] == pytest.approx(sums[150]["gap"]["R"]["mean"]) and list(fig.data[1].error_y.array)[3] == pytest.approx(sums[250]["gap"]["E_L1"]["se"])
    assert fig.data[0].marker.color == C.COLOR_P and fig.data[1].marker.color == C.COLOR_LIGHT
    assert_locked(fig)


def test_plan_matrix_has_one_panel_per_plan_and_sizes_dots_by_quantity():
    plans = [[[(1, 60.0), (3, 20.0)]], [], [[(2, 200.0)]]]
    fig = strict(V.plan_matrix_figure, [("Optimum", C.COLOR_GOOD, plans), ("Reaktiv", C.COLOR_R, [[], [], []])], 3, 3)
    assert len(fig.data) == 2 and list(fig.data[0].x) == [1, 1, 3] and list(fig.data[0].y) == [1, 3, 2] and len(fig.data[1].x) == 0
    sizes = list(fig.data[0].marker.size)
    assert sizes[0] > sizes[1] and sizes[2] == 24.0 and min(sizes) >= 7.0                                 # Größe wächst mit der Menge, mit Ober- und Untergrenze
    assert [a.text for a in fig.layout.annotations] == ["Optimum", "Reaktiv"] and fig.layout.height == 150 * 2 + 90
    assert_locked(fig)
