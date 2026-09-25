"""PDF-Export des Tourenplans: Zeichenbereinigung (Latin-1-Kernschrift: Gedankenstrich, Minuszeichen, Euro, Emoji, griechische Buchstaben stürzen fpdf2 sonst ab), Inhalt, Randfälle."""
import pytest

import irp_constants as C
import irp_frozen as FZ
import irp_live as LV
import irp_pdf_export as PDF
import irp_results as R
import irp_ui_panel as UI

DATA = R.load_results()
BASE = R.find_cell(DATA)
SETTINGS = dict(customers=20, capacity=300, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=0)


@pytest.fixture(scope="module")
def live():
    inst = FZ.frozen("base", 0)
    original = LV.make_live_instance
    LV.make_live_instance = lambda *a, **k: inst
    try:
        return LV.solve_live(20, 300, 6, 0.3, 5, 3, 0.5, 2, 0)
    finally:
        LV.make_live_instance = original


def has(data, text):
    """Steht `text` im (unkomprimierten) Inhaltsstrom? Kernschrift = Latin-1, Klammern und Rückstriche sind in PDF-Zeichenketten maskiert."""
    return text.encode("latin-1").replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") in data


def make(live, day, right="P", compress=False, message="Meldung", note="", cell="Basisfall"):
    return PDF.generate_irp_pdf(SETTINGS, live, day, right, message, note, cell, compress=compress)


def test_pdf_text_replaces_the_characters_that_crash_the_core_fonts():
    assert PDF.pdf_text("a – b — c ‑ d − e") == "a - b - c - d - e" and PDF.pdf_text("≥ ≤ → ≈ € ±") == ">= <= -> ca. EUR +-" and PDF.pdf_text("γ σ τ κ μ Δ") == "gamma sigma tau kappa mu Delta"
    assert PDF.pdf_text("„Zitat“ ‘x’") == '"Zitat" \'x\''
    assert PDF.pdf_text("✅Ok") == "Ok" and PDF.pdf_text("⚠️ Achtung") == "(!) Achtung" and PDF.pdf_text("🛢️ Öl") == " Öl" and PDF.pdf_text("Bündeln größer ä ö ü ß") == "Bündeln größer ä ö ü ß"
    assert PDF.pdf_text("🦄") == "?" and PDF.pdf_text("a · b") == "a - b" and PDF.pdf_text("◀ ▶") == "< >"
    assert PDF.pdf_text("x").encode("latin-1") == b"x" and PDF.pdf_text("日本").encode("latin-1") == b"??"


def test_pdf_is_created_and_contains_the_settings_the_rules_and_the_tours(live):
    d = LV.pickup_days(live, 1)[0]
    data = make(live, d)
    assert data[:5] == b"%PDF-" and len(data) > 3000
    for needle in ("Inventory Routing: Wer geh", "Einstellungen", "Kunden", "Mitnahmeschwelle gamma", "Seed", "Kennzahlen der Instanz", "Reaktiv", "Bündeln P(3; 0,5)", "Früher liefern E(2)",
                   "Meldung (aus der vorgerechneten Messreihe)", f"Tourenplan Tag {d}: Reaktiv", f"Tourenplan Tag {d}: Bündeln P(3; 0,5)", "Mitnahme", "Hinweise zum Modell", "Tour 1:"):
        assert has(data, needle), needle


def test_pdf_with_the_early_rule_on_the_right_and_without_a_tour(live):
    d = next(d for d in range(1, 121) if LV.day_view(live, "E", d)["tours"])
    assert has(make(live, d, "E"), f"Tourenplan Tag {d}: Früher liefern E(2)")
    empty = next(d for d in range(1, 121) if not LV.day_view(live, "R", d)["tours"] and not LV.day_view(live, "P", d)["tours"])
    data = make(live, empty)
    assert data.count("An diesem Tag fährt diese Regel keine Tour.".encode("latin-1")) == 2 and b"Tour 1:" not in data


def test_pdf_survives_every_dangerous_character_in_the_texts(live):
    nasty = "Gedankenstrich – Minus −5 Euro € Emoji ✅ ⚠️ 🛢️ griechisch γ σ ≥ ≤ → ± „Zitat“ 日本 Ω"
    for right in ("P", "E"):
        data = PDF.generate_irp_pdf(SETTINGS, live, 5, right, nasty, nasty, nasty)
        assert data[:5] == b"%PDF-"
    assert has(make(live, 5, message=nasty), "Gedankenstrich - Minus -5 Euro EUR")


def test_pdf_carries_the_message_the_note_and_the_cell_text(live):
    data = make(live, 5, message="Bündeln lohnt: 17,0 % Gewinn", note="Hinweis zur Zelle.", cell="20 Kunden, Wagen 300")
    assert has(data, "Bündeln lohnt: 17,0 % Gewinn") and has(data, "Vergleichszelle der Messreihe: 20 Kunden, Wagen 300.") and has(data, "Hinweis zur Zelle.") and data.count(b"Wagen 300") >= 2      # auch im Hinweistext


def test_pdf_lists_all_stops_of_a_busy_day_and_can_be_compressed(live):
    busiest = max(range(1, 121), key=lambda d: LV.day_view(live, "P", d)["n_stops"] + LV.day_view(live, "R", d)["n_stops"])
    plain, packed = make(live, busiest), make(live, busiest, compress=True)
    assert len(packed) < len(plain) and packed[:5] == b"%PDF-"
    view = LV.day_view(live, "P", busiest)
    for tour in view["tours"]:
        assert has(plain, f"Tour {tour['tour']}: {len(tour['stops'])} Stopps, Ladung {tour['load']:.0f} von 300")


def test_pdf_shows_the_selected_rule_parameters_and_kpis(live):
    data = PDF.generate_irp_pdf(dict(SETTINGS, horizon=12, gamma=0.25), dict(live, H=12.0, gamma=0.25), 5, "P", "m", "", "c", compress=False)
    assert has(data, "Bündeln P(12; 0,25)") and has(data, "Vorschau H (Tage)") and has(data, "0,25")
    x = UI.live_numbers(live)
    assert has(make(live, 5), f"{round(x['P']['J']):,}".replace(",", "."))


def test_pdf_has_at_least_one_page_even_without_any_tour(live):
    small = make(live, next(d for d in range(1, 121) if not LV.day_view(live, "R", d)["tours"] and not LV.day_view(live, "P", d)["tours"]))
    assert b"/Type /Page" in small and C.MEASURED_N == 200
