"""PDF-Export des Tourenplans eines Tages (fpdf2, Helvetica-Kernschrift, nur Text und Tabellen).

Die Kernschriften kennen nur Latin-1: Umlaute sind erlaubt, aber Gedankenstrich (U+2013), Minuszeichen (U+2212), Euro-Zeichen, Emoji, die griechischen
Buchstaben gamma und sigma usw. lassen fpdf2 abstürzen. Deshalb läuft jeder Text durch pdf_text()."""
import time

import irp_constants as C
import irp_format as F
import irp_live as LV
import irp_ui_panel as UI

_REPLACEMENTS = {
    "–": "-", "—": "-", "‑": "-", "−": "-", "≥": ">=", "≤": "<=", "→": "->", "≈": "ca.", "€": "EUR", "±": "+-", "·": "-", "“": '"', "”": '"', "„": '"',
    "‘": "'", "’": "'", "⚠️": "(!)", "⚠": "(!)", "✅": "", "ℹ️": "", "γ": "gamma", "σ": "sigma", "τ": "tau", "κ": "kappa", "μ": "mu", "Δ": "Delta",
    "◀": "<", "▶": ">", "🛢️": "", "🛢": "", "📐": "", "🎯": "", "📊": "", "🔧": "", "🎲": "", "🗺️": "", "🗺": "", "📈": "", "📄": "", "🧮": "",
}


def pdf_text(text):
    """Text für die Helvetica-Kernschrift: bekannte Sonderzeichen ersetzen, den Rest Latin-1-sicher machen."""
    for old, new in _REPLACEMENTS.items():
        text = text.replace(old, new)
    return text.encode("latin-1", "replace").decode("latin-1")


def generate_irp_pdf(settings, live, day, right_rule, message_text, note, cell_text, compress=True):
    """Tourenplan eines Tages als PDF: Einstellungen, Kennzahlen der drei Regeln über 120 Tage, Meldung, Touren des Tages je Regel, Hinweise.

    settings: dict der Reglerwerte (customers, capacity, fleet, sigma, penalty, horizon, gamma, early, seed); live: irp_live.solve_live; day: gezeigter Tag
    (ab 1); right_rule: 'P' oder 'E' (die Regel neben reaktiv); message_text: Text der Meldung; note: Hinweis zur Zell-Zuordnung (leer, wenn exakt);
    cell_text: Beschreibung der Vergleichszelle der Messreihe."""
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    pdf = FPDF()
    pdf.set_compression(compress)
    pdf.add_page()

    def line(text, height=7, width=0):
        pdf.cell(width, height, pdf_text(text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    def heading(text):
        pdf.set_font("Helvetica", "B", 12)
        line(text, 8)
        pdf.set_font("Helvetica", "", 10)

    def pairs(rows):
        for label, value in rows:
            pdf.cell(80, 6, pdf_text(label), border=0)
            line(value, 6)

    def table(headers, widths, rows, size=8):
        pdf.set_font("Helvetica", "B", size)
        pdf.set_fill_color(230, 230, 230)
        for header, width in zip(headers, widths):
            pdf.cell(width, 7, pdf_text(header), border=1, fill=True, new_x=XPos.RIGHT, new_y=YPos.TOP)
        pdf.ln(7)
        pdf.set_font("Helvetica", "", size)
        for row in rows:                                    # der automatische Seitenumbruch von fpdf2 hält die Zeile zusammen
            for value, width in zip(row, widths):
                pdf.cell(width, 7, pdf_text(str(value)), border=1, new_x=XPos.RIGHT, new_y=YPos.TOP)
            pdf.ln(7)

    def keep_together(height):
        if pdf.get_y() + height > pdf.h - pdf.b_margin:
            pdf.add_page()

    H, g, L = live["H"], live["gamma"], live["early"]
    pdf.set_font("Helvetica", "B", 16)
    line("Inventory Routing: Wer gehört heute auf die Tour?", 10)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(120, 120, 120)
    line(f"Erstellt: {time.strftime('%d.%m.%Y %H:%M')}  -  sebastianhanisch.net", 6)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(3)

    heading("Einstellungen")
    pairs([("Kunden", str(settings["customers"])), ("Wagenkapazität", str(settings["capacity"])), ("Touren je Tag (höchstens)", str(settings["fleet"])),
           ("Verbrauchsschwankung", F.fmt_level(settings["sigma"])), ("Fehlmengenstrafe je Einheit", str(settings["penalty"])),
           ("Vorschau H (Tage)", str(settings["horizon"])), ("Mitnahmeschwelle gamma", F.fmt_level(settings["gamma"])),
           ("Früher liefern L (Tage)", str(settings["early"])), ("Seed", str(settings["seed"]))])
    pdf.ln(3)

    heading(f"Kennzahlen der Instanz über {live['days']} Tage (eine einzelne Instanz)")
    x = UI.live_numbers(live)
    rows = []
    for name in ("R", "P", "E"):
        v = x[name]
        rows.append([UI.rule_label(name, H, g, L), F.fmt_cost(v["J"]), F.fmt_num(v["gain"], 1, True) if name != "R" else "Bezug", v["routes"], F.fmt_cost(v["km"]),
                     F.fmt_num(v["util"], 2), F.fmt_num(v["short"], 0)])
    table(["Regel", "Kosten", "Gewinn (%)", "Touren", "Fahr-km", "Auslastung", "Fehlmenge"], [58, 24, 24, 20, 24, 24, 24], rows)
    pdf.ln(3)

    heading("Meldung (aus der vorgerechneten Messreihe)")
    pdf.set_font("Helvetica", "", 9)
    pdf.multi_cell(0, 5, pdf_text(message_text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 4.5, pdf_text(f"Vergleichszelle der Messreihe: {cell_text}. {note}".strip()), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 10)
    pdf.ln(2)

    for rule in dict.fromkeys(("R", right_rule)):
        view = LV.day_view(live, rule, day)
        keep_together(40)
        heading(f"Tourenplan Tag {day}: {UI.rule_label(rule, H, g, L)}")
        if not view["tours"]:
            line("An diesem Tag fährt diese Regel keine Tour.", 6)
            pdf.ln(2)
            continue
        for tour in view["tours"]:
            keep_together(30)
            pdf.set_font("Helvetica", "B", 10)
            line(f"Tour {tour['tour']}: {len(tour['stops'])} Stopps, Ladung {tour['load']:.0f} von {live['Q']:.0f}, Länge {tour['length']:.1f} km", 6)
            pdf.set_font("Helvetica", "", 8)
            table(["Nr.", "Kunde", "Art", "Bestand vorher", "Füllstand", "Reichweite (Tage)", "Liefermenge"], [12, 18, 26, 32, 26, 40, 30],
                  [[i, s["customer"], s["kind"], f"{s['fill_before']:.0f}", F.fmt_share(s["fill_share"]), f"{s['reach']:.1f}".replace(".", ","), f"{s['qty']:.0f}"]
                   for i, s in enumerate(tour["stops"], 1)], size=7)
            pdf.ln(3)

    keep_together(60)
    heading("Hinweise zum Modell")
    pdf.set_font("Helvetica", "", 9)
    for text in [
        "Stark stilisiert: ein Depot, Euklid-Distanz, ein Fahrzeugtyp, keine Zeitfenster oder Fahrerregeln, Lieferung am selben Tag, verlorene Fehlmenge ohne "
        "Rückstau, jeder Besuch füllt den Tank voll. Parameter erfunden, nicht kalibriert: die Gewinne in Prozent gelten für dieses Modell.",
        f"Eine einzelne Instanz streut; die Aussage tragen die vorgerechneten {C.MEASURED_N} Instanzen je Zelle der Messreihe. Zell-Zuordnung: {cell_text}.",
        "Die Regel P(H; gamma) ist eine einfache, gewählte Regel und schließt in der Kleininstanz etwa die Hälfte der Lücke zum Optimum; sie garantiert keine "
        "geringere Fehlmenge.",
    ]:
        pdf.multi_cell(0, 5, pdf_text("- " + text), new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    return bytes(pdf.output())
