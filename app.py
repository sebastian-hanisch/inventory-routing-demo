"""
Inventory Routing: Wer gehört heute auf die Tour? - interaktive Fall-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Ein Depot beliefert Kunden mit Tanks (Heizöl, Gase, Getränke, Ersatzteile): Jeder Kunde verbraucht jeden Tag, und jemand entscheidet, welche Kunden heute
auf die Tour gehören. Live: EINE Instanz über 120 Tage, alle drei Regeln (reaktiv R, Bündeln P(H, gamma), früher liefern E(L)) laufen über denselben
Verbrauchsstrom. Vorgerechnet: die Messreihe über 200 Instanzen je Zelle (data/irp_results.json), die die Aussage trägt. Nur der Exakt-Tab hat einen Knopf
(CP-SAT auf der Kleininstanz).

Lauffähig mit: streamlit run app.py
"""
import time

import streamlit as st

import irp_constants as C
import irp_format as F
import irp_live as LV
import irp_oracle as OR
import irp_results as R
import irp_ui_panel as UI
import irp_visualization as V
from irp_pdf_export import generate_irp_pdf
from irp_presets import (SETTING_SPECS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed,
                         sync_query_params)

st.set_page_config(page_title="Inventory Routing – Sebastian Hanisch", layout="wide")

SCENARIO_KEYS = list(SETTING_SPECS)
DATA = R.load_results()
META = DATA["_meta"]
# Zahlen der Texte: alle aus der vorgerechneten Messreihe (kein Text-Zahlen-Widerspruch möglich)
_BASE = R.find_cell(DATA)
_Q100, _Q150, _Q600 = R.find_cell(DATA, Q=100.0), R.find_cell(DATA, Q=150.0), R.find_cell(DATA, Q=600.0)
_SIGMA0 = R.find_cell(DATA, sigma=0.0)
_F1 = R.find_cell(DATA, F=1)
_DEF = lambda cell: R.p_gain(cell, C.DEFAULT_H, C.DEFAULT_GAMMA)
_NEG = R.negative_summary(DATA)
_MECH = R.mechanism(_BASE)
_DIST = META["base_distribution"]
_SUM150, _SUM250 = R.oracle_summary(DATA, 150), R.oracle_summary(DATA, 250)
_RHO = META["spearman"]
_E_BASE = [R.e_gain(_BASE, L) for L in C.EARLY_OPTIONS]
ORACLE_TIME_LIMIT = 20.0            # Sekunden je Aufruf des exakten Lösers (die Kleininstanz braucht 0 bis 2 s)
ORACLE_COOLDOWN = 4.0               # Sekunden zwischen zwei Aufrufen (Cloud-Rechner schonen)
GAP_HELP = "Lücke zum Optimum: Kosten der Regel minus Kosten des Optimums, in % der Kosten des Optimums (positiv = teurer als das Optimum). Das Optimum erlaubt keine Fehlmenge, die Regeln sind mit Strafe bewertet."


@st.cache_data(show_spinner=False, max_entries=C.CACHE_ENTRIES)
def _live(customers, capacity, fleet, sigma, penalty, H, gamma, early, seed):
    """Live-Instanz mit allen drei Regeln (Zehntelsekunden), je Einstellung zwischengespeichert. Der Aufruf geht über das Modul, damit die Tests die Rechnung
    zählen oder ersetzen können."""
    return LV.solve_live(customers, capacity, fleet, sigma, penalty, H, gamma, early, seed)


st.title("🛢️ Inventory Routing: Wer gehört heute auf die Tour?")
st.markdown(
    """
Kunden mit Tanks verbrauchen jeden Tag, und jemand entscheidet, **wer heute auf die Tour gehört**: nur wer melden würde (**reaktiv**), oder auch, wer bald melden wird und
fast auf dem Weg liegt (**bündeln**)? Die Demo zeigt live auf **einer Instanz über 120 Tage**, was das spart, und vorgerechnet über **200 Instanzen je Zelle**, wovon es abhängt
– vor allem von der **Wagenkapazität** im Verhältnis zur Liefermenge, kaum von der Unsicherheit – und wann Bündeln nichts bringt oder sogar teurer ist. Das exakte Optimum einer
Kleininstanz (7 Kunden, 6 Tage) dient als Maßstab. In der Fachsprache ein **Inventory Routing Problem**: Tourenplanung mit **Bestand beim Kunden** über mehrere Tage. Die Tourenbausteine
(Savings, 2-opt) kennt man aus `vrp_demo`, `alns-demo` und `vrp-nachbarschaften-demo`; die Frage „wie weit vorausschauen“ stellt schon `leercontainer-demo`, hier gekoppelt an eine Tour;
im Tourenplanungs-Zweig sind `nahverkehr-demo` (ein Tag mit Ereignissen) und `fernverkehr-demo` (Ressourcen entlang der Route) die Schwestern. Wie das Modell funktioniert, steht im Expander
„Wie funktioniert diese Demo?“ weiter unten, die formale Beschreibung im Expander „📐 Mathematische Formulierung“.
"""
)

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(C.PRESETS)
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(3)
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP[name])

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    st.markdown("**Der Betrieb**")
    customers = st.select_slider("Kunden", options=list(C.CUSTOMER_OPTIONS), key="customers_slider",
                                 help="Zahl der Kunden im Gebiet 100 × 100 (Dichte). Die Stufen sind die gemessenen Zellen der Messreihe (10 / 20 / 40).")
    capacity = st.select_slider("Wagenkapazität", options=list(C.CAPACITY_OPTIONS), key="capacity_slider",
                                help="Der wichtigste Hebel: wie viel ein Wagen fasst, im Verhältnis zur Liefermenge je Besuch (etwa 100 Einheiten). Passt nur eine Lieferung in den Wagen "
                                     "(Q = 100), gibt es nichts zu bündeln.")
    _qcell = R.find_cell(DATA, Q=float(capacity))
    st.caption(f"Q/q ≈ {F.fmt_num(_qcell['Q_over_qL'])} Lieferungen je Wagen (mittlere Liefermenge q ≈ {F.fmt_num(_qcell['qL'], 0)} bei reaktivem Nachliefern, Messreihe)")
    fleet = st.select_slider("Touren je Tag", options=list(C.FLEET_OPTIONS), key="fleet_slider",
                             help="Wie viele Touren (Wagen) je Tag höchstens fahren können. Bei 1 oder 2 wird die Flotte knapp: reichen die Touren nicht, wird der am wenigsten "
                                  "dringende Kunde zurückgestellt und es entstehen Fehlmengen.")
    sigma = st.select_slider("Verbrauchsschwankung", options=list(C.SIGMA_OPTIONS), key="sigma_slider", format_func=F.fmt_level,
                             help="Variationskoeffizient des Tagesverbrauchs (Gamma-verteilt). 0 = deterministisch: zeigt, dass der Gewinn kein Sicherheitseffekt ist.")
    penalty = st.select_slider("Fehlmengenstrafe", options=list(C.PENALTY_OPTIONS), key="penalty_slider", format_func=lambda v: f"{v} je Einheit",
                               help="Strafe je verlorener Einheit Fehlmenge, in Fahrstrecken-Einheiten (Fehlmenge geht verloren, kein Rückstau).")
    st.markdown("**Die Regel**")
    horizon = st.select_slider("Vorschau H", options=list(C.HORIZON_OPTIONS), key="horizon_slider", format_func=lambda v: f"{v} Tage",
                               help="Bündeln: bis zu welcher Restreichweite (in Tagen über der Meldegrenze) Kunden auf der Tour mitgenommen werden dürfen.")
    gamma = st.select_slider("Mitnahmeschwelle γ", options=list(C.GAMMA_OPTIONS), key="gamma_slider", format_func=F.fmt_level,
                             help="Bündeln: größte erlaubte Mehrstrecke als Anteil der Hin- und Rückfahrt Depot–Kunde. Bei γ = 1 kostet die Mitnahme höchstens eine eigene Fahrt.")
    early = st.select_slider("Früher liefern (Tage)", options=list(C.EARLY_OPTIONS), key="early_slider", format_func=lambda v: f"{v} Tage",
                             help="Gegenprobe E(L): die Meldegrenze wird um L Tage angehoben, ohne auf die Tour zu schauen. Wirkt immer: es werden alle drei Regeln gerechnet.")
    st.markdown("**Die gezeigte Instanz**")
    seed = st.number_input("Seed", *bounds("seed_input"), key="seed_input", step=1,
                           help="Nummer der gezeigten Instanz. Die Kennzahlen der Messreihe stehen auf den Seeds 0 bis 199; die Beispielszenarien zeigen Instanzen außerhalb (200 bis 299).")
    st.button("🎲 Neue Instanz", width="stretch", on_click=randomize_seed, help="Würfelt einen neuen Seed.")

sync_query_params({key: st.session_state[key] for key in SCENARIO_KEYS})
customers, capacity, fleet, penalty, horizon, early, seed = (int(x) for x in (customers, capacity, fleet, penalty, horizon, early, seed))
sigma, gamma = float(sigma), float(gamma)

cell, exact, diff = R.nearest_cell(DATA, customers, capacity, fleet, sigma, penalty)
note = R.assignment_note(customers, capacity, fleet, sigma, penalty, cell, diff)

live = _live(customers, capacity, fleet, sigma, penalty, horizon, gamma, early, seed)

# ---------------------------------------------------------------------------------------------------
# Hauptansicht (Kernabschnitt ①, live: eine Instanz)
# ---------------------------------------------------------------------------------------------------
st.markdown("## 🛢️ Wie viel spart es, mehr Kunden auf eine Tour mitzunehmen?")
st.caption(f"Live-Instanz (eine Instanz, {live['days']} Tage): Seed {live['seed']}, {live['N']} Kunden, Wagen {capacity}, höchstens {fleet} Tour{'en' if fleet > 1 else ''} je Tag, Verbrauchsschwankung "
           f"{F.fmt_level(sigma)}, Strafe {penalty} je Einheit. Alle drei Regeln laufen bei jeder Einstellung neu (jede in der Regel unter 0,05 s, je Einstellung zwischengespeichert). "
           "Eine einzelne Instanz – die vorgerechnete Messreihe unten trägt die Aussage.")

metric_rows = [st.columns(2), st.columns(2)]
UI.render_metrics(metric_rows[0] + metric_rows[1], live)
st.caption(UI.shortage_caption(live))
msg_state = UI.render_message(live, cell, note)
_, msg_text = UI.message(live, cell, note)

st.markdown("**Diese Instanz und die Messreihe nebeneinander**")
st.dataframe(UI.comparison_table(live, cell), width="stretch", hide_index=True)
st.caption(UI.distribution_sentence(cell, horizon, gamma) + (" " + note if note else ""))

st.markdown("#### 🗺️ Tagesansicht: Wer fährt heute, und wer wird mitgenommen?")
_pick_days = LV.pickup_days(live, 1)
if "day_slider" not in st.session_state:
    st.session_state["day_slider"] = _pick_days[0] if _pick_days else 1
day = st.slider("Tag", 1, live["days"], key="day_slider",
                help="Reine Anzeigewahl im Ergebnisbereich (kein Regler der Rechnung): alle 120 Tage sind schon gerechnet, hier wählen Sie nur, welchen Tag Sie sehen.")
RIGHT_OPTIONS = {"Bündeln neben Reaktiv": "P", "Früher liefern neben Reaktiv": "E"}
right_label = st.radio("Anzeige", list(RIGHT_OPTIONS), horizontal=True, key="view_select",
                       help="Reine Anzeigewahl im Ergebnisbereich: die Karte rechts zeigt Bündeln oder die Gegenprobe „früher liefern“. Alle drei Regeln sind schon gerechnet; die Kennzahlen oben ändern sich nicht.")
right_rule = RIGHT_OPTIONS[right_label]
_days_text = ", ".join(str(d) for d in LV.pickup_days(live, 8))
st.caption("Kunden: Farbe = Füllstand am Tagesanfang (rot leer, grün voll), Ring = fällig (unter der Meldegrenze), Raute = mitgenommen, obwohl noch nicht fällig; Linienfarbe = Tour. "
           + (f"Tage, an denen Bündeln Kunden mitnimmt (die ersten): {_days_text}." if _days_text else "Bei dieser Einstellung nimmt Bündeln keinen Kunden mit."))
views = UI.render_day("main", live, day, right_rule)
st.plotly_chart(V.trajectory_figure(live, day), width="stretch", key="main_trajectory")
_fills = {r: LV.mean_fill(live, r) for r in ("R", "P", "E")}
st.caption("Oben die kumulierten Fahr-km, unten der mittlere Füllstand der Kunden am Tagesanfang (Mittel über die 120 Tage: reaktiv "
           f"{F.fmt_share(_fills['R'])}, Bündeln {F.fmt_share(_fills['P'])}, früher liefern {F.fmt_share(_fills['E'])}). An einzelnen Tagen kann Bündeln mehr fahren als reaktiv (es nimmt Kunden schon "
           "heute mit, die sonst später eine eigene Tour brauchten), über die Zeit fährt es im Basisfall weniger Kilometer; früher liefern hält dort die Tanks am vollsten und fährt am meisten. "
           "Die gestrichelte Linie ist der gezeigte Tag.")

pdf_slot = st.container()

st.markdown("---")

# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ② (vorgerechnet): was die Messreihe zeigt
# ---------------------------------------------------------------------------------------------------
st.markdown("### 📐 Was die Messreihe über 200 Instanzen zeigt")
st.markdown(
    f"""
Kernfrage: Wie viel spart Bündeln, und wovon hängt es ab? Die Antwort steht auf **{C.MEASURED_N} gepaarten Instanzen je Zelle** (alle Regeln auf denselben Instanzen und Verbrauchsströmen, 120 Tage),
vorgerechnet und **nie live** gerechnet: der Sweep braucht etwa 126 s auf 12 Prozessen. Gezeigt wird die gemessene Zelle, die Ihrer Einstellung am nächsten liegt:
**{R.cell_setting_text(cell)}**. Gewinn in % der reaktiven Kosten, Mittel ± Standardfehler; ein Vorzeichen gilt nur ab 2 Standardfehlern. Die Regel ist die eingestellte
**P({horizon}; {F.fmt_level(gamma)})**.
"""
)
if note:
    st.info(note)

st.markdown("**1 · Kapazitätshebel** – wie viel spart Bündeln, je nachdem, wie viele Lieferungen in einen Wagen passen?")
cap_rows = R.capacity_rows(DATA, horizon, gamma)
st.plotly_chart(V.capacity_figure(cap_rows, capacity, horizon, gamma), width="stretch", key="core_capacity")
st.dataframe(UI.capacity_frame(cap_rows, capacity), width="stretch", hide_index=True)
st.caption(f"Gewinn über die Wagenkapazität Q (übrige Parameter Basis, {C.MEASURED_N} Instanzen je Stufe): passt nur eine Lieferung in den Wagen (Q/q ≈ {F.fmt_num(_Q100['Q_over_qL'])}), gibt es nichts zu bündeln "
           f"({F.fmt_num(_DEF(_Q100)['mean'], 1, True)} % für die Standardregel), mit Platz für drei Lieferungen sind es {F.fmt_num(_DEF(_BASE)['mean'], 1, True)} %, mit Platz für sechs "
           f"{F.fmt_num(_DEF(_Q600)['mean'], 1, True)} %. Rangkorrelation zwischen Gewinn und Q/q über {_RHO['n']} Konfigurationen: {F.fmt_num(_RHO['rho'], 2)} – Q/q ist ein starker, aber nicht "
           "der alleinige Vorhersager (Geometrie und Dichte wirken zusätzlich). Hell: die beste Zelle des Gitters, kreuzvalidiert gewählt.")

st.markdown("**2 · Vorschau × Mitnahmeschwelle** – wie weit vorausschauen, und wie großzügig mitnehmen?")
h1, h2 = st.columns([3, 2])
gs = R.grid_summary(cell)
with h1:
    st.plotly_chart(V.heatmap_figure(cell, horizon, gamma), width="stretch", key="core_heat")
with h2:
    st.markdown(f"In dieser Zelle sind **{gs['positive']} von {gs['n']}** Kombinationen im Mittel billiger als reaktiv, **{gs['negative']}** teurer. Die beste liegt bei "
                f"H = {R.parse_p_key(gs['best_key'])[0]}, γ = {F.fmt_level(R.parse_p_key(gs['best_key'])[1])} ({F.fmt_num(gs['best'], 1, True)} %), **{gs['near_best']}** Kombinationen liegen höchstens "
                "3 Prozentpunkte darunter. Ihre Einstellung ist umrandet.")
    st.caption("Mit kleinem γ (nur fast kostenlose Mitnahmen) wächst der Gewinn bis H ≈ 6–8 Tage und bleibt dann flach; mit γ = 1 (Mitnahme bis zum Preis einer eigenen Hin- und Rückfahrt) kippt er "
               "bei weiter Vorschau ins Negative. Vorschau allein, ohne γ-Schwelle, ist keine Lösung.")

st.markdown("**3 · Unsicherheit und Strafe** – hilft Bündeln gegen Störungen?")
unc = R.uncertainty_rows(DATA, horizon, gamma)
u1, u2 = st.columns(2)
with u1:
    st.plotly_chart(V.level_figure(unc["sigma"], "Verbrauchsschwankung σ", sigma, fmt=F.fmt_level), width="stretch", key="core_sigma")
    st.dataframe(UI.uncertainty_frame(unc["sigma"], "Verbrauchsschwankung", fmt=F.fmt_level), width="stretch", hide_index=True)
with u2:
    st.plotly_chart(V.level_figure(unc["penalty"], "Fehlmengenstrafe je Einheit", float(penalty), fmt=lambda v: f"{v:g}"), width="stretch", key="core_penalty")
    st.dataframe(UI.uncertainty_frame(unc["penalty"], "Fehlmengenstrafe", fmt=lambda v: f"{v:g}"), width="stretch", hide_index=True)
st.caption(f"Der Gewinn ist fast flach über σ und Strafe: ohne jede Zufallsschwankung (σ = 0) bleibt der Effekt der Standardregel vollständig ({F.fmt_num(_DEF(_SIGMA0)['mean'], 1, True)} %). Bündeln ist damit "
           "ein Auslastungs-, kein Sicherheitseffekt („proaktiv hilft gegen Störungen“ ist hier nicht das Argument); nur bei knapper Flotte kommt ein Sicherheitseffekt dazu (Abschnitt 5).")

st.markdown("**4 · Regeln im Vergleich** – früher liefern gegen bündeln")
r1, r2 = st.columns(2)
with r1:
    st.plotly_chart(V.rules_figure(R.rule_rows(cell, horizon, gamma, early)), width="stretch", key="core_rules")
    st.dataframe(UI.rule_rows_frame(R.rule_rows(cell, horizon, gamma, early)), width="stretch", hide_index=True)
with r2:
    st.markdown("**Mechanismus der Standardregel P(3; 0,5)** (Mittel je Instanz und 120 Tage, gewählte Zelle):")
    st.dataframe(UI.mechanism_frame(cell), width="stretch", hide_index=True)
st.caption(f"Einfach früher zu liefern ist keine Alternative: im Basisfall {F.fmt_num(_E_BASE[0]['mean'], 1, True)} / {F.fmt_num(_E_BASE[1]['mean'], 1, True)} / {F.fmt_num(_E_BASE[2]['mean'], 1, True)} % für 1 / 2 / 3 Tage, "
           f"in {META['E_negative_pairs'][0]} von {META['E_negative_pairs'][1]} Regel-Zellen-Paaren teurer als reaktiv (positiv nur bei knapper Flotte oder sehr hoher Strafe). Grund: mehr Besuche ohne Tourenbezug "
           f"({F.fmt_num(_MECH['visits'][0])} → {F.fmt_num(_MECH['visits'][2])} Besuche, {F.fmt_num(_MECH['routes'][0])} → {F.fmt_num(_MECH['routes'][2])} Touren bei E(2); dafür keine Fehlmenge: Sicherheit wird mit Fahrkosten erkauft), "
           f"während Bündeln Touren einspart ({F.fmt_num(_MECH['routes'][0])} → {F.fmt_num(_MECH['routes'][1])}) und die Wagen füllt "
           f"(Auslastung {F.fmt_num(_MECH['util'][0], 2)} → {F.fmt_num(_MECH['util'][1], 2)}). Der Gewinn kommt aus weniger, volleren Touren, nicht aus weniger Besuchen: die Besuche steigen sogar "
           f"({F.fmt_num(_MECH['visits'][0])} → {F.fmt_num(_MECH['visits'][1])} in 120 Tagen).")

st.markdown("**5 · Knappe Flotte** – hier verhindert Vorziehen vor allem Fehlmengen")
st.dataframe(UI.shortage_frame(R.shortage_rows(DATA)), width="stretch", hide_index=True)
st.caption(f"Bei knapper Flotte ist ein großer Teil des Gewinns Vermeidung von Fehlmengen. **Keine Garantie:** in der Basis sinkt die Fehlmenge im Mittel ({F.fmt_num(_BASE['short_R'])} → {F.fmt_num(_BASE['short_P'])} Einheiten), "
           f"aber in {F.fmt_num(100 * _BASE['short_cmp']['more'], 1)} % der Instanzen ist sie beim Bündeln höher: Bündeln verändert die Bestandslage, spätere Engpässe sind möglich. In der Basis machen die Fehlmengen nur einen kleinen "
           "Teil der Kosten aus, sie erklären den Gewinn dort nicht.")

st.markdown("**6 · Regime** – wo lohnt Bündeln, wo nicht? (alle 27 gemessenen Zellen)")
regime = R.regime_rows(DATA, horizon, gamma)
st.plotly_chart(V.regime_figure(regime, cell["name"]), width="stretch", key="core_regime")
st.dataframe(UI.regime_frame(regime, cell["name"]), width="stretch", hide_index=True)
st.caption(f"Urteil in drei Zuständen, wie die Meldung oben: „Bündeln lohnt“ (Gewinn der gewählten Regel mindestens 5 % und über 2 Standardfehler), „Bündeln bringt hier wenig“ (dazwischen) und „teurer als reaktiv“ (unter −2 Standardfehler). Über alle "
           f"{len(DATA['cells'])} Zellen und 28 Gitterkombinationen sind **{_NEG[0]} von {_NEG[1]}** proaktiven Kombinationen im Mittel **teurer als reaktiv**, davon {_NEG[2]} mit γ = 1,0 und keine mit γ ≤ 0,25. "
           "Regime ohne Nutzen: (i) fehlende Kapazität (Q/q ≈ 1) und (ii) eine zu großzügige Mitnahmeregel bei viel Vorschau und kleinem Wagen.")

st.markdown("**7 · Exakter Maßstab** – wie weit ist die Regel vom Optimum? (vorgerechnet, Kleininstanz)")
o1, o2 = st.columns([2, 3])
with o1:
    st.plotly_chart(V.oracle_figure({150: _SUM150, 250: _SUM250}), width="stretch", key="core_oracle")
with o2:
    st.dataframe(UI.oracle_frame({150: _SUM150, 250: _SUM250}), width="stretch", hide_index=True)
st.caption(f"7 Kunden, 6 Tage, eine Tour je Tag, deterministischer Verbrauch, {_SUM150['n']} Instanzen je Wagen, alle bewiesen optimal (0 bis 2 s je Instanz). Die beste einfache Regel schließt etwa die **Hälfte** des "
           f"möglichen Vorteils ({F.fmt_share(_SUM150['closed_P3'])} bzw. {F.fmt_share(_SUM250['closed_P3'])} der reaktiven Lücke); die andere Hälfte liegt im Wissen über die kommenden Tage. "
           "**Für die stochastische Basis (20 Kunden, 120 Tage) gibt es kein Optimum**, es ist nicht gerechnet; die Kleininstanz gibt nur die Größenordnung.")

# PDF (nach der Live-Rechnung, damit der Tourenplan des gezeigten Tages drinsteht)
with pdf_slot:
    st.download_button(
        "📄 Tourenplan als PDF herunterladen",
        data=generate_irp_pdf(dict(customers=customers, capacity=capacity, fleet=fleet, sigma=sigma, penalty=penalty, horizon=horizon, gamma=gamma, early=early, seed=seed),
                              live, day, right_rule, msg_text, note, R.cell_setting_text(cell)),
        file_name="inventory_routing_tourenplan.pdf", mime="application/pdf", key="primary_pdf_download",
        help="Einstellungen, Kennzahlen der drei Regeln, Meldung und die Touren des gezeigten Tages (Halteliste mit Menge, Bestand und Reichweite).")

st.markdown("---")

# ---------------------------------------------------------------------------------------------------
# Ansichten
# ---------------------------------------------------------------------------------------------------
with st.expander("🔧 Wie wir das erreichen – Regeln im Vergleich"):
    tabs = st.tabs(["🗺️ Tagesansicht", "📊 Regeln", "🎯 Exaktes Optimum", "📈 Messreihe"])
    with tabs[0]:
        st.markdown(
            "Die **Tagesansicht** ist eine Simulation von 120 Tagen: Jeden Morgen kennt die Regel den Bestand jedes Kunden und die Raten; sie wählt die Kunden der Tour(en), die Touren entstehen "
            "per **Savings** (Clarke-Wright) mit **2-opt**, geliefert wird am selben Morgen bis zur Obergrenze des Tanks, dann verbraucht jeder Kunde (Gamma-verteilt, für alle Regeln derselbe Strom). "
            "**Reaktiv** nimmt nur Fällige, **Bündeln** dazu Kunden mit kleiner Restreichweite auf den vorhandenen Touren, **früher liefern** hebt nur die Meldegrenze an. Unten die Tabelle aller Tage."
        )
        st.dataframe(UI.day_frame(live), width="stretch", hide_index=True, height=UI.CHART_TABLE_HEIGHT)
    with tabs[1]:
        st.markdown(
            "Alle drei Regeln auf **dieser** Instanz. **R** liefert nur an Kunden unter der Meldegrenze, **P(H; γ)** nimmt zusätzlich Kunden mit Restreichweite bis 1 + κσ + H Tage auf den vorhandenen Touren mit, "
            "wenn die billigste Einfügung höchstens γ · Hin- und Rückfahrt kostet und der Wagen es trägt, **E(L)** hebt die Meldegrenze um L Tage an, ohne auf die Tour zu schauen. "
            "Rechts der Gewinn der Messreihe (Mittel ± Standardfehler, 200 Instanzen, nächstliegende Zelle)."
        )
        st.dataframe(UI.rule_table(live, cell), width="stretch", hide_index=True)
        st.caption("Eine einzelne Instanz: Unterschiede zwischen den Regeln streuen; die Messreihe trägt die Aussage. Bündeln fährt weniger Touren mit höherer Auslastung; früher liefern fährt mehr Besuche ohne Tourenbezug.")
    with tabs[2]:
        st.markdown(
            "Das **exakte Optimum** kennt alle kommenden Tage und plant mehrtägig (CP-SAT von OR-Tools: je Tag ein Kreis durch das Depot mit optionalen Kunden, Mengen, Bestand ≥ 0). Es gibt es nur für die "
            "**Kleininstanz**: 7 Kunden, 6 Tage, eine Tour je Tag, Tanks 3 bis 6 Tage, deterministischer Verbrauch, ganzzahlige Mengen. Der Vergleich mit den Regeln zeigt, wie weit sie vom Optimum entfernt sind."
        )
        st.info("Für die stochastische Basis (20 Kunden, 120 Tage, Verbrauchsschwankung) gibt es **kein Optimum**: das Modell ist zu groß für den exakten Löser und der künftige Verbrauch ist nicht bekannt. "
                "Die Zahlen des Abschnitts „7 · Exakter Maßstab“ und dieser Kleininstanz sind nur die Größenordnung der Lücke.")
        wagon = st.radio("Wagen der Kleininstanz", [150, 250], horizontal=True, key="oracle_wagon", help="Wagenkapazität der Kleininstanz (die Messreihe hat je 30 Instanzen für beide Wagen).")
        oseed = st.number_input("Seed der Kleininstanz", 0, 299, key="oracle_seed", step=1, help="Nummer der Kleininstanz (die vorgerechneten 30 Instanzen je Wagen sind die Seeds 0 bis 29).")
        st.caption(f"Zeitlimit {F.fmt_num(ORACLE_TIME_LIMIT, 0)} s, Pause {F.fmt_num(ORACLE_COOLDOWN, 0)} s zwischen zwei Aufrufen; die Instanzen sind auf {OR.MAX_CUSTOMERS} Kunden und {OR.MAX_DAYS} Tage begrenzt. "
                   f"Verglichen wird mit den Regeln, wie sie oben eingestellt sind: {UI.rule_label('P', horizon, gamma, early)} und {UI.rule_label('E', horizon, gamma, early)}.")
        okey = (int(wagon), int(oseed), horizon, gamma, early)
        if st.button("🧮 Optimum berechnen (CP-SAT)", key="oracle_button", help="Löst die Kleininstanz exakt mit CP-SAT (0 bis 2 s) und vergleicht das Optimum mit den Regeln."):
            left = OR.cooldown_left(st.session_state.get("oracle_last"), time.monotonic(), ORACLE_COOLDOWN)
            if left > 0:
                st.warning(f"Bitte noch {F.fmt_num(left, 0)} s warten: zwischen zwei Aufrufen liegt eine Pause von {F.fmt_num(ORACLE_COOLDOWN, 0)} s.")
            else:
                st.session_state["oracle_last"] = time.monotonic()
                try:
                    with st.spinner("Rechne das Optimum der Kleininstanz (CP-SAT) …"):
                        result = OR.solve_small(int(wagon), int(oseed), horizon, gamma, early, time_limit=ORACLE_TIME_LIMIT)
                    st.session_state["oracle_result"] = {"key": okey, "result": result}
                except ImportError:
                    st.error("OR-Tools ist in dieser Umgebung nicht installiert: der exakte Löser steht hier nicht zur Verfügung.")
        stored = st.session_state.get("oracle_result")
        if stored is not None and stored["key"] != okey:
            st.info("Die Einstellungen haben sich seit dem letzten Aufruf geändert: bitte den Knopf erneut drücken, um das Optimum dazu zu berechnen.")
        elif stored is not None and stored["result"] is None:
            st.warning("Diese Kleininstanz ist unzulässig (der Tagesbedarf ist mit einer Tour je Tag nicht deckbar): es gibt keinen Plan ohne Fehlmenge. Anderen Seed wählen.")
        elif stored is not None:
            res = stored["result"]
            cmp_ = res["compare"]
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Optimum (CP-SAT)", F.fmt_num(cmp_["optimum"]["J"]), delta=f"{res['status']}, {F.fmt_num(res['time'], 1)} s", delta_color="off", delta_arrow="off")
            m2.metric("Reaktiv", F.fmt_num(cmp_["R"]["J"]), delta=f"{F.fmt_num(cmp_['R']['gap'], 1, True)} % Lücke", delta_color="inverse", help=GAP_HELP)
            m3.metric(UI.rule_label("P", horizon, gamma, early), F.fmt_num(cmp_["P"]["J"]), delta=f"{F.fmt_num(cmp_['P']['gap'], 1, True)} % Lücke", delta_color="inverse", help=GAP_HELP)
            m4.metric(UI.rule_label("E", horizon, gamma, early), F.fmt_num(cmp_["E"]["J"]), delta=f"{F.fmt_num(cmp_['E']['gap'], 1, True)} % Lücke", delta_color="inverse", help=GAP_HELP)
            if res["status"] != "OPTIMAL":
                st.warning(f"Das Zeitlimit von {F.fmt_num(ORACLE_TIME_LIMIT, 0)} s wurde erreicht: der Status ist {res['status']}, das Ergebnis ist die beste gefundene, **nicht bewiesen optimale** Lösung "
                           "(auf einem Rechner mit wenigen Kernen braucht CP-SAT länger; mit 8 Arbeitern sind es sonst 0,3 bis 1,2 s). Die Lücken sind dann nur eine obere Grenze.")
            st.dataframe(UI.compare_small_frame(cmp_), width="stretch", hide_index=True)
            st.plotly_chart(V.plan_matrix_figure([("Optimum (CP-SAT)", C.COLOR_GOOD, cmp_["optimum"]["plans"]), ("Reaktiv", C.COLOR_R, cmp_["R"]["plans"]),
                                                  (UI.rule_label("P", horizon, gamma, early), C.COLOR_P, cmp_["P"]["plans"])], OR.SMALL_CUSTOMERS, OR.SMALL_DAYS),
                            width="stretch", key="oracle_plan")
            st.markdown("**Plan des Optimums** (je Tag die Tour: Kunde und Liefermenge):")
            st.dataframe(UI.small_plan_frame(cmp_["optimum"]["plans"]), width="stretch", hide_index=True)
            st.caption("Der Zielwert des Optimums ist eindeutig; bei Gleichstand kann CP-SAT einen anderen, gleich guten Plan liefern. Das Optimum erlaubt keine Fehlmenge; reaktiv und die Regeln sind mit Strafe bewertet.")
        st.markdown("**Vorgerechnet: die 30 Kleininstanzen je Wagen** (bewiesen optimal, Seeds 0 bis 29)")
        st.dataframe(UI.oracle_instances_frame(DATA, int(wagon)), width="stretch", hide_index=True, height=UI.CHART_TABLE_HEIGHT)
    with tabs[3]:
        st.markdown("Weitere vorgerechnete Auswertungen der Messreihe (200 Instanzen je Zelle, 27 Zellen).")
        names = [c["name"] for c in R.cells(DATA)]
        pick = st.selectbox("Zelle der Messreihe", names, index=names.index(cell["name"]) if cell["name"] in names else 0, key="cell_select",
                            format_func=lambda n: R.cell_label(R.by_name(DATA, n)) if n == "base" else f"{R.cell_group(R.by_name(DATA, n))}: {R.cell_label(R.by_name(DATA, n))}",
                            help="Reine Anzeigewahl: zeigt das H-γ-Gitter einer beliebigen gemessenen Zelle (auch der nicht auf den Reglerstufen liegenden: Geometrie, Meldegrenze, Tankgröße).")
        pcell = R.by_name(DATA, pick)
        st.plotly_chart(V.heatmap_figure(pcell, horizon, gamma), width="stretch", key="tab_heat")
        pg = R.p_gain(pcell, horizon, gamma)
        st.caption(f"Zelle **{R.cell_label(pcell)}**, umrandet Ihre Regel P({horizon}; {F.fmt_level(gamma)}): {F.fmt_band(pg['mean'], pg['se'])} %, Median {F.fmt_num(pg['median'], 1, True)} %, "
                   f"Quartile [{F.fmt_num(pg['q1'], 1, True)}; {F.fmt_num(pg['q3'], 1, True)}], in {F.fmt_num(100 * pg['share_loss'], 0)} % der Instanzen ein Verlust. Beste Zelle des Gitters, kreuzvalidiert: "
                   f"{F.fmt_num(pcell['cv_gain'], 1, True)} %.")
        st.markdown(f"**Die schlimmsten Kombinationen** ({_NEG[0]} von {_NEG[1]} im Mittel teurer als reaktiv):")
        st.dataframe(UI.negative_frame(DATA), width="stretch", hide_index=True)
        st.markdown("**Verteilung des Gewinns der Standardregel im Basisfall** (nicht schief verteilt: Mittel und Median liegen nahe beieinander):")
        st.caption(f"Einzelgewinne der Standardregel P(3; 0,5), Basis: Minimum {F.fmt_num(_DIST['min'], 1, True)}, Q1 {F.fmt_num(_DIST['q1'], 1, True)}, Median {F.fmt_num(_DIST['median'], 1, True)}, "
                   f"Q3 {F.fmt_num(_DIST['q3'], 1, True)}, P90 {F.fmt_num(_DIST['p90'], 1, True)}, Maximum {F.fmt_num(_DIST['max'], 1, True)} %; Instanzen mit Verlust {F.fmt_num(100 * _DIST['loss_share'], 1)} %.")

with st.expander("Wie funktioniert diese Demo?"):
    st.markdown(
        f"""
**Instanz.** {C.BASE_CFG['N']} Kunden gleichverteilt in 100 × 100, das Depot in der Mitte, Euklid-Distanz, Kosten = Strecke. Verbrauchsrate μ je Tag zwischen 6 und 14, Tank C = μ · (8 bis 14 Tage), Anfangsbestand
25 bis 100 % des Tanks. Der Tagesverbrauch ist Gamma-verteilt mit Mittel μ und Variationskoeffizient σ (σ = 0: deterministisch), für alle Regeln **derselbe** Strom. **Flotte und Lieferung:** Wagenkapazität Q, bis zu
F Touren je Tag, Lieferung am selben Morgen, der Tank wird bis zur Obergrenze gefüllt (Menge = min(C − Bestand; Q)), jeder Kunde höchstens einmal je Tag; reichen die Touren nicht, wird der am wenigsten dringende Kunde
zurückgestellt. Fehlmenge geht verloren (Strafe je Einheit, kein Rückstau).

**Zielgröße.** J = Fahrstrecke + Strafe · Fehlmenge − τ · (Endbestand − Anfangsbestand) über 120 Tage; τ = Fahrkosten je gelieferter Einheit der reaktiven Regel (bewertet den Restbestand, damit Vollfüllen am Ende nicht bevorzugt wird).
Der Gewinn ist in % der reaktiven Kosten angegeben. In der Messreihe ändert sich der Basisgewinn bei τ = 0 und 2 · τ kaum (17,0 / 17,0 / 17,1 %); in der Live-Instanz ist τ der Satz der reaktiven Regel auf dieser Instanz.

**Die Regeln.** **R (reaktiv):** fällig ist ein Kunde, wenn sein Bestand unter μ · (1 + κσ) fällt (er würde heute unter die Meldegrenze fallen; κ = 1); nur Fällige kommen auf die Tour, die Touren entstehen per Savings (Clarke-Wright) und 2-opt.
**P(H; γ) (bündeln):** wie R, dazu werden Kunden mit Restreichweite bis 1 + κσ + H Tage **auf den vorhandenen Touren** mitgenommen, wenn die billigste Einfügung höchstens γ · (Hin- und Rückfahrt Depot–Kunde) kostet und die Kapazität reicht; gibt es keine fällige Tour,
wird nichts mitgenommen. H = 0 ist R. **E(L) (früher liefern):** die Meldegrenze wird um L Tage angehoben, ohne Tourenbezug: die Gegenprobe „früher“ gegen „bündeln“. Es werden **immer alle drei Regeln** gerechnet, damit jeder Regler in jedem Zustand wirkt.

**Warum der Gewinn aus volleren Wagen kommt.** Im Basisfall fährt Bündeln {F.fmt_num(_MECH['routes'][1])} statt {F.fmt_num(_MECH['routes'][0])} Touren in 120 Tagen bei mehr Besuchen ({F.fmt_num(_MECH['visits'][0])} → {F.fmt_num(_MECH['visits'][1])}, davon {F.fmt_num(_MECH['opt_visits'])} Mitnahmen):
jeder Besuch ist kleiner als eine Wagenladung, und wer mitgenommen wird, spart eine eigene Fahrt. Früher zu liefern hebt die Meldegrenze, aber nicht auf die Tour: mehr Besuche, keine eingesparte Tour.

**Was die Live-Zahlen bedeuten.** Eine Instanz streut (Einzelgewinne im Basisfall {F.fmt_num(_DIST['q1'], 1)} bis {F.fmt_num(_DIST['q3'], 1)} % zwischen den Quartilen, Minimum {F.fmt_num(_DIST['min'], 1)}, Maximum {F.fmt_num(_DIST['max'], 1)}); die Live-Zahlen tragen deshalb die Überschrift
„eine Instanz“, die Meldung stützt sich auf die **vorgerechnete Messreihe**: „Bündeln lohnt“ (Gewinn mindestens 5 % und über 2 Standardfehler), „Bündeln bringt hier wenig“ (dazwischen) oder „Diese Mitnahmeregel ist hier teurer als reaktiv“ (Gewinn unter −2 Standardfehler).
**Die Stufen der Regler sind gemessene Zellen.** Die Messreihe variiert je Zelle nur **einen** Parameter ausgehend vom Basisfall; für Kombinationen jenseits davon zeigt die App die nächstliegende gemessene Zelle und sagt es.

**Grenzen dieses Modells** (bewusst so gewählt, damit die Aussage ehrlich bleibt):

- **Stark stilisiert** – ein Depot, Euklid, ein Fahrzeugtyp, keine Zeitfenster oder Fahrerregeln, Lieferung am selben Tag, verlorene Fehlmenge ohne Rückstau, jeder Besuch füllt den Tank voll, bekannte Raten und beobachteter Bestand, Verbrauch unabhängig zwischen Kunden und Tagen. Parameter **erfunden, nicht kalibriert**; der Basisfall Q/q ≈ 3 ist praxisnah für Tankwagen, aber nicht belegt.
- **Die Regel ist einfach und selbst gewählt.** Der Standardwert P(3; 0,5) wurde nach einer Erkundung auf den Seeds 0 bis 59 gewählt, die im Sweep enthalten sind (die „beste Zelle“ ist kreuzvalidiert); auf den frischen Seeds 200 bis 299 bestätigt sich der Basisgewinn (siehe README). Das größere Gitter könnte mehr bringen.
- **Die Tourgüte** von Savings + 2-opt ist nur bis 6 Kunden gegen das exakte Optimum geprüft (mittlere Lücke 0,12 %); die Vergleiche zwischen den Regeln nutzen dieselben Bausteine, absolute Kosten sind nicht überzubewerten.
- **Das Optimum** gibt es nur für die Kleininstanz (7 Kunden, 6 Tage, eine Tour je Tag, deterministisch), nicht für die stochastische Basis; die Lücke der Regel zum Optimum (12 bis 13 %) muss bei größeren Instanzen nicht gleich sein.
- **Keine Garantie auf weniger Fehlmengen:** Bündeln senkt sie im Mittel, aber in {F.fmt_num(100 * _BASE['short_cmp']['more'], 1)} % der Basis-Instanzen ist sie höher.
- **Nicht Teil dieser Demo** – Mengenwahl je Besuch, Zeitfenster und Fahrerregeln, mehrere Depots und Fahrzeugtypen, Korrelation im Verbrauch und Wochenmuster, Rückstau statt verlorener Fehlmenge, Lieferzeit größer als null, ein Optimum für die stochastische Basis, Kalibrierung an echten Tankdaten.
        """
    )

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Zustand und Dynamik.** Bestand $I_i(t)$ des Kunden $i$ am Tagesanfang mit $0 \le I_i \le C_i$; Verbrauch $d_i(t)$ mit Mittel $\mu_i$ (Gamma-verteilt, Variationskoeffizient $\sigma$). Entscheidung je Tag: Menge $x_i(t) \ge 0$ und Touren,
die alle Kunden mit $x_i > 0$ bedienen, je Tour $\sum x_i \le Q$, $x_i \le C_i - I_i$. Dynamik und Fehlmenge:
$$I_i(t+1) = \max\bigl(0,\; I_i(t) + x_i(t) - d_i(t)\bigr),\qquad s_i(t) = \max\bigl(0,\; d_i(t) - I_i(t) - x_i(t)\bigr).$$

**Ziel.** Über $T = 120$ Tage, mit Fahrstrecke $c(\text{Touren}_t)$, Strafe $p$ je Einheit Fehlmenge und Bewertung $\tau$ des Restbestands:
$$\min\; \sum_t c(\text{Touren}_t) + p \sum_{t,i} s_i(t) - \tau \Bigl(\sum_i I_i(T) - \sum_i I_i(0)\Bigr).$$

**Regeln.** Meldegrenze $\mu_i(1 + \kappa\sigma)$: fällig ist $i$, wenn $I_i(t) < \mu_i (1 + \kappa\sigma)$ (**R**, $\kappa = 1$). **P(H; γ):** nimm $i$ zusätzlich mit, wenn
$$\frac{I_i}{\mu_i} \le 1 + \kappa\sigma + H,\qquad \Delta_i \le \gamma \cdot 2\, d(0, i)$$
und die Kapazität reicht, mit der Mehrstrecke $\Delta_i$ der billigsten Einfügung in eine vorhandene Tour; $H = 0$ ist $R$. **E(L):** fällig ist $i$, wenn $I_i(t) < \mu_i (1 + \kappa\sigma + L)$, ohne Mitnahme.

**Gewinn und Meldung.** $g = 100 \cdot (J_R - J_\pi) / J_R$ in Prozent der reaktiven Kosten; in der Messreihe Mittel der gepaarten Differenz über $N = 200$ Instanzen mit Standardfehler $\text{SE} = s/\sqrt{N}$. Ein Vorzeichen gilt als belastbar, wenn
$|\bar g| > 2\,\text{SE}$. Meldung: „lohnt“ für $\bar g \ge 5\,\%$ und $\bar g > 2\,\text{SE}$, „teurer“ für $\bar g < -2\,\text{SE}$, sonst „bringt wenig“.

**Exakter Maßstab (Kleininstanz).** Binäre Bogenvariablen $x^t_{ij}$ (ein Kreis durch das Depot je Tag mit optionalen Kunden), ganzzahlige Mengen $q^t_i$ und Bestände $I^t_i \ge 0$:
$$\min\; \sum_{t,i,j} d_{ij}\, x^t_{ij} - \tau \sum_i I^T_i,\qquad I^t_i = I^{t-1}_i + q^t_i - \mu_i,\quad I^{t-1}_i + q^t_i \le C_i,\quad \sum_i q^t_i \le Q.$$

Implementiert in `irp_model.py` (Instanz), `irp_routing.py` (Savings, 2-opt, Einfügung), `irp_policy.py` (Regeln, Simulation), `irp_oracle.py` (CP-SAT, nur Kleininstanz), `irp_live.py` (Live-Instanz) und `irp_results.py` (Messreihe, Urteil).
        """
    )

st.markdown("---")

st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)
