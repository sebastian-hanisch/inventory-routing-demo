"""Konstanten der Inventory-Routing-Demo (Wer gehört heute auf die Tour?).

Modell und Zahlen aus messreihe_inventory_routing/ (siehe bestand-planung/plan_inventory_routing.html). Einstellbar sind die Kundenzahl, die
Wagenkapazität, die Touren je Tag, die Verbrauchsschwankung, die Fehlmengenstrafe, die Vorschau H, die Mitnahmeschwelle gamma, das frühere
Liefern L und der Seed. Alle Regler stehen bewusst auf den GEMESSENEN Stufen: zu jeder Reglerstellung gibt es eine vorgerechnete
Vergleichsspalte (die Messreihe variiert je Zelle einen Parameter ausgehend vom Basisfall; für Kombinationen jenseits der gemessenen
Zellen zeigt die App die nächstliegende Zelle und sagt es). Es laufen IMMER alle drei Regeln (R, P, E), kein Regler ist je wirkungslos."""

DAYS = 120                                   # Tage je Live-Instanz und je Instanz der Messreihe

# --- Regler ------------------------------------------------------------------------------------------------------------------------
CUSTOMER_OPTIONS, CUSTOMER_DEFAULT = (10, 20, 40), 20
CAPACITY_OPTIONS, CAPACITY_DEFAULT = (100, 150, 200, 300, 450, 600, 1000), 300
FLEET_OPTIONS, FLEET_DEFAULT = (1, 2, 6), 6
SIGMA_OPTIONS, SIGMA_DEFAULT = (0.0, 0.15, 0.3, 0.6, 1.0), 0.3
PENALTY_OPTIONS, PENALTY_DEFAULT = (1, 5, 25, 100), 5                      # je Einheit Fehlmenge, in Fahrstrecken-Einheiten
PENALTY_PFAC = {1: 0.01, 5: 0.05, 25: 0.25, 100: 1.0}                      # Strafe = pfac * Gebietsgröße (100): dieselben Werte wie der Sweep
HORIZON_OPTIONS, HORIZON_DEFAULT = (1, 2, 3, 4, 6, 8, 12), 3               # Vorschau H in Tagen
GAMMA_OPTIONS, GAMMA_DEFAULT = (0.1, 0.25, 0.5, 1.0), 0.5                  # Mitnahmeschwelle als Anteil der Hin- und Rückfahrt
EARLY_OPTIONS, EARLY_DEFAULT = (1, 2, 3), 2                                # früher liefern (Tage)
SEED_RANGE, SEED_DEFAULT = (0, 299), 211
CACHE_ENTRIES = 24                                                         # Ergebnisse je Einstellung (st.cache_data)

# Basisfall der Messreihe (Konfiguration "base" in tools/sweep.py); die übrigen Zellen weichen davon in einem Parameter ab
BASE_CFG = {"N": 20, "Q": 300.0, "F": 6, "sigma": 0.3, "pfac": 0.05, "kappa": 1.0, "depot": "center", "clustered": False, "tank_lo": 8.0, "tank_hi": 14.0}
MEASURED_N = 200                                                           # Instanzen je Zelle, Seeds 0..199
DEFAULT_H, DEFAULT_GAMMA = 3, 0.5                                          # Standardregel P(3; 0,5)

# --- Meldungen und Urteile ---------------------------------------------------------------------------------------------------------
WIN_MIN_PCT = 5.0                          # "Bündeln lohnt": mittlerer Gewinn mindestens 5 % ...
SE_FACTOR = 2.0                            # ... und ein Vorzeichen gilt nur, wenn der Betrag des Mittels mehr als 2 Standardfehler beträgt
STATE_LOHNT, STATE_WENIG, STATE_TEURER = "lohnt", "wenig", "teurer"

# --- Farben --------------------------------------------------------------------------------------------------------------------------
COLOR_R, COLOR_P, COLOR_E = "#c77700", "#2a6fb0", "#7d5ba6"
COLOR_GOOD, COLOR_BAD, COLOR_NEUTRAL, COLOR_LIGHT = "#2e7d4f", "#c0392b", "#9aa5b4", "#9fc2e6"
RULE_COLORS = {"R": COLOR_R, "P": COLOR_P, "E": COLOR_E}
TOUR_COLORS = ("#2a6fb0", "#c0392b", "#2e7d4f", "#7d5ba6", "#c77700", "#0f8b8d")
CHART_HEIGHT = 380

# --- Presets (Plan Abschnitt 7; Abnahmekriterien in irp_stories.py) ------------------------------------------------------------------
# Der Seed bestimmt nur die GEZEIGTE Instanz: die Messreihe steht auf den Seeds 0..199, die Anzeige-Seeds liegen ausserhalb. Seed 211 kommt aus
# tools/tune_presets.py (Suche über 200 bis 299): die Instanz erfüllt für alle fünf Presets die qualitativen Kriterien aus irp_stories.day_criteria und
# liegt im Standard mit +17,0 % genau auf dem Mittel der Messreihe (+17,0 %).
PRESETS = {
    "Standard": dict(customers=20, capacity=300, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=SEED_DEFAULT),
    "Wagen fast voll": dict(customers=20, capacity=100, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=SEED_DEFAULT),
    "Großer Wagen": dict(customers=20, capacity=600, fleet=6, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=SEED_DEFAULT),
    "Zu großzügig": dict(customers=20, capacity=150, fleet=6, sigma=0.3, penalty=5, horizon=12, gamma=1.0, early=2, seed=SEED_DEFAULT),
    "Knappe Flotte": dict(customers=20, capacity=300, fleet=1, sigma=0.3, penalty=5, horizon=3, gamma=0.5, early=2, seed=SEED_DEFAULT),
}
PRESET_HELP = {
    "Standard": "Der Grundfall: Bündeln spart etwa ein Sechstel der Fahrkosten, weil die Wagen voller werden; früher liefern ist dagegen teurer als reaktiv.",
    "Wagen fast voll": "Passt nur eine Lieferung in den Wagen, gibt es nichts zu bündeln.",
    "Großer Wagen": "Mit Platz für sechs Lieferungen spart Bündeln fast ein Drittel.",
    "Zu großzügig": "Wer jeden mitnimmt, der ungefähr auf dem Weg liegt (γ = 1,0, Vorschau 12 Tage, kleiner Wagen), zahlt drauf.",
    "Knappe Flotte": "Eine Tour je Tag: hier verhindert Vorziehen vor allem Fehlmengen.",
}
