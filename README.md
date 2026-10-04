# Inventory Routing: Wer gehört heute auf die Tour? – Streamlit-Demo

**[→ Demo live ausprobieren](https://sebastianhanisch-inventory-routing-demo.streamlit.app/)**

Interaktive Fall-Demo (Bestandsmanagement und Tourenplanung): Ein Depot beliefert Kunden mit Tanks (Heizöl, Gase, Getränke, Ersatzteile). Jeder Kunde verbraucht jeden Tag, und jemand entscheidet, **welche Kunden heute überhaupt auf die Tour gehören**, wie viel
sie bekommen und in welcher Reihenfolge – in der Fachsprache das **Inventory Routing Problem (IRP)**: Tourenplanung mit **Bestand beim Kunden** über mehrere Tage. Die Demo beantwortet: **Wie viel spart es, Kunden mit kleiner Restreichweite gleich mitzunehmen
(„bündeln"), statt erst dann zu liefern, wenn sie melden („reaktiv") – wovon hängt das ab, und wann lohnt es nicht?** Live auf **einer Instanz über 120 Tage** (alle drei Regeln über denselben Verbrauchsstrom, Karte je Tag mit Füllständen, Touren und Mitnehmern,
Halteliste, PDF des Tourenplans) und **vorgerechnet** über 200 gepaarte Instanzen je Zelle, die die Aussage tragen; das **exakte Optimum** einer Kleininstanz (7 Kunden, 6 Tage, CP-SAT) ist der Maßstab.

Teil des Portfolios für die Website „Sebastian Hanisch – Operations Research und Machine Learning". **Fall-Demo der Themenseite „Bestandsmanagement"** und Anschluss an die Straßenlogistik-Seite: die Tourenbausteine (Savings, 2-opt) kennt man aus `vrp_demo`, `alns-demo`
und `vrp-nachbarschaften-demo`; neu ist die **Bestandsschicht über die Tage**. Alle Instanzen sind deterministisch (Seed), die Streuung kommt aus der Instanzstichprobe.

## Warum dieses Problem

Im Portfolio kannte keine Demo Verbrauchsraten, Tankstände beim Kunden oder Nachschubtouren; das Kernmodell ist neu. Es ist **keine fünfte „starr gegen reaktiv"-Demo**: beide Regeln sind reine Online-Regeln auf demselben beobachteten Zustand, es gibt keinen starren Plan,
der von einer Störung getroffen wird, und die Unsicherheit ändert fast nichts (Befund 3). Es ist aber das **zweite Exemplar der Frage „wie weit vorausschauen"** (nach der Leercontainer-Repositionierung in `leercontainer-demo`), diesmal gekoppelt an eine Tour: die
Vorschau wirkt nur über den Einfügepreis der Tour, und sie kippt bei zu großzügiger Mitnahme ins Negative. Der eigene Aufhänger sind deshalb nicht „proaktiv ist besser", sondern **der Kapazitätshebel, die Gegenprobe „früher liefern gegen bündeln", die Kipp-Zone der Mitnahmeschwelle
und die Lücke zum Optimum**.

**Ehrlich zur Erwartbarkeit:** das Grundergebnis „Bündeln spart etwa ein Sechstel" steht ähnlich in der Literatur (Bündelungsvorteile von Inventory Routing gegenüber getrennter Planung in der Größenordnung 10–30 %). Es ist hier eine eigene Messung an einem eigenen, einfachen
Modell, keine Übernahme und keine Neuentdeckung; tragfähig machen den Fund die Zusatzbefunde 2, 3, 5 und 7.

## Befunde und Korrekturen gegenüber dem Plan

Der Detailplan (`plan_inventory_routing`) und die Vorab-Messreihe (`messreihe_inventory_routing`) sind verbindlich; beim Bau haben sich diese Punkte ergeben:

- **AP 0 (a) – Zellen-Abdeckung.** Die Regler stehen auf gemessenen Stufen (3 · 7 · 3 · 5 · 4 = 1260 Kombinationen aus Kunden, Wagen, Touren je Tag, Verbrauchsschwankung und Fehlmengenstrafe). Die Messreihe variiert je Zelle **einen** Parameter ausgehend vom Basisfall; auf den
  Reglerstufen liegen **19** der 27 Zellen, deshalb sind **19 von 1260** Kombinationen exakt gemessen (127 weichen in einem Parameter ab, 374 in zweien, 500 in dreien, 240 in vieren). Die Vergleichsspalte zeigt für alle anderen die nächstliegende gemessene Zelle mit ausdrücklichem
  Hinweis (Abstand = Summe der Stufenabstände; bei Gleichstand gewinnt die Zelle mit dem Parameter weiter vorn in der Reihe Wagen, Flotte, Kunden, Verbrauchsschwankung, Strafe). **Acht Zellen sind nicht über die Regler erreichbar** (geklumpte Kunden, Depot weit außen und in der Ecke,
  Meldegrenze κ = 0 und 2, kurze und lange Tanks, Wagen 250 mit einer Tour); sie stehen in der Regime-Tabelle und im Reiter „Messreihe". Eine Zusatzmessung (zum Beispiel Flotte 2 mit Wagen 300) wurde nicht gemacht: der Hinweis genügt.
- **AP 0 (b) – Der Standardwert P(3; 0,5) hält auf frischen Seeds.** Er wurde nach einer Erkundung auf den Seeds 0 bis 59 gewählt, die im Sweep (Seeds 0 bis 199) enthalten sind. Bestätigung auf den **Seeds 200 bis 299** (100 Instanzen, die weder bei der Wahl noch im Sweep vorkamen;
  `tools/confirm_default.py`): Basisfall **+16,7 ± 0,3 %** (Median 16,5 %, Quartile [14,8; 18,1], Minimum 9,3, Maximum 24,9, in 0 % der Instanzen ein Verlust) gegen +17,0 ± 0,3 % im Sweep; die Differenz der Mittel ist −0,3 % (−0,8 Standardfehler der Differenz). Auch Wagen 100 (+0,4 ± 0,1 %),
  Wagen 600 (+29,5 ± 0,5 %) und die knappe Flotte (+19,4 ± 0,5 %) stimmen mit dem Sweep überein. Die Wahl hängt also nicht an den Sweep-Seeds.
- **AP 0 (c) – Preset-Seed außerhalb der Stichprobe.** Suche über die Seeds 200 bis 299 (`tools/tune_presets.py`): 97 von 100 Seeds erfüllen für alle fünf Presets die qualitativen Tageskriterien; gewählt ist **Seed 211**, dessen Standard-Instanz mit +17,0 % genau auf dem Mittel der Messreihe liegt. Alle Presets
  stehen auf gemessenen Stufen (jedes liegt auf einer exakt gemessenen Zelle).
- **Befund zur Rechenzeit des Optimums:** CP-SAT löst die Kleininstanz mit **8 Arbeitern in 0,3 bis 1,2 s**, mit 1 bis 4 Arbeitern dauert dieselbe Instanz **4 bis 25 s** (gemessen; im Portfolio der 8 Arbeiter steckt eine schnelle Teilstrategie). Die App
  und die Tests nutzen 8 Arbeiter; auf einem Rechner mit wenigen Kernen kann es länger dauern, das Zeitlimit (20 s) liefert dann eine nicht bewiesene Lösung mit ausdrücklichem Status.
- **Zahl aus der Vorab-Messreihe korrigiert:** `ERGEBNIS.md` nennt für Wagen 250 im Optimum 2,50 Touren in 6 Tagen; der Bericht der Messreihe (`oracle_report_Q250.txt`) und die Neuberechnung ergeben **2,53**. Die README und die App verwenden 2,53.
- **Der Plan wich an vier Stellen ab:** (1) der Seed-Bereich der Instanz ist 0 bis 299 statt 0 bis 199, damit die Anzeige-Seeds außerhalb der Messreihe (200 bis 299) einstellbar sind; die Kennzahlen der Messreihe stehen weiterhin auf 0 bis 199; (2) die Anzeigewahl neben den Tages-Karten hat
  die Optionen „Bündeln neben Reaktiv" und „Früher liefern neben Reaktiv" statt „Reaktiv, Bündeln": beide Regeln sind ohnehin immer gerechnet, so kann die Gegenprobe auf der Karte gezeigt werden; (3) die Regime-Tabelle und die Diagramme des Kernabschnitts zeigen den Gewinn der **eingestellten** Regel P(H; γ) in allen 27
  Zellen (die Messreihe hat alle 28 Kombinationen je Zelle), damit auch Vorschau und Schwelle dort wirken; (4) τ (Bewertung des Restbestands) ist in der Live-Instanz der Fahrkostensatz je gelieferter Einheit der reaktiven Regel **auf dieser Instanz** (in der Messreihe das Mittel über 200 Instanzen; der Basisgewinn
  ändert sich bei τ = 0 und 2 · τ kaum: 17,0 / 17,0 / 17,1 %).
- **Plan des Optimums nicht eindeutig:** Bei Seed 19 (Q = 150) lieferte ein Wiederholungslauf des exakten Maßstabs 11 statt 12 Besuche im Optimum bei gleichem Zielwert: **der Zielwert ist eindeutig, der Plan nicht** (bei Gleichstand liefert CP-SAT andere, gleich gute Pläne). Deshalb prüfen die Tests nur den Zielwert.

## Modell

Fachmodell und Annahmen (Erläuterung im Expander „Wie funktioniert diese Demo?", Formeln im Expander „📐 Mathematische Formulierung"):

- **Instanz (Basis).** 20 Kunden gleichverteilt in 100 × 100, Depot in der Mitte, Euklid-Distanz, Kosten = Strecke; Verbrauchsrate μ je Tag zwischen 6 und 14, Tank C = μ · (8 bis 14 Tage), Anfangsbestand 25 bis 100 % des Tanks; Tagesverbrauch Gamma-verteilt mit Mittel μ und
  Variationskoeffizient σ (Basis 0,3; σ = 0 deterministisch), für alle Regeln **derselbe** Strom.
- **Flotte und Lieferung.** Wagenkapazität Q = 300, bis zu F = 6 Touren je Tag, Lieferung am selben Morgen, der Tank wird bis zur Obergrenze gefüllt (Menge = min(C − Bestand; Q)), jeder Kunde höchstens einmal je Tag; reichen die Touren nicht, wird der am wenigsten dringende Kunde zurückgestellt.
- **Zielgröße.** J = Fahrstrecke + Strafe · Fehlmenge − τ · (Endbestand − Anfangsbestand) über 120 Tage; die Fehlmenge geht verloren (Strafe 5 je Einheit); τ bewertet den Restbestand, damit Vollfüllen am Ende nicht bevorzugt wird. Gewinn in % der mittleren reaktiven Kosten.
- **R – reaktiv:** fällig ist ein Kunde, wenn sein Bestand unter μ · (1 + κσ) fällt (κ = 1); nur Fällige kommen auf die Tour (Savings + 2-opt).
- **P(H; γ) – bündeln:** wie R, dazu werden Kunden mit Restreichweite bis 1 + κσ + H Tage **auf den vorhandenen Touren** mitgenommen, wenn die billigste Einfügung höchstens γ · (Hin- und Rückfahrt Depot–Kunde) kostet und die Kapazität reicht; gibt es keine fällige Tour, wird nichts mitgenommen. H = 0 ist R (getestet).
  Gitter H ∈ {1, 2, 3, 4, 6, 8, 12} Tage, γ ∈ {0,1; 0,25; 0,5; 1,0}, Standard P(3; 0,5).
- **E(L) – früher liefern:** die Meldegrenze wird um L ∈ {1, 2, 3} Tage angehoben, ohne Tourenbezug: die Gegenprobe „früher" gegen „bündeln".
- **Exakter Maßstab:** 7 Kunden, 6 Tage, deterministisch, eine Tour je Tag, Tanks 3 bis 6 Tage, Wagen 150 bzw. 250, ganzzahlig; CP-SAT (Kreis mit optionalen Knoten je Tag, Mengen, Bestand ≥ 0); 30 Instanzen je Wagen, alle bewiesen optimal.

## Methodik

- **Live:** eine Instanz über 120 Tage, alle drei Regeln bei jeder Reglerstellung neu gerechnet (Instanz und Regeln zusammen etwa 0,013 bis 0,025 s, deshalb kein Knopf; die App speichert das Ergebnis je Einstellung mit `st.cache_data`). Es werden **immer alle drei Regeln** gerechnet, damit kein Regler wirkungslos ist.
- **Vorgerechnet:** 27 Konfigurationen × 32 Regeln × 200 Instanzen × 120 Tage (Sweep 126 s auf 12 Prozessen), gepaart auf denselben Instanzen und Verbrauchsströmen; Gewinn = Mittel der gepaarten Differenz ± Standardfehler, dazu Median und Quartile der Einzelgewinne; die „beste Zelle" des Gitters wird kreuzvalidiert gewählt
  (Auswahl auf den geraden, Bewertung auf den ungeraden Seeds und umgekehrt). Ein Vorzeichen gilt nur ab 2 Standardfehlern.
- **Meldung in drei Zuständen** (aus der Messreihe, nicht aus der einen Instanz): „Bündeln lohnt" (Gewinn der eingestellten Regel mindestens 5 % und über 2 Standardfehler), „Bündeln bringt hier wenig" (dazwischen), „Diese Mitnahmeregel ist hier teurer als reaktiv" (unter −2 Standardfehler).
- **Kern unverändert:** `irp_model`, `irp_routing` und `irp_policy` sind mechanisch aus `ir.py` der Messreihe aufgeteilt; die Bitgleichheit ist belegt (Abschnitt Tests).

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stammen aus `data/irp_results.json` (200 Instanzen je Zelle) und werden in `tests/test_claims.py` nachgerechnet.

| Frage | Befund |
|---|---|
| 1 · Größenordnung | Bündeln P(3; 0,5) spart im Basisfall **+17,0 ± 0,3 %** Fahrkosten (Median 16,9 %, Quartile [15,1; 18,7], Minimum 10,5, Maximum 24,8; in **0 %** der 200 Instanzen ein Verlust, keine schiefe Verteilung). Die kreuzvalidiert gewählte beste Zelle des Gitters (H = 8, γ = 0,25) bringt +20,5 %. Mechanismus je 120 Tage: Touren 132,8 → 99,7 (−25 %), Besuche 238,1 → 260,5 (+9 %, davon 101,7 Mitnahmen), Auslastung 0,59 → 0,79, Fahr-km 14.694 → 12.220 (−16,8 %). Der Gewinn kommt aus **weniger, volleren Touren**, nicht aus weniger Besuchen. |
| 2 · Kapazitätshebel | Gewinn über die Wagenkapazität Q = 100 / 150 / 200 / 300 / 450 / 600 / 1000 (Q/q = 1,1 / 1,5 / 2,0 / 3,0 / 4,5 / 6,0 / 10,1): **+0,4 / +6,3 / +10,7 / +17,0 / +25,4 / +29,9 / +33,2 %**; beste Zelle kreuzvalidiert +1,3 / +9,0 / +13,7 / +20,5 / +29,6 / +34,6 / +40,2 %. Passt nur eine Lieferung in den Wagen, gibt es nichts zu bündeln. Rangkorrelation Gewinn gegen Q/q über 23 Konfigurationen: 0,70 (Geometrie und Dichte wirken zusätzlich). |
| 3 · Unsicherheit spielt kaum eine Rolle | Verbrauchsschwankung σ = 0 / 0,15 / 0,3 / 0,6 / 1,0: **+16,5 / +17,0 / +17,0 / +17,1 / +17,3 %**; Fehlmengenstrafe 1 / 5 / 25 / 100: +16,9 / +17,0 / +17,5 / +19,0 %. Ohne jede Zufallsschwankung bleibt der Effekt vollständig: es ist ein **Auslastungs-, kein Sicherheitseffekt**. |
| 4 · Kipp-Zone der Mitnahmeschwelle | Im Basisfall sind 26 von 28 Gitterkombinationen billiger als reaktiv, 5 liegen höchstens 3 Prozentpunkte unter der besten. Über alle 27 Zellen sind **46 von 756** proaktiven Kombinationen im Mittel teurer als reaktiv, davon **44 mit γ = 1,0** und keine mit γ ≤ 0,25; die schlimmsten: Wagen 150 mit zwei Touren, H = 12, γ = 1: −30,0 ± 0,7 %; Wagen 150 (H = 12, γ = 1): −26,3 %; Wagen 100: −15,3 %. Vorschau allein, ohne γ-Schwelle, ist keine Lösung. |
| 5 · Früher liefern ist keine Alternative | E(L) mit L = 1 / 2 / 3 Tage: **−7,0 / −16,3 / −27,1 %** (teurer als reaktiv); in 75 von 81 Regel-Zellen-Paaren teurer (positiv nur bei knapper Flotte oder sehr hoher Strafe): mehr Besuche ohne Tourenbezug (303,6 gegen 238,1), keine eingesparte Tour (137,0 gegen 132,8), Fahr-km 17.341 gegen 14.694. E(2) vermeidet dafür in der Basis jede Fehlmenge (0,0 gegen 14,4 Einheiten): Sicherheit wird hier mit Fahrkosten erkauft, Bündeln spart sie. Bündeln ist Selektion nach Tourenlage, nicht Vorlaufzeit. |
| 6 · Knappe Flotte | Eine Tour je Tag (Wagen 300): **+19,2 ± 0,3 %**, Fehlmenge 386 → 130 Einheiten; Wagen 250: +30,4 ± 1,1 %, Fehlmenge 1973 → 639; zwei Touren mit Wagen 150: +7,9 ± 0,3 % (511 → 353). Im Basisfall sinkt die Fehlmenge im Mittel (14,4 → 8,5 Einheiten), **aber in 22,5 % der Instanzen ist sie beim Bündeln höher: keine Garantie**. |
| 7 · Exakter Maßstab | Kleininstanz, 30 Instanzen je Wagen, alle bewiesen optimal: reaktiv liegt **28,7 ± 2,1 %** (Wagen 150) bzw. **32,3 ± 2,2 %** (Wagen 250) über dem Optimum, P(3; 0,5) 12,7 ± 1,8 % bzw. 11,5 ± 1,4 %: die Regel schließt **48 % bzw. 61 %** der reaktiven Lücke, also etwa die **Hälfte**. Touren in 6 Tagen (Optimum / reaktiv / P(3; 0,5)): 3,03 / 4,67 / 3,70 bzw. 2,53 / 4,67 / 3,37. |
| Standardwert nach Erkundung gewählt | P(3; 0,5) wurde auf den Seeds 0 bis 59 gewählt; auf den frischen Seeds 200 bis 299: +16,7 ± 0,3 % (AP 0 (b) oben). |

## Ehrliche Grenzen

- **Stark stilisiert:** ein Depot, Euklid-Distanz, ein Fahrzeugtyp, keine Zeitfenster oder Fahrerregeln, Lieferung am selben Tag, verlorene Fehlmenge ohne Rückstau, jeder Besuch füllt den Tank voll, bekannte Raten und beobachteter Bestand, Verbrauch unabhängig zwischen Kunden und Tagen.
  **Parameter erfunden, nicht kalibriert:** Gebiet, Raten, Tankgrößen, Wagengröße, Strafe; die Gewinne in Prozent gelten für dieses Modell. Der Basisfall Q/q ≈ 3 ist praxisnah für Tankwagen, aber nicht belegt.
- **Die Regel P(H; γ) ist einfach und selbst gewählt**; die beste Gitterzelle liegt oft am Rand (H = 8 bis 12, γ = 0,1 bis 0,25), ein größeres Gitter könnte mehr bringen. Der Standardwert wurde nach einer Erkundung gewählt (siehe AP 0 (b)); die Kreuzvalidierung der „besten Zelle" trennt Auswahl und Bewertung,
  bleibt aber an dieser Instanzfamilie orientiert.
- **Tourgüte:** Savings + 2-opt sind nur bis 6 Kunden gegen das exakte CVRP geprüft (mittlere Lücke 0,12 %, Median 0, Maximum 6,13 %, in 93 % der Instanzen exakt optimal); bei 20 bis 40 Kunden ist die Güte nicht gemessen. Die Vergleiche zwischen den Regeln nutzen dieselben Bausteine und sind davon weniger betroffen als absolute Kosten.
- **Das Optimum gibt es nur für die Kleininstanz** (7 Kunden, 6 Tage, eine Tour je Tag, deterministisch, τ = 0,6). **Für die stochastische Basis (20 Kunden, 120 Tage) gibt es kein Optimum**, es ist nicht gerechnet; die Lücke von 12 bis 13 % muss bei größeren Instanzen nicht gleich sein.
- **Live-Zahlen sind eine Instanz:** die Einzelgewinne streuen (Basis: Quartile 15,1 bis 18,7 %); die Meldung stützt sich deshalb auf die vorgerechnete Messreihe. Für Kombinationen jenseits der gemessenen Zellen zeigt die Vergleichsspalte die nächstliegende Zelle (nur 19 von 1260 Kombinationen sind exakt gemessen).
- **Keine Garantie auf weniger Fehlmengen:** Bündeln senkt sie im Mittel, aber in 22,5 % der Basis-Instanzen ist sie höher.
- **τ-Bewertung des Restbestands** ist eine Modellwahl; bei knappen Flotten und in der Kleininstanz (6 Tage) ist ihr Einfluss stärker als im Basisfall.
- **Flottengrenze:** bei knapper Flotte wird nach Dringlichkeit zurückgestellt, nicht optimiert.

## Tests

`python -m pytest tests/ -v` (Stand: 418 Tests grün, 76 s auf dem Entwicklungsrechner) – die Tests laufen ohne Netz, ohne Wall-Clock-Annahmen und ohne Abhängigkeit vom Zufallsgenerator:

- **Bitgleichheit zur Messreihe** (`test_frozen_reference.py`): 14 eingefrorene Instanzen (Koordinaten, Verbrauchsrate, Tank, Anfangsbestand und der komplette Verbrauchsstrom, verlustfrei) aus 12 Konfigurationen × alle 32 Regeln × alle 12 Kennzahlen stimmen mit `raw_sweep.npz` der Messreihe überein. Die CI installiert immer das neueste NumPy, dessen Zufallsströme sich ändern dürfen:
  deshalb laufen Referenzwerte über eingefrorene Instanzen, `make_instance` wird nur auf Invarianten geprüft (und auf Gleichheit mit der eingefrorenen Instanz nur, wenn die NumPy-Version dieselbe ist). Ganzzahlige Kennzahlen müssen exakt stimmen, Gleitkommasummen auf einer anderen Plattform bis auf 1e-9 relativ (NumPy summiert je nach Version anders), auf derselben Plattform strikt gleich.
- **Korrektheits-Checks der Messreihe** (`test_checks.py`, `tests/irp_checks.py`): Grenzfall ohne Verbrauch, H = 0 == reaktiv, unabhängige Neubewertung und exakte Bestandsbilanz, Handinstanz (Tourkosten 14 / 24 / 18), Savings + 2-opt nie besser als das exakte CVRP, Zweig-Tests je Regel (Nullspalten-Falle; die **Fehlmenge bei σ = 0 ist exakt 0 und richtig**: die Meldegrenze deckt den Tagesverbrauch),
  Determinismus; CP-SAT-Modell gegen eine unabhängige exakte Rechnung. Das **volle Bau-Gate** (`tools/check_full.py`, einmal lokal) wiederholt außerdem den GANZEN Sweep gegen die Rohdaten und alle 60 Kleininstanzen.
- **Einzelbausteine an Handinstanzen** (`test_model_units.py`), **CP-SAT nur im Zielwert** (`test_oracle.py`, der Plan ist bei Gleichstand nicht eindeutig), Ergebnisdatei, Zell-Zuordnung und Urteil (`test_results.py`), Presets (`test_presets.py`), Abnahmekriterien an künstlichen Schwellen, jedes Kriterium einzeln (`test_stories.py`, `test_preset_stories.py`), Live-Instanz, Panel, Figuren, PDF
  (jedes Sonderzeichen, das die Kernschriften abstürzen lässt), Werkzeuge und die Zahlen dieser README (`test_claims.py`).
- **Streamlit-AppTest** (`test_app.py`): Skelett und wörtlicher Footer, jedes Preset, Permalink, alle Regler an Min und Max, alle drei Meldungszustände, kein wirkungsloser Regler, Tagesansicht, alle Kernabschnitte, Exakt-Tab (Kleininstanz, Pause zwischen zwei Aufrufen, unzulässige Instanz, fehlender Löser), PDF.
- **Fehler-Einbau-Test** (`tools/mutation_check.py`): 812 maschinell erzeugte Fehler (Vergleichsoperatoren, Plus/Minus, and/or, Wahrheitswerte, min/max, Zahlen +1), verteilt über alle Module und Werkzeuge, je einer in einer eigenen Kopie; die Kopie der unveränderten Dateien besteht zuvor die Tests (Selbstprüfung des Werkzeugs). Erster Lauf: 590 von 814 gefunden, 224 überlebt. Für die echten Lücken (Gleichstände beim Savings-Verfahren, Kapazitätstoleranz, Schwellenränder, Standardwerte, Wortlaut der Meldungen, Kennzahlen der Werkzeuge, Fehlerbalken und Legenden der Figuren, PDF-Kopfzeilen) kamen die Tests `test_gaps_*.py`; der Wiederholungslauf der Überlebenden fand 130 weitere, ein dritter Lauf nach `test_gaps_round2.py` 33 weitere (zwei Mutanten entfielen mit entferntem totem Code). Endstand: **753 von 812 gefunden, 59 überlebend**. Die 59 sind Vergleiche am Maßnullpunkt (`<` gegen `<=` bei exakt gleichen Gleitkommazahlen), Grenzen wirkungsloser Schleifen, Darstellungsparameter und Werkzeug-Rahmen; sie sind in `tools/mutants.py` (`EQUIVALENT_NOTES`) nach Gruppen begründet. Das ist eine Einschätzung, kein Beweis der Gleichwertigkeit. Laufzeit je Lauf etwa 15 Minuten bei 10 parallelen Aufträgen.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt: Hauptansicht (live), Kernabschnitt ② (vorgerechnet), Ansichten, Expander, Footer |
| `irp_constants.py` | Regler-Stufen, Presets, Farben, feste Parameter |
| `irp_presets.py` | Regler-Spezifikation, Permalink (Begrenzen und Einrasten), Presets, Seed-Knopf |
| `irp_model.py`, `irp_routing.py`, `irp_policy.py` | Instanz, Tourenbausteine (Savings, 2-opt, Einfügung), Regeln R / P / E und Simulation: mechanisch aus `ir.py` der Messreihe, Logik unverändert |
| `irp_oracle.py` | CP-SAT-Modell der Kleininstanz (OR-Tools erst beim Aufruf importiert), Größenbegrenzung, Vergleich mit den Regeln |
| `irp_live.py` | Live-Instanz: drei Regeln, Bestandsverlauf, Tagessicht, Mitnehmer |
| `irp_results.py` | Laden und Auswerten von `data/irp_results.json`: Zell-Zuordnung, Urteil, Gitter, Kapazitätshebel, Regime, Exakter Maßstab |
| `irp_stories.py` | Abnahmekriterien der Presets |
| `irp_visualization.py`, `irp_ui_panel.py`, `irp_pdf_export.py`, `irp_format.py` | Figuren, Panel und Tabellen, Tourenplan-PDF (fpdf2, Sonderzeichen-Bereinigung), Zahlenformate |
| `data/irp_results.json` | Aggregate der Messreihe (27 Zellen, Gitter, Regeln E, Mechanismus) und der Orakel-Läufe (klein, keine Rohdaten je Instanz) |
| `tools/sweep.py`, `tools/analyze.py`, `tools/oracle_run.py` | Reproduktion der Messreihe (Sweep 126 s auf 12 Prozessen; Orakel unter 2 Minuten je Wagen; nicht in der CI) |
| `tools/build_results.py`, `tools/freeze_reference.py`, `tools/confirm_default.py`, `tools/tune_presets.py` | Ergebnisdatei bauen, Testreferenz einfrieren, Standardwert bestätigen, Preset-Seeds suchen |
| `tools/check_full.py`, `tools/mutation_check.py`, `tools/gen_mutants.py`, `tools/mutants.py` | volles Bau-Gate, Fehler-Einbau-Test mit Mutantenliste |
| `tests/` | Tests (siehe oben), `tests/data/` die eingefrorenen Instanzen |
| `requirements.txt`, `requirements-dev.txt`, `.github/workflows/tests.yml` | Abhängigkeiten (ungepinnt, `>=`), CI (bei Push und wöchentlich) |

## Bewusst nicht umgesetzt

- **Mengenwahl je Besuch** (ein Standardelement vieler IRP-Regeln; hier füllt jeder Besuch den Tank voll), **Zeitfenster und Fahrerregeln** (siehe `fernverkehr-demo`), **mehrere Depots und Fahrzeugtypen**.
- **Korrelation im Verbrauch, Wochenmuster, Rückstau** statt verlorener Fehlmenge, Lieferzeit größer als null.
- **Ein Optimum für die stochastische Basis** und eine Kalibrierung an echten Tankdaten.
- Ein größeres Regelgitter oder eine gelernte Mitnahmeregel.

## Reproduktion der Messreihe

Die Rohdaten (`raw_sweep.npz`, 9 MB) stehen nicht im Repo; die Aggregate stehen in `data/irp_results.json`.

```bash
python tools/sweep.py                # 27 Konfigurationen x 32 Regeln x 200 Instanzen x 120 Tage, 126 s auf 12 Prozessen -> tools/_out/raw_sweep.npz
python tools/analyze.py              # -> tools/_out/sweep_data.json (Aggregate, Kreuzvalidierung, Fehlmengen-Vergleich)
python tools/oracle_run.py 30 120 150   # Kleininstanz gegen CP-SAT, Wagen 150 (und 250) -> tools/_out/oracle_data_Q150.json
python tools/build_results.py        # -> data/irp_results.json (ohne Argumente aus ../bestand-planung/messreihe_inventory_routing/ und tools/_out/)
python tools/confirm_default.py      # Standardwert auf den Seeds 200 bis 299
```

`tools/_out/` steht in der `.gitignore`. Ein Wiederholungslauf des ganzen Sweeps mit den `irp_`-Modulen (`tools/check_full.py`) lieferte alle 2.073.600 Werte (27 × 32 × 12 × 200) **bitgleich** zur Messreihe; die 60 Kleininstanzen liefern dieselben Zielwerte (größte Abweichung 1e-13),
Seed 19 bei Wagen 150 einen anderen, gleich guten Plan (siehe oben).

## Lokal ausführen

```bash
pip install -r requirements-dev.txt
streamlit run app.py
```

Tests: `python -m pytest tests/ -v`. Volles Bau-Gate: `python tools/check_full.py`. Preset-Abstimmung: `python tools/tune_presets.py [erster_Seed] [letzter_Seed]`. Fehler-Einbau: `python tools/mutation_check.py [Modul] [--jobs N]`.

## Verwandte Demos mit demselben mathematischen Modell

Stand 2026-09-24. Kein Portfolio-Stück teilt das ganze Modell (Verbrauch und Tankstände beim Kunden über mehrere Tage, Nachschubtouren); geteilt sind Bausteine und Fragestellungen:

- **Tourenbausteine:** `vrp_demo`, `alns-demo` und `vrp-nachbarschaften-demo` (kapazitiertes Tourenproblem, Savings und 2-opt): hier eine eigene kleine Fassung, gegen ein exaktes CVRP geprüft; neu ist die Bestandsschicht.
- **Das Vorschau-Fenster-Muster** („wie weit vorausschauen") steht als erstes Exemplar in `leercontainer-demo`; diese Demo ist das zweite, gekoppelt an eine Tour, mit dem Kipppunkt bei zu großzügiger Mitnahme.
- **Schwestern im Tourenplanungs-Zweig:** `nahverkehr-demo` (ein Tag mit Ereignissen: Same-Day-Aufträge, Änderungen, Preis der Planänderung) und `fernverkehr-demo` (Ressourcen entlang der Route). Bei ihnen ist der Plan ein Tag; hier ist er ein fortlaufender Zustand über 120 Tage.
- **Kein „starr gegen reaktiv":** das Muster (`fahrzeugflotte-demo`, `robuste-kaiplatz-demo`, `blockzuweisung-demo`, `hofrobust-demo`) kommt hier nicht vor: beide Regeln sind reine Online-Regeln, die Störung (σ) ändert den Gewinn kaum.
- **Hindsight-Optimum** als Maßstab (`revenue-management-demo`, `routenresilienz-demo`): hier als exakter Mehrtageslöser auf einer Kleininstanz.

---

Gebaut mit Streamlit, Plotly, OR-Tools und fpdf2.

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zum Thema: [Bestandsmanagement optimieren](https://sebastianhanisch.net/bestandsmanagement-optimierung.html).
