"""Wiederverwendbares Panel: Kennzahlen (2 x 2), Meldung in drei Zuständen, Vergleichstabelle Instanz gegen Messreihe, Tagesansicht und die Tabellen der
Kernabschnitte ② und der Ansichten (Plan Abschnitt 6/8). Die Tabellen sind reine Funktionen (pandas), die Darstellung ist getrennt."""
import pandas as pd
import streamlit as st

import irp_constants as C
import irp_format as F
import irp_live as LV
import irp_results as R
import irp_visualization as V

CHART_TABLE_HEIGHT = C.CHART_HEIGHT + 60


def rule_label(rule, H, gamma, early):
    """Beschriftung einer Regel mit ihren Parametern: 'Reaktiv', 'Bündeln P(3; 0,5)', 'Früher liefern E(2)'."""
    return {"R": "Reaktiv", "P": f"Bündeln P({int(H)}; {F.fmt_level(gamma)})", "E": f"Früher liefern E({int(early)})"}[rule]


def utilization(res):
    """Mittlere Wagenauslastung (Ladung je Tour in Teilen der Kapazität)."""
    return res["util_sum"] / res["n_routes"] if res["n_routes"] else 0.0


# ---------------------------------------------------------------------------------------------------
# Live-Instanz: Kennzahlen und Meldung
# ---------------------------------------------------------------------------------------------------
def live_numbers(live):
    """Kennzahlen der Live-Instanz je Regel: Kosten J, Gewinn gegenüber reaktiv (%), Differenz der Kosten, Touren, Besuche, Mitnahmen, Auslastung, Fehlmenge, km."""
    jr = live["rules"]["R"]["J"]
    out = {}
    for name, rl in live["rules"].items():
        res = rl["res"]
        out[name] = dict(J=rl["J"], gain=rl["gain"], diff=rl["J"] - jr, routes=res["n_routes"], visits=res["n_visits"], picks=res["opt_visits"],
                         util=utilization(res), short=res["short"], km=res["routing"], deferred=res["deferred"])
    return out


def render_metrics(columns, live):
    """Vier Kennzahlen im 2 x 2-Raster: Kosten reaktiv, Kosten Bündeln mit Gewinn, Kosten früher liefern mit Gewinn, Touren / Auslastung / Fehlmenge je
    Regel. Eine Instanz. Das Delta ist immer 'Kosten der Regel minus Kosten reaktiv' (negativ = billiger, deshalb delta_color='inverse')."""
    x = live_numbers(live)
    H, g, L = live["H"], live["gamma"], live["early"]
    columns[0].metric("Kosten reaktiv (R)", F.fmt_cost(x["R"]["J"]), delta="Bezug: nur Fällige auf die Tour", delta_color="off", delta_arrow="off",
                      help="Zielgröße J = Fahrstrecke + Strafe · Fehlmenge − τ · Bestandsänderung über 120 Tage, für die reaktive Regel (nur Kunden unter der "
                           "Meldegrenze kommen auf die Tour). Eine einzelne Instanz.")
    columns[1].metric(f"Kosten {rule_label('P', H, g, L)}", F.fmt_cost(x["P"]["J"]),
                      delta=f"{F.fmt_cost(x['P']['diff'], True)} ({F.fmt_num(x['P']['gain'], 1, True)} %)", delta_color="inverse",
                      help="Wie reaktiv, dazu werden Kunden mit kleiner Restreichweite auf den vorhandenen Touren mitgenommen, wenn die Einfügung höchstens "
                           "γ · Hin- und Rückfahrt zum Kunden kostet. Delta = Kosten Bündeln minus Kosten reaktiv (negativ heißt billiger), in Klammern der Gewinn in % der reaktiven Kosten.")
    columns[2].metric(f"Kosten {rule_label('E', H, g, L)}", F.fmt_cost(x["E"]["J"]),
                      delta=f"{F.fmt_cost(x['E']['diff'], True)} ({F.fmt_num(x['E']['gain'], 1, True)} %)", delta_color="inverse",
                      help="Die Meldegrenze wird um L Tage angehoben, ohne auf die Tour zu schauen (Gegenprobe: früher gegen bündeln). "
                           "Delta = Kosten früher liefern minus Kosten reaktiv (positiv heißt teurer), in Klammern der Gewinn in % der reaktiven Kosten.")
    columns[3].metric("Touren in 120 Tagen (R / P / E)", f"{x['R']['routes']} / {x['P']['routes']} / {x['E']['routes']}",
                      delta=f"Auslastung {F.fmt_num(x['R']['util'], 2)} / {F.fmt_num(x['P']['util'], 2)} / {F.fmt_num(x['E']['util'], 2)}",
                      delta_color="off", delta_arrow="off",
                      help="Zahl der Touren und mittlere Wagenauslastung (Ladung in Teilen der Kapazität), je für reaktiv (R), Bündeln (P) und früher liefern (E). "
                           "Bündeln fährt weniger, volle Touren. Die verlorene Fehlmenge steht unter den Kennzahlen.")
    return x


def shortage_caption(live):
    """Zeile unter den Kennzahlen: verlorene Fehlmenge und Besuche je Regel (die Kennzahl-Karten sind zu schmal dafür)."""
    x = live_numbers(live)
    return (f"Fehlmenge in 120 Tagen (Einheiten, R / P / E): {F.fmt_num(x['R']['short'], 0)} / {F.fmt_num(x['P']['short'], 0)} / {F.fmt_num(x['E']['short'], 0)}; "
            f"Besuche: {x['R']['visits']} / {x['P']['visits']} / {x['E']['visits']}, davon Mitnahmen beim Bündeln: {x['P']['picks']}.")


def distribution_sentence(cell, H, gamma):
    """Verteilung der Einzelgewinne der gewählten Regel in der Messreihe (Median, Quartile, Anteil Verlust)."""
    g = R.p_gain(cell, H, gamma)
    return (f"Verteilung der Einzelgewinne von P({int(H)}; {F.fmt_level(gamma)}) über {C.MEASURED_N} Instanzen: Mittel {F.fmt_num(g['mean'], 1, True)} %, "
            f"Median {F.fmt_num(g['median'], 1, True)} %, Quartile [{F.fmt_num(g['q1'], 1, True)}; {F.fmt_num(g['q3'], 1, True)}], "
            f"in {F.fmt_num(100 * g['share_loss'], 0)} % der Instanzen ein Verlust (eine einzelne Instanz streut, die Meldung stützt sich deshalb auf die Messreihe).")


def message(live, cell, note=""):
    """Bedingte Meldung der Hauptansicht in drei Zuständen aus der VORGERECHNETEN Messreihe (nicht aus der einen Instanz). Rückgabe (Zustand, Text) mit
    Zustand 'lohnt', 'wenig' oder 'teurer'."""
    H, g = int(live["H"]), live["gamma"]
    j = R.judge(cell, H, g)
    rule = f"P({H}; {F.fmt_level(g)})"
    where = f"Messreihe ({R.cell_setting_text(cell)}; {C.MEASURED_N} Instanzen)"
    band = f"{F.fmt_num(j['mean'], 1, True)} ± {F.fmt_num(j['se'], 1)} %"
    today = f"Auf dieser einen Instanz: {F.fmt_num(live['rules']['P']['gain'], 1, True)} %."
    tail = f" {note}" if note else ""
    if j["state"] == C.STATE_LOHNT:
        return j["state"], (f"✅ Bündeln lohnt: {rule} spart laut {where} {F.fmt_num(j['mean'], 1)} ± {F.fmt_num(j['se'], 1)} % Fahrkosten gegenüber "
                            f"reaktivem Nachliefern (Median {F.fmt_num(j['median'], 1)} %, in {F.fmt_num(100 * j['share_loss'], 0)} % der Instanzen ein Verlust). {today}{tail}")
    if j["state"] == C.STATE_WENIG:
        return j["state"], (f"ℹ️ Bündeln bringt hier wenig: {rule} liegt laut {where} bei {band} gegenüber reaktivem Nachliefern "
                            f"(Median {F.fmt_num(j['median'], 1, True)} %, in {F.fmt_num(100 * j['share_loss'], 0)} % der Instanzen ein Verlust): entweder unter 5 % oder nicht "
                            f"von 0 zu unterscheiden. {today}{tail}")
    return j["state"], (f"⚠️ Diese Mitnahmeregel ist hier teurer als reaktiv: {rule} kostet laut {where} {F.fmt_num(-j['mean'], 1)} ± {F.fmt_num(j['se'], 1)} % mehr "
                        f"als reaktives Nachliefern (in {F.fmt_num(100 * j['share_loss'], 0)} % der Instanzen ein Verlust). Eine kleinere Mitnahmeschwelle γ "
                        f"oder weniger Vorschau H hilft. {today}{tail}")


def render_message(live, cell, note=""):
    state, text = message(live, cell, note)
    {C.STATE_LOHNT: st.success, C.STATE_WENIG: st.info, C.STATE_TEURER: st.warning}[state](text)
    return state


def comparison_table(live, cell):
    """Diese Instanz gegen die Messreihe (nächstliegende Zelle): Gewinn von Bündeln und früher liefern in %, mit Streuung der Messreihe."""
    H, g, L = int(live["H"]), live["gamma"], int(live["early"])
    x = live_numbers(live)
    pm, em = R.p_gain(cell, H, g), R.e_gain(cell, L)
    col_live = f"Diese Instanz (Seed {live['seed']}, 1 Instanz)"
    col_ms = f"Messreihe ({C.MEASURED_N} Instanzen)"
    rows = [
        (f"Gewinn {rule_label('P', H, g, L)} (%)", F.fmt_num(x["P"]["gain"], 1, True), F.fmt_band(pm["mean"], pm["se"])),
        (f"Median [Q1; Q3] der Einzelgewinne {rule_label('P', H, g, L)} (%)", "eine Instanz", f"{F.fmt_num(pm['median'], 1, True)} [{F.fmt_num(pm['q1'], 1, True)}; {F.fmt_num(pm['q3'], 1, True)}]"),
        (f"Gewinn {rule_label('E', H, g, L)} (%)", F.fmt_num(x["E"]["gain"], 1, True), F.fmt_band(em["mean"], em["se"])),
    ]
    return pd.DataFrame([{"Kennzahl": a, col_live: b, col_ms: c} for a, b, c in rows])


# ---------------------------------------------------------------------------------------------------
# Regeln-Tabelle
# ---------------------------------------------------------------------------------------------------
def rule_table(live, cell):
    """Die drei Regeln auf dieser Instanz: Kosten, Fahr-km, Touren, Besuche, Mitnahmen, Auslastung, Fehlmenge, Gewinn und der Gewinn der Messreihe (Mittel ±
    Standardfehler; nur für die Zeilen Bündeln und früher liefern)."""
    H, g, L = int(live["H"]), live["gamma"], int(live["early"])
    x = live_numbers(live)
    meas = {"R": "Bezug", "P": F.fmt_band(*[R.p_gain(cell, H, g)[k] for k in ("mean", "se")]), "E": F.fmt_band(*[R.e_gain(cell, L)[k] for k in ("mean", "se")])}
    rows = []
    for name in ("R", "P", "E"):
        v = x[name]
        rows.append({"Regel": rule_label(name, H, g, L), "Kosten": round(v["J"]), "Fahr-km": round(v["km"]), "Touren": v["routes"], "Besuche": v["visits"],
                     "Mitnahmen": v["picks"], "Auslastung": round(v["util"], 2), "Fehlmenge (Einheiten)": round(v["short"]),
                     "Gewinn dieser Instanz (%)": round(v["gain"], 1) if name != "R" else 0.0, "Gewinn Messreihe (%)": meas[name]})
    return pd.DataFrame(rows)


def mechanism_frame(cell):
    """Mechanismus der Standardregel P(3; 0,5) und von früher liefern E(2) gegenüber reaktiv in der Zelle (Mittel je Instanz und 120 Tage)."""
    m = R.mechanism(cell)
    rows = [("Touren", *(F.fmt_num(v) for v in m["routes"])), ("Besuche", *(F.fmt_num(v) for v in m["visits"])), ("davon Mitnahmen", "0", F.fmt_num(m["opt_visits"]), "0"),
            ("mittlere Wagenauslastung", *(F.fmt_num(v, 2) for v in m["util"])), ("Fahr-km", *(F.fmt_int(v) for v in m["km"])),
            ("Fehlmenge (Einheiten)", *(F.fmt_num(v) for v in m["short"])), ("Kunden-Tage mit Fehlmenge", *(F.fmt_num(v) for v in m["stockout_days"]))]
    return pd.DataFrame([{"Kennzahl (je 120 Tage)": a, "Reaktiv": b, "Bündeln P(3; 0,5)": c, "Früher liefern E(2)": d} for a, b, c, d in rows])


# ---------------------------------------------------------------------------------------------------
# Tagesansicht
# ---------------------------------------------------------------------------------------------------
def halteliste_frame(view):
    """Halteliste aller Touren eines Tages: Stopps in Fahrtreihenfolge mit Menge, Bestand vor der Lieferung, Reichweite und Art (fällig / Mitnahme)."""
    rows = []
    for tour in view["tours"]:
        for i, s in enumerate(tour["stops"], 1):
            rows.append({"Tour": tour["tour"], "Nr.": i, "Kunde": s["customer"], "Art": s["kind"], "Bestand vorher": round(s["fill_before"]),
                         "Füllstand vorher": F.fmt_share(s["fill_share"]), "Reichweite (Tage)": round(s["reach"], 1), "Liefermenge": round(s["qty"])})
    return pd.DataFrame(rows, columns=["Tour", "Nr.", "Kunde", "Art", "Bestand vorher", "Füllstand vorher", "Reichweite (Tage)", "Liefermenge"])


def render_day(prefix, live, day, right_rule):
    """Karten von reaktiv und der rechten Regel (Bündeln oder früher liefern) nebeneinander, darunter die Haltelisten und der Verlauf über die Tage."""
    H, g, L = live["H"], live["gamma"], live["early"]
    views = {r: LV.day_view(live, r, day) for r in ("R", right_rule)}
    cols = st.columns(2)
    for col, rule in zip(cols, ("R", right_rule)):
        v = views[rule]
        with col:
            st.markdown(f"**{rule_label(rule, H, g, L)}** – Tag {day}")
            st.plotly_chart(V.day_map_figure(live, rule, day), width="stretch", key=f"{prefix}_map_{rule}")
            if v["tours"]:
                pick = f", davon {v['n_pickups']} Mitnahme{'n' if v['n_pickups'] != 1 else ''}" if rule == "P" else ""
                st.caption(f"{len(v['tours'])} Tour{'en' if len(v['tours']) != 1 else ''}, {v['n_stops']} Stopp{'s' if v['n_stops'] != 1 else ''}{pick}, {F.fmt_num(v['km'], 0)} km.")
            else:
                st.caption("An diesem Tag fährt diese Regel keine Tour.")
    tabs = st.tabs([f"Halteliste {rule_label(r, H, g, L)}" for r in ("R", right_rule)])
    for tab, rule in zip(tabs, ("R", right_rule)):
        with tab:
            df = halteliste_frame(views[rule])
            if len(df):
                st.dataframe(df, width="stretch", hide_index=True)
            else:
                st.caption("Keine Tour an diesem Tag.")
    return views


# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ②: Tabellen (vorgerechnet)
# ---------------------------------------------------------------------------------------------------
def capacity_frame(rows, marked_q):
    out = []
    for r in rows:
        out.append({"Wagenkapazität Q": f"{r['Q']}" + ("  ◀ eingestellt" if r["Q"] == marked_q else ""), "Q/q (Lieferungen je Wagen)": F.fmt_num(r["Q_over_q"]),
                    "Gewinn Bündeln (Mittel ± SE, %)": F.fmt_band(r["gain"], r["se"]), "beste Zelle, kreuzvalidiert (%)": F.fmt_num(r["cv_gain"], 1, True),
                    "Touren R → P (Standardregel)": f"{F.fmt_num(r['routes_R'])} → {F.fmt_num(r['routes_P'])}",
                    "Auslastung R → P": f"{F.fmt_num(r['util_R'], 2)} → {F.fmt_num(r['util_P'], 2)}"})
    return pd.DataFrame(out)


def uncertainty_frame(rows, name, fmt=lambda v: str(v)):
    return pd.DataFrame([{name: fmt(r["level"]), "Gewinn Bündeln (Mittel ± SE, %)": F.fmt_band(r["gain"], r["se"])} for r in rows])


def rule_rows_frame(rows):
    """Regeln im Vergleich (aus irp_results.rule_rows): Gewinn gegenüber reaktiv."""
    return pd.DataFrame([{"Regel": lab, "Gewinn gegenüber reaktiv (%)": "Bezug" if (se == 0.0 and mean == 0.0) else (F.fmt_band(mean, se) if se is not None else F.fmt_num(mean, 1, True))}
                         for lab, mean, se in rows])


def shortage_frame(rows):
    """Knappe Flotte: Gewinn, Fehlmenge vorher und nachher, Anteil der Instanzen mit mehr Fehlmenge beim Bündeln."""
    return pd.DataFrame([{"Zelle": r["label"], "Gewinn Standardregel (%)": F.fmt_band(r["gain"], r["se"]),
                          "Fehlmenge reaktiv → Bündeln (Einheiten)": f"{F.fmt_num(r['short_R'], 0)} → {F.fmt_num(r['short_P'], 0)}",
                          "Instanzen mit mehr Fehlmenge": F.fmt_num(100 * r["cmp"]["more"], 1) + " %"} for r in rows])


def regime_frame(rows, highlight_name=None):
    """Regime-Tabelle: alle Zellen mit Gewinn der gewählten Regel (Mittel ± SE), Median, Quartile, Anteil Verlust und Urteil."""
    out = []
    for r in rows:
        out.append({"Gruppe": r["group"], "Zelle": ("▶ " if r["name"] == highlight_name else "") + r["label"], "Q/q": F.fmt_num(r["Q_over_q"]),
                    "Gewinn Bündeln (Mittel ± SE, %)": F.fmt_band(r["gain"], r["se"]), "Median [Q1; Q3]": f"{F.fmt_num(r['median'], 1, True)} [{F.fmt_num(r['q1'], 1, True)}; {F.fmt_num(r['q3'], 1, True)}]",
                    "Anteil Verlust": F.fmt_share(r["share_loss"]), "Urteil": R.STATE_TEXT[r["state"]]})
    return pd.DataFrame(out)


def negative_frame(data, limit=12):
    """Die schlimmsten Zellen des Gitters (Mittel am stärksten unter 0): Zelle, Regel, Gewinn."""
    rows = []
    for c, key, mean, se in R.negative_pairs(data)[:limit]:
        H, g = R.parse_p_key(key)
        rows.append({"Zelle": R.cell_label(c), "Regel": f"P({H}; {F.fmt_level(g)})", "Gewinn (Mittel ± SE, %)": F.fmt_band(mean, se)})
    return pd.DataFrame(rows)


def oracle_frame(summaries):
    """Exakter Maßstab (vorgerechnet): Lücke zum bewiesenen Optimum je Regel, Anteil der reaktiven Lücke, den P(3; 0,5) schließt, Touren in 6 Tagen."""
    rows = []
    for wagon, s in summaries.items():
        row = {"Wagen": wagon, "Instanzen (bewiesen optimal)": f"{s['n']} ({s['n_optimal']})"}
        for k, label in R.ORACLE_POLICIES:
            row[f"Lücke {label} (%)"] = F.fmt_band(s["gap"][k]["mean"], s["gap"][k]["se"], 1, False)
        row["P(3; 0,5) schließt Anteil der reaktiven Lücke"] = F.fmt_share(s["closed_P3"])
        row["Touren in 6 Tagen: Optimum / R / P(3; 0,5)"] = f"{F.fmt_num(s['routes']['optimum'], 2)} / {F.fmt_num(s['routes']['R'], 2)} / {F.fmt_num(s['routes']['P3'], 2)}"
        rows.append(row)
    return pd.DataFrame(rows)


def oracle_instances_frame(data, wagon):
    """Die 30 vorgerechneten Kleininstanzen eines Wagens: Optimum, reaktiv, Bündeln, Lücken in %."""
    rows = []
    for r in R.oracle_rows(data, wagon):
        jo = r["J_oracle"]
        rows.append({"Seed": r["seed"], "Optimum": F.fmt_num(jo), "reaktiv": F.fmt_num(r["R"]), "P(3; 0,5)": F.fmt_num(r["P_H3_g0.50"]), "P(8; 0,25)": F.fmt_num(r["P_H8_g0.25"]),
                     "Lücke reaktiv (%)": F.fmt_num(100 * (r["R"] - jo) / jo), "Lücke P(3; 0,5) (%)": F.fmt_num(100 * (r["P_H3_g0.50"] - jo) / jo),
                     "Touren Optimum / R / P": f"{r['oracle_routes']} / {r['R_routes']} / {r['P_H3_g0.50_routes']}", "Status": r["status"]})
    return pd.DataFrame(rows)


def small_plan_frame(plans):
    """Plan der Kleininstanz: je Tag die Tour (Kunden in Reihenfolge mit Menge)."""
    rows = []
    for t, plan in enumerate(plans, 1):
        for route in plan:
            rows.append({"Tag": t, "Tour": " → ".join(f"{c} ({q:.0f})" for c, q in route)})
    return pd.DataFrame(rows, columns=["Tag", "Tour"])


def compare_small_frame(cmp_):
    """Optimum, reaktiv, Bündeln und früher liefern auf der Kleininstanz: Kosten, Lücke, Touren, Fahr-km, Fehlmenge."""
    rows = [{"Plan": "Optimum (CP-SAT)", "Kosten J": F.fmt_num(cmp_["optimum"]["J"]), "Lücke zum Optimum (%)": "0,0", "Touren": cmp_["optimum"]["routes"],
             "Fahr-km": F.fmt_num(cmp_["optimum"]["routing"]), "Fehlmenge": F.fmt_num(cmp_["optimum"]["short"], 0)}]
    for key, label in (("R", "Reaktiv"), ("P", "Bündeln"), ("E", "Früher liefern")):
        v = cmp_[key]
        rows.append({"Plan": label, "Kosten J": F.fmt_num(v["J"]), "Lücke zum Optimum (%)": F.fmt_num(v["gap"], 1, True), "Touren": v["routes"], "Fahr-km": F.fmt_num(v["routing"]),
                     "Fehlmenge": F.fmt_num(v["short"], 0)})
    return pd.DataFrame(rows)


def day_frame(live):
    return pd.DataFrame(LV.day_table(live))
