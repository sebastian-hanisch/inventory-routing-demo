"""Plotly-Figuren der Inventory-Routing-Demo: Tageskarte (Touren, Füllstände, Mitnehmer), Verlauf über die Tage, Kapazitätshebel, Heatmap Vorschau x
Mitnahmeschwelle, Gewinn über Stufen, Regeln im Vergleich, Regime-Balken, exakter Maßstab und Plan der Kleininstanz.

Konventionen des Portfolios: Achsen `fixedrange` (Touch-Scrollen), Vorlage plotly_white, Legende unten, Farben über alle Figuren konsistent, KEINE
Farblisten in Marker-Eigenschaften (je Merkmal eine eigene Spur; numerische Werte mit Farbskala sind erlaubt und erzeugen keine Plotly-Warnungen).
Plotly wird erst in den Funktionen importiert, damit die reine Rechnung ohne Plotly testbar bleibt."""
import irp_constants as C
import irp_format as F
import irp_live as LV
import irp_results as R

LEGEND_BOTTOM = dict(orientation="h", yref="container", yanchor="bottom", y=0.0, x=0)
FILL_SCALE = [[0.0, "#c0392b"], [0.5, "#e0a800"], [1.0, "#2e7d4f"]]      # Füllstand: leer rot, halb gelb, voll grün


def _lock_axes(fig):
    """Achsen fest: verhindert Zoomen und Verschieben per Touch, damit die Seite scrollbar bleibt (Hover bleibt)."""
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


# ---------------------------------------------------------------------------------------------------
# Tagesansicht
# ---------------------------------------------------------------------------------------------------
def day_map_figure(live, rule, day):
    """Depot, Kunden mit Füllstand als Farbe (Tagesanfang), Touren der Regel an diesem Tag; Kunden auf einer Tour mit Ring in der Tourfarbe, mitgenommene
    Kunden (nur Bündeln) als Raute."""
    import plotly.graph_objects as go

    view = LV.day_view(live, rule, day)
    xy, cap, inv = live["xy"], live["C"], view["I"]
    n = live["N"]
    size = float(xy[:, 0].max()) if xy.shape[0] else 100.0
    fig = go.Figure()
    for tour in view["tours"]:
        color = C.TOUR_COLORS[(tour["tour"] - 1) % len(C.TOUR_COLORS)]
        pts = [0] + [s["customer"] for s in tour["stops"]] + [0]
        fig.add_trace(go.Scatter(x=[float(xy[p, 0]) for p in pts], y=[float(xy[p, 1]) for p in pts], mode="lines", line=dict(color=color, width=2.5),
                                 hoverinfo="skip", showlegend=False))
        ring = [s for s in tour["stops"] if s["kind"] == "fällig"]
        if ring:
            fig.add_trace(go.Scatter(x=[float(xy[s["customer"], 0]) for s in ring], y=[float(xy[s["customer"], 1]) for s in ring], mode="markers",
                                     marker=dict(symbol="circle-open", size=19, color=color, line=dict(width=3, color=color)), hoverinfo="skip", showlegend=False))
    share = [float(inv[i] / cap[i]) for i in range(1, n + 1)]
    fig.add_trace(go.Scatter(
        x=[float(xy[i, 0]) for i in range(1, n + 1)], y=[float(xy[i, 1]) for i in range(1, n + 1)], mode="markers+text", name="Kunde (Füllstand als Farbe)",
        text=[str(i) for i in range(1, n + 1)], textposition="top center", textfont=dict(size=9),
        marker=dict(size=11, color=share, colorscale=FILL_SCALE, cmin=0.0, cmax=1.0, line=dict(color="white", width=1),
                    colorbar=dict(title=dict(text="Füllstand", side="right"), thickness=9, len=0.7, tickvals=[0, 0.5, 1], ticktext=["0 %", "50 %", "100 %"])),
        customdata=[[float(inv[i]), float(cap[i]), float(inv[i] / live["mu"][i])] for i in range(1, n + 1)],
        hovertemplate="Kunde %{text}: Bestand %{customdata[0]:.0f} von %{customdata[1]:.0f}, Reichweite %{customdata[2]:.1f} Tage<extra></extra>"))
    picks = [s for t in view["tours"] for s in t["stops"] if s["kind"] == "Mitnahme"]
    if picks:
        fig.add_trace(go.Scatter(x=[float(xy[s["customer"], 0]) for s in picks], y=[float(xy[s["customer"], 1]) for s in picks], mode="markers",
                                 name="mitgenommen (nicht fällig)", marker=dict(symbol="diamond-open", size=20, color=C.COLOR_P, line=dict(width=3, color=C.COLOR_P)),
                                 text=[f"Kunde {s['customer']}: nicht fällig, Reichweite {s['reach']:.1f} Tage, mitgenommen" for s in picks],
                                 hovertemplate="%{text}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[float(xy[0, 0])], y=[float(xy[0, 1])], mode="markers", name="Depot",
                             marker=dict(symbol="square", size=13, color="#1c2430", line=dict(color="white", width=1)), hovertemplate="Depot<extra></extra>"))
    lo = float(min(xy[:, 0].min(), xy[:, 1].min()))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT + 60, margin=dict(t=10, b=110, l=10, r=10), legend=LEGEND_BOTTOM,
                      xaxis=dict(range=[min(lo, 0.0) - 4, size + 4], title="km", constrain="domain"),
                      yaxis=dict(range=[min(lo, 0.0) - 4, size + 4], scaleanchor="x", scaleratio=1, title="km", constrain="domain"))
    return _lock_axes(fig)


def trajectory_figure(live, day):
    """Oben die kumulierten Fahr-km je Regel über die Tage, unten der mittlere Füllstand der Kunden am Tagesanfang; senkrechte Linie = gezeigter Tag."""
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.5, 0.5], vertical_spacing=0.08)
    days = list(range(1, live["days"] + 1))
    for name in ("R", "P", "E"):
        rl = live["rules"][name]
        km = np.cumsum(rl["res"]["day_route_km"])
        label = {"R": "Reaktiv", "P": f"Bündeln P({int(live['H'])}; {F.fmt_level(live['gamma'])})", "E": f"Früher liefern E({int(live['early'])})"}[name]
        fig.add_trace(go.Scatter(x=days, y=[float(v) for v in km], mode="lines", name=label, line=dict(color=C.RULE_COLORS[name], width=2),
                                 hovertemplate=f"{label}: %{{y:.0f}} km bis Tag %{{x}}<extra></extra>"), row=1, col=1)
        fill = rl["I"][:live["days"], 1:] / live["C"][1:]
        fig.add_trace(go.Scatter(x=days, y=[float(v) for v in fill.mean(axis=1)], mode="lines", name=label, showlegend=False,
                                 line=dict(color=C.RULE_COLORS[name], width=2), hovertemplate=f"{label}: mittlerer Füllstand %{{y:.0%}} an Tag %{{x}}<extra></extra>"),
                      row=2, col=1)
    fig.add_vline(x=day, line=dict(color="#4b5d75", width=1, dash="dot"), row="all", col=1)
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT + 40, margin=dict(t=10, b=110, l=10, r=10), legend=LEGEND_BOTTOM)
    fig.update_yaxes(title_text="Fahr-km (kumuliert)", rangemode="tozero", row=1, col=1)
    fig.update_yaxes(title_text="mittlerer Füllstand", tickformat=".0%", range=[0, 1], row=2, col=1)
    fig.update_xaxes(title_text="Tag", range=[1, live["days"]], row=2, col=1)
    return _lock_axes(fig)


# ---------------------------------------------------------------------------------------------------
# Kernabschnitt ②: vorgerechnet
# ---------------------------------------------------------------------------------------------------
def capacity_figure(rows, marked_q, H, gamma):
    """Gewinn über die Wagenkapazität: gewählte Regel P(H; gamma) (Mittel ± Standardfehler) und beste Zelle des Gitters (kreuzvalidiert)."""
    import plotly.graph_objects as go

    labels = [f"Q = {r['Q']}<br>Q/q = {F.fmt_num(r['Q_over_q'])}" + ("<br>◀ eingestellt" if r["Q"] == marked_q else "") for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=[r["gain"] for r in rows], name=f"Bündeln P({H}; {F.fmt_level(gamma)})", marker_color=C.COLOR_P,
                         error_y=dict(type="data", array=[r["se"] for r in rows], visible=True, color="#4b5d75", thickness=1.2),
                         hovertemplate="%{y:+.1f} % gegenüber reaktiv<extra></extra>"))
    fig.add_trace(go.Bar(x=labels, y=[r["cv_gain"] for r in rows], name="beste Zelle des Gitters (kreuzvalidiert)", marker_color=C.COLOR_LIGHT,
                         hovertemplate="%{y:+.1f} % gegenüber reaktiv<extra></extra>"))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT, barmode="group", margin=dict(t=20, b=110, l=10, r=10), legend=LEGEND_BOTTOM,
                      yaxis=dict(title="Gewinn gegenüber reaktiv (%)"))
    return _lock_axes(fig)


def heatmap_figure(cell, H, gamma):
    """Gewinn (%) über Vorschau H (Spalten) und Mitnahmeschwelle gamma (Zeilen) in der Zelle; die gewählte Kombination ist umrandet."""
    import plotly.graph_objects as go

    z = R.heat_matrix(cell)
    fig = go.Figure(go.Heatmap(z=z, x=list(range(len(C.HORIZON_OPTIONS))), y=list(range(len(C.GAMMA_OPTIONS))), colorscale="RdYlGn", zmid=0,
                               texttemplate="%{z:.1f}", showscale=False, hovertemplate="Gewinn %{z:+.1f} %<extra></extra>"))
    j, i = C.HORIZON_OPTIONS.index(int(H)), C.GAMMA_OPTIONS.index(float(gamma))
    fig.add_shape(type="rect", x0=j - 0.5, x1=j + 0.5, y0=i - 0.5, y1=i + 0.5, line=dict(color="#1c2430", width=3))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 60, margin=dict(t=20, b=50, l=10, r=10),
                      xaxis=dict(tickvals=list(range(len(C.HORIZON_OPTIONS))), ticktext=[f"H = {h}" for h in C.HORIZON_OPTIONS], title="Vorschau H (Tage)"),
                      yaxis=dict(tickvals=list(range(len(C.GAMMA_OPTIONS))), ticktext=[f"γ = {F.fmt_level(g)}" for g in C.GAMMA_OPTIONS], autorange="reversed"))
    return _lock_axes(fig)


def level_figure(rows, xlabel, marked, fmt=lambda v: str(v)):
    """Gewinn der gewählten Regel über die Stufen eines Parameters (Verbrauchsschwankung oder Strafe); die eingestellte Stufe ist hervorgehoben (eigene Spur)."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for highlight, name, color in ((False, "gemessene Stufe", C.COLOR_LIGHT), (True, "eingestellt", C.COLOR_P)):
        sel = [r for r in rows if (r["level"] == marked) == highlight]
        if not sel:
            continue
        fig.add_trace(go.Bar(x=[fmt(r["level"]) for r in sel], y=[r["gain"] for r in sel], name=name, marker_color=color,
                             error_y=dict(type="data", array=[r["se"] for r in sel], visible=True, color="#4b5d75", thickness=1.2),
                             hovertemplate="%{y:+.1f} % gegenüber reaktiv<extra></extra>"))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 80, barmode="overlay", margin=dict(t=20, b=100, l=10, r=10), legend=LEGEND_BOTTOM,
                      xaxis=dict(title=xlabel, type="category"), yaxis=dict(title="Gewinn (%)", rangemode="tozero"))
    return _lock_axes(fig)


def rules_figure(rows):
    """Regeln im Vergleich (waagerechte Balken, Mittel ± Standardfehler): billiger als reaktiv grün, teurer rot, Bezug grau. rows aus irp_results.rule_rows."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for name, color, test in (("billiger als reaktiv", C.COLOR_GOOD, lambda v: v > 0), ("teurer als reaktiv", C.COLOR_BAD, lambda v: v < 0),
                              ("Bezug: reaktiv", C.COLOR_NEUTRAL, lambda v: v == 0)):
        sel = [r for r in rows if test(r[1])]
        if not sel:
            continue
        kw = {}
        if any(r[2] for r in sel):
            kw["error_x"] = dict(type="data", array=[r[2] or 0.0 for r in sel], visible=True, color="#4b5d75", thickness=1.2)
        fig.add_trace(go.Bar(y=[r[0] for r in sel], x=[r[1] for r in sel], orientation="h", name=name, marker_color=color,
                             hovertemplate="%{y}: %{x:+.1f} % gegenüber reaktiv<extra></extra>", **kw))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 40, barmode="overlay", margin=dict(t=20, b=100, l=10, r=10), legend=LEGEND_BOTTOM,
                      xaxis_title="Gewinn gegenüber reaktiv (%)")
    return _lock_axes(fig)


def short_label(label, limit=30):
    text = label.split(" (")[0]
    return text if len(text) <= limit else text[:limit - 1] + "…"


def regime_figure(rows, marked_name):
    """Gewinn der gewählten Regel je gemessener Zelle (Mittel ± Standardfehler), waagerechte Balken; die Zelle der Einstellung ist hervorgehoben."""
    import plotly.graph_objects as go

    fig = go.Figure()
    for highlight, name, color in ((False, "gemessene Zelle", C.COLOR_LIGHT), (True, "Zelle Ihrer Einstellung", C.COLOR_P)):
        sel = [r for r in rows if (r["name"] == marked_name) == highlight]
        if not sel:
            continue
        fig.add_trace(go.Bar(y=[r["label"] for r in sel], x=[r["gain"] for r in sel], orientation="h", name=name, marker_color=color,
                             error_x=dict(type="data", array=[r["se"] for r in sel], visible=True, color="#4b5d75", thickness=1.2),
                             hovertemplate="%{y}: %{x:+.1f} % gegenüber reaktiv<extra></extra>"))
    labels = [r["label"] for r in rows]
    fig.update_layout(template="plotly_white", height=max(C.CHART_HEIGHT, 24 * len(rows) + 130), barmode="overlay", bargap=0.25, margin=dict(t=20, b=100, l=10, r=10),
                      legend=LEGEND_BOTTOM, xaxis_title="Gewinn gegenüber reaktiv (%)")
    fig.update_yaxes(autorange="reversed", tickmode="array", tickvals=labels, ticktext=[short_label(t) for t in labels], tickfont=dict(size=10))
    return _lock_axes(fig)


def oracle_figure(summaries):
    """Lücke zum bewiesenen Optimum je Regel (Mittel ± Standardfehler), je Wagen eine Gruppe; summaries: {Wagen: irp_results.oracle_summary}."""
    import plotly.graph_objects as go

    fig = go.Figure()
    colors = {150: C.COLOR_P, 250: C.COLOR_LIGHT}
    for wagon, s in summaries.items():
        fig.add_trace(go.Bar(x=[label for _, label in R.ORACLE_POLICIES], y=[s["gap"][k]["mean"] for k, _ in R.ORACLE_POLICIES], name=f"Wagen {wagon}",
                             marker_color=colors.get(wagon, C.COLOR_NEUTRAL),
                             error_y=dict(type="data", array=[s["gap"][k]["se"] for k, _ in R.ORACLE_POLICIES], visible=True, color="#4b5d75", thickness=1.2),
                             hovertemplate="%{x}: %{y:.1f} % über dem Optimum<extra></extra>"))
    fig.update_layout(template="plotly_white", height=C.CHART_HEIGHT - 40, barmode="group", margin=dict(t=20, b=100, l=10, r=10), legend=LEGEND_BOTTOM,
                      yaxis=dict(title="Lücke zum Optimum (%)", rangemode="tozero"))
    return _lock_axes(fig)


def plan_matrix_figure(plans_by_rule, n_customers, n_days):
    """Besuche der Kleininstanz je Kunde und Tag (Punktgröße = Liefermenge), ein Teilbild je Plan; plans_by_rule: [(Beschriftung, Farbe, plans)] mit plans =
    Liste je Tag von Touren [(Kunde, Menge)...]."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=len(plans_by_rule), cols=1, shared_xaxes=True, vertical_spacing=0.07, subplot_titles=[p[0] for p in plans_by_rule])
    for k, (label, color, plans) in enumerate(plans_by_rule, 1):
        pts = [(t + 1, c, q) for t, plan in enumerate(plans) for route in plan for c, q in route]
        fig.add_trace(go.Scatter(x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers", name=label, showlegend=False,
                                 marker=dict(size=[max(7.0, min(24.0, 5.0 + p[2] / 8.0)) for p in pts], color=color, opacity=0.85, line=dict(color="white", width=1)),
                                 text=[f"Tag {p[0]}, Kunde {p[1]}, Menge {p[2]:.0f}" for p in pts], hovertemplate="%{text}<extra></extra>"), row=k, col=1)
        fig.update_yaxes(range=[0.4, n_customers + 0.6], dtick=1, row=k, col=1, title_text="Kunde")
    fig.update_xaxes(range=[0.4, n_days + 0.6], dtick=1, title_text="Tag", row=len(plans_by_rule), col=1)
    fig.update_layout(template="plotly_white", height=150 * len(plans_by_rule) + 90, margin=dict(t=30, b=50, l=10, r=10))
    return _lock_axes(fig)
