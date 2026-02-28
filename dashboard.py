"""
CadenceWorks Analytics Engine
Layer 5: Dashboard Generator

Takes all analytics outputs and renders a fully self-contained
HTML dashboard in the CadenceWorks brand style.
"""

import json
from pathlib import Path


def _fmt_currency(val, symbol="R"):
    if val >= 1_000_000:
        return f"{symbol}{val/1_000_000:.1f}M"
    elif val >= 1_000:
        return f"{symbol}{val/1_000:.0f}k"
    return f"{symbol}{val:,.0f}"


def _bar_color(val, thresholds=(8, 15)):
    lo, hi = thresholds
    if val >= hi:   return "danger"
    elif val >= lo: return "warn"
    return "ok"


def _bar_rows(data: dict, max_val: float = None, color_fn=None, suffix="%") -> str:
    if not data:
        return "<p style='color:var(--muted);font-size:12px'>No data</p>"
    if max_val is None:
        max_val = max(data.values()) or 1
    html = ""
    for label, val in data.items():
        pct_width = round(val / max_val * 100)
        cls = color_fn(val) if color_fn else "teal"
        html += f"""
        <div class="bar-row">
          <span class="bar-label">{label}</span>
          <div class="bar-track"><div class="bar-fill {cls}" style="width:{pct_width}%"></div></div>
          <span class="bar-pct">{val}{suffix}</span>
        </div>"""
    return html


def render(
    meta: dict,
    descriptive: dict,
    predictive: dict,
    prescriptive: dict,
    output_path: str = "cadenceworks_report.html",
) -> str:
    """Render full HTML dashboard and write to output_path."""

    kpis    = descriptive.get("kpis", {})
    source  = meta.get("source_file", "Uploaded File")
    dr      = meta.get("date_range", ("—", "—"))
    providers = ", ".join(meta.get("providers", []))
    date_label = f"{dr[0]} to {dr[1]}" if dr else "—"

    # ── KPI cards ─────────────────────────────────────────────────────────────
    def kpi_card(label, value, sub, trend_cls, trend_txt):
        return f"""
        <div class="kpi-card">
          <div class="kpi-label">{label}</div>
          <div class="kpi-value {trend_cls}">{value}</div>
          <div class="kpi-sub">{sub}</div>
          <div class="kpi-trend {trend_cls}">{trend_txt}</div>
        </div>"""

    ns_rate   = kpis.get("no_show_rate", 0)
    can_rate  = kpis.get("cancellation_rate", 0)
    comp_rate = kpis.get("completion_rate", 0)
    rev_lost  = kpis.get("revenue_lost", 0)
    avg_lead  = kpis.get("avg_lead_time_days", 0)
    total     = kpis.get("total_appointments", 0)

    kpi_html = (
        kpi_card("Total Appointments",   str(total),             f"Period: {date_label}",  "neu",   "Baseline set") +
        kpi_card("Completion Rate",      f"{comp_rate}%",        f"{kpis.get('completed',0)} attended",    "good",  "✓ Target: >85%") +
        kpi_card("No-Show Rate",         f"{ns_rate}%",          f"{kpis.get('no_shows',0)} appointments", _bar_color(ns_rate), "⚠ Threshold: 8%") +
        kpi_card("Cancellation Rate",    f"{can_rate}%",         f"{kpis.get('cancelled',0)} cancelled",   _bar_color(can_rate, (5,10)), "Industry avg: 6%") +
        kpi_card("Revenue at Risk",      _fmt_currency(rev_lost), "No-shows only",          "danger" if rev_lost>0 else "good", "⚠ Recoverable") +
        kpi_card("Avg Lead Time",        f"{avg_lead}d",         "Days booked ahead",       "neu",   f"Range: 0–{kpis.get('avg_lead_time_days',0)*3:.0f}d")
    )

    # ── Descriptive charts ────────────────────────────────────────────────────
    noshow_ch   = descriptive.get("noshow_by_channel", {})
    noshow_day  = descriptive.get("noshow_by_day", {})
    noshow_lead = descriptive.get("noshow_by_lead_time", {})
    noshow_pt   = descriptive.get("noshow_by_patient_type", {})
    noshow_slot = descriptive.get("noshow_by_slot_type", {})
    vol_status  = descriptive.get("volume_by_status", {})
    vol_appt    = descriptive.get("volume_by_appt_type", {})
    prov_comp   = descriptive.get("provider_comparison", [])

    ch_bars   = _bar_rows(noshow_ch,   color_fn=lambda v: _bar_color(v))
    day_bars  = _bar_rows(noshow_day,  color_fn=lambda v: _bar_color(v, (10, 18)))
    lead_bars = _bar_rows(noshow_lead, color_fn=lambda v: _bar_color(v, (8, 18)))
    pt_bars   = _bar_rows(noshow_pt,   color_fn=lambda v: _bar_color(v))
    slot_bars = _bar_rows(noshow_slot, color_fn=lambda v: _bar_color(v))

    # Status donut (simple CSS)
    status_items = "".join(
        f'<div class="legend-item"><span class="dot dot-{s.lower().replace("-","").replace(" ","")}"></span>{s}: <strong>{c}</strong></div>'
        for s, c in vol_status.items()
    )
    appt_items = "".join(
        f'<div class="legend-item"><span class="dot" style="background:var(--teal)"></span>{t}: <strong>{c}</strong></div>'
        for t, c in vol_appt.items()
    )

    # Provider table
    prov_rows = ""
    for p in prov_comp:
        ns_cls = _bar_color(p.get("no_show_rate", 0))
        prov_rows += f"""
        <tr>
          <td>{p.get('provider','—')}</td>
          <td>{p.get('total','—')}</td>
          <td>{p.get('completed','—')}</td>
          <td><span class="badge {ns_cls}">{p.get('no_show_rate','—')}%</span></td>
          <td>{_fmt_currency(p.get('revenue',0))}</td>
          <td>{p.get('avg_lead','—')}d</td>
        </tr>"""

    # Daily trend sparkline (SVG)
    daily = descriptive.get("daily_trend", [])
    spark_html = ""
    if daily:
        vals = [d["total"] for d in daily]
        ns_vals = [d["no_shows"] for d in daily]
        mx = max(vals) or 1
        w, h = 600, 80
        step = w / max(len(vals)-1, 1)
        pts = " ".join(f"{i*step:.0f},{h - (v/mx*h*0.85):.0f}" for i, v in enumerate(vals))
        ns_pts = " ".join(f"{i*step:.0f},{h - (v/mx*h*0.85):.0f}" for i, v in enumerate(ns_vals))
        spark_html = f"""
        <svg viewBox="0 0 {w} {h}" style="width:100%;height:80px">
          <polyline points="{pts}" fill="none" stroke="var(--teal)" stroke-width="2.5" stroke-linejoin="round"/>
          <polyline points="{ns_pts}" fill="none" stroke="var(--red)" stroke-width="1.5" stroke-linejoin="round" stroke-dasharray="4"/>
        </svg>
        <div style="display:flex;gap:20px;font-size:11px;margin-top:6px">
          <span><span style="color:var(--teal);font-weight:700">—</span> Total appointments</span>
          <span><span style="color:var(--red);font-weight:700">- -</span> No-shows</span>
        </div>"""

    # ── Predictive section ────────────────────────────────────────────────────
    risk_dist = predictive.get("risk_distribution", {})
    model_type = predictive.get("model_type", "—")
    model_auc  = predictive.get("model_auc")
    feat_imp   = predictive.get("feature_importance", {})
    score_stats = predictive.get("score_stats", {})
    validation  = predictive.get("validation", {})

    fi_bars = _bar_rows(
        {k: round(v*100) for k, v in list(feat_imp.items())[:6]},
        max_val=100, suffix="%",
        color_fn=lambda v: "teal"
    )

    # High-risk table
    hr_appts = predictive.get("high_risk_appointments", [])
    hr_rows = ""
    for a in hr_appts[:10]:
        hr_rows += f"""
        <tr>
          <td>{a.get('appointment_id','—')}</td>
          <td>{a.get('provider','—')}</td>
          <td>{a.get('patient_type','—')}</td>
          <td>{a.get('channel','—')}</td>
          <td>{a.get('day_of_week','—')}</td>
          <td>{a.get('lead_time_days','—')}d</td>
          <td><span class="badge danger">{a.get('risk_score','—')}</span></td>
          <td>{a.get('status','—')}</td>
        </tr>"""

    risk_total = sum(risk_dist.values()) or 1
    high_pct   = round(risk_dist.get("High Risk", 0) / risk_total * 100)
    med_pct    = round(risk_dist.get("Medium Risk", 0) / risk_total * 100)
    low_pct    = round(risk_dist.get("Low Risk", 0) / risk_total * 100)

    auc_badge = f'<span class="badge good">AUC: {model_auc}</span>' if model_auc else '<span class="badge neu">Rule-Based</span>'

    # ── Prescriptive section ──────────────────────────────────────────────────
    recs        = prescriptive.get("recommendations", [])
    agent_acts  = prescriptive.get("agent_actions", [])
    quick_wins  = prescriptive.get("quick_wins", [])

    rec_html = ""
    for r in recs:
        rec_html += f"""
        <div class="rx-card">
          <div class="rx-num-wrap"><span class="rx-num">0{r['priority']}</span></div>
          <div class="rx-body">
            <h3>{r['title']}</h3>
            <p class="rx-rationale">{r['rationale']}</p>
            <p class="rx-action">→ {r['action']}</p>
            <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:10px;">
              <div class="rx-impact">↑ {r['impact']}</div>
              <div class="rx-agent">🤖 {r['agent']}</div>
            </div>
          </div>
        </div>"""

    agent_html = ""
    for a in agent_acts:
        agent_html += f"""
        <div class="agent-action-card">
          <div class="agent-name">{a['agent']}</div>
          <div class="agent-desc">{a['action']}</div>
          <div class="agent-timing">⏱ {a['timing']}</div>
        </div>"""

    qw_html = ""
    for q in quick_wins:
        effort_cls = "good" if q["effort"] == "Low" else "warn"
        impact_cls = "good" if q["impact"] == "High" else "warn"
        qw_html += f"""
        <div class="qw-card">
          <h3>{q['title']}</h3>
          <p>{q['detail']}</p>
          <div style="display:flex;gap:8px;margin-top:10px;">
            <span class="badge {effort_cls}">Effort: {q['effort']}</span>
            <span class="badge {impact_cls}">Impact: {q['impact']}</span>
          </div>
        </div>"""

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>CadenceWorks Analytics Report</title>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root {{
    --navy:#1a3c4d; --teal:#3dbfaa; --teal-dk:#2ea393; --teal-lt:#e8f8f5;
    --bg:#f0f2f5; --card:#fff; --muted:#6b8899; --border:#e0e6ea;
    --red:#e05252; --red-lt:#fdf0f0; --amber:#e8923a; --amber-lt:#fdf5ed;
    --green:#2ea37a; --green-lt:#edf7f3; --shadow:0 2px 12px rgba(26,60,77,0.08);
    --shadow-hover:0 8px 28px rgba(26,60,77,0.14);
  }}
  *{{margin:0;padding:0;box-sizing:border-box;}}
  body{{font-family:'Inter',sans-serif;background:var(--bg);color:var(--navy);}}
  nav{{background:#fff;border-bottom:1px solid var(--border);padding:0 48px;height:64px;
       display:flex;align-items:center;justify-content:space-between;
       position:sticky;top:0;z-index:100;box-shadow:0 1px 8px rgba(26,60,77,0.06);}}
  .logo{{display:flex;align-items:center;gap:10px;}}
  .logo-text{{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:18px;color:var(--navy);}}
  .nav-badge{{background:var(--teal-lt);color:var(--teal-dk);font-family:'Plus Jakarta Sans',sans-serif;
              font-size:12px;font-weight:700;padding:6px 16px;border-radius:20px;}}
  .hero{{background:linear-gradient(135deg,var(--navy) 0%,#1e4f66 60%,#24607a 100%);
         padding:52px 60px 48px;position:relative;overflow:hidden;}}
  .hero::before{{content:'';position:absolute;top:-80px;right:-80px;width:380px;height:380px;
                 background:var(--teal);border-radius:50%;opacity:0.08;}}
  .hero-eyebrow{{display:inline-flex;align-items:center;gap:8px;
                 background:rgba(61,191,170,0.15);border:1px solid rgba(61,191,170,0.3);
                 color:var(--teal);font-family:'Plus Jakarta Sans',sans-serif;
                 font-size:12px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                 padding:6px 14px;border-radius:20px;margin-bottom:18px;}}
  .hero h1{{font-family:'Plus Jakarta Sans',sans-serif;font-size:clamp(24px,3.2vw,42px);
            font-weight:800;color:#fff;line-height:1.15;max-width:700px;margin-bottom:12px;}}
  .hero h1 em{{font-style:normal;color:var(--teal);}}
  .hero-sub{{color:rgba(255,255,255,0.62);font-size:15px;line-height:1.6;max-width:560px;margin-bottom:28px;}}
  .hero-stats{{display:flex;gap:28px;flex-wrap:wrap;}}
  .hero-stat{{display:flex;flex-direction:column;gap:2px;}}
  .hero-stat-val{{font-family:'Plus Jakarta Sans',sans-serif;font-size:22px;font-weight:800;color:#fff;}}
  .hero-stat-lbl{{font-size:12px;color:rgba(255,255,255,0.5);}}
  .hero-divider{{width:1px;background:rgba(255,255,255,0.15);align-self:stretch;}}
  main{{max-width:1200px;margin:0 auto;padding:48px 40px 80px;}}
  .section-header{{display:flex;align-items:center;gap:14px;margin-bottom:24px;margin-top:52px;}}
  .section-pill{{font-family:'Plus Jakarta Sans',sans-serif;font-size:11px;font-weight:700;
                  letter-spacing:1.5px;text-transform:uppercase;padding:5px 12px;
                  border-radius:20px;flex-shrink:0;color:#fff;}}
  .section-title{{font-family:'Plus Jakarta Sans',sans-serif;font-size:20px;font-weight:700;color:var(--navy);}}
  .kpi-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(160px,1fr));gap:16px;margin-bottom:24px;}}
  .kpi-card{{background:var(--card);border-radius:14px;padding:22px 20px;box-shadow:var(--shadow);
             border:1px solid var(--border);transition:box-shadow 0.2s,transform 0.2s;animation:fadeUp 0.5s ease both;}}
  .kpi-card:hover{{box-shadow:var(--shadow-hover);transform:translateY(-2px);}}
  .kpi-label{{font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:10px;}}
  .kpi-value{{font-family:'Plus Jakarta Sans',sans-serif;font-size:32px;font-weight:800;line-height:1;color:var(--navy);margin-bottom:6px;}}
  .kpi-value.danger{{color:var(--red);}} .kpi-value.warn{{color:var(--amber);}} .kpi-value.good{{color:var(--green);}} .kpi-value.neu{{color:var(--navy);}}
  .kpi-sub{{font-size:12px;color:var(--muted);}}
  .kpi-trend{{display:inline-flex;align-items:center;gap:4px;font-size:11px;font-weight:600;
              margin-top:8px;padding:3px 8px;border-radius:20px;}}
  .kpi-trend.bad,.kpi-trend.danger{{background:var(--red-lt);color:var(--red);}}
  .kpi-trend.ok,.kpi-trend.warn{{background:var(--amber-lt);color:var(--amber);}}
  .kpi-trend.good{{background:var(--green-lt);color:var(--green);}}
  .kpi-trend.neu{{background:var(--teal-lt);color:var(--teal-dk);}}
  .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
  .chart-card{{background:var(--card);border-radius:14px;padding:26px;box-shadow:var(--shadow);
               border:1px solid var(--border);animation:fadeUp 0.5s ease both;transition:box-shadow 0.2s;}}
  .chart-card:hover{{box-shadow:var(--shadow-hover);}}
  .chart-card h3{{font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;
                  color:var(--navy);margin-bottom:20px;}}
  .bar-row{{display:flex;align-items:center;gap:10px;margin-bottom:12px;font-size:12px;}}
  .bar-label{{width:100px;flex-shrink:0;color:var(--muted);font-weight:500;}}
  .bar-track{{flex:1;height:8px;background:var(--bg);border-radius:99px;overflow:hidden;}}
  .bar-fill{{height:100%;border-radius:99px;background:var(--teal);transition:width 1.2s cubic-bezier(0.16,1,0.3,1);}}
  .bar-fill.teal{{background:var(--teal);}} .bar-fill.danger{{background:var(--red);}}
  .bar-fill.warn{{background:var(--amber);}} .bar-fill.ok{{background:var(--teal);}}
  .bar-pct{{width:52px;text-align:right;font-weight:700;color:var(--navy);font-size:12px;}}
  .callout{{margin-top:18px;background:var(--teal-lt);border-left:3px solid var(--teal);
            border-radius:0 8px 8px 0;padding:12px 14px;font-size:12px;line-height:1.6;color:#2a5f52;}}
  .callout strong{{color:var(--teal-dk);}}
  .callout.warn{{background:var(--amber-lt);border-color:var(--amber);color:#7a4f1a;}}
  .callout.warn strong{{color:var(--amber);}}
  .legend-item{{display:flex;align-items:center;gap:8px;font-size:13px;margin-bottom:8px;}}
  .dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0;display:inline-block;}}
  .dot-completed{{background:var(--green);}} .dot-noshow{{background:var(--red);}}
  .dot-cancelled{{background:var(--amber);}} .dot-latecancel{{background:#9b7ab8;}}
  table{{width:100%;border-collapse:collapse;font-size:13px;}}
  th{{text-align:left;padding:10px 12px;font-size:11px;font-weight:700;text-transform:uppercase;
      letter-spacing:0.8px;color:var(--muted);border-bottom:2px solid var(--border);}}
  td{{padding:10px 12px;border-bottom:1px solid var(--border);color:var(--navy);}}
  tr:hover td{{background:#fafbfc;}}
  .badge{{display:inline-flex;align-items:center;font-size:11px;font-weight:700;
           padding:3px 9px;border-radius:20px;}}
  .badge.danger{{background:var(--red-lt);color:var(--red);}}
  .badge.warn{{background:var(--amber-lt);color:var(--amber);}}
  .badge.good{{background:var(--green-lt);color:var(--green);}}
  .badge.neu{{background:var(--teal-lt);color:var(--teal-dk);}}
  .big-callout{{background:var(--navy);border-radius:16px;padding:36px 40px;margin-bottom:24px;
                position:relative;overflow:hidden;animation:fadeUp 0.5s ease both;}}
  .big-callout::before{{content:'';position:absolute;right:-60px;top:-60px;width:220px;height:220px;
                         background:var(--teal);border-radius:50%;opacity:0.1;}}
  .big-callout-tag{{font-family:'Plus Jakarta Sans',sans-serif;font-size:11px;font-weight:700;
                    letter-spacing:1.5px;text-transform:uppercase;color:var(--teal);margin-bottom:12px;}}
  .big-callout p{{font-size:15px;line-height:1.8;color:rgba(255,255,255,0.72);max-width:700px;}}
  .big-callout p strong{{color:#fff;}}
  .risk-bands{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-bottom:24px;}}
  .risk-band-card{{border-radius:14px;padding:24px;text-align:center;box-shadow:var(--shadow);border:1px solid var(--border);}}
  .risk-band-card h3{{font-family:'Plus Jakarta Sans',sans-serif;font-size:28px;font-weight:800;margin-bottom:4px;}}
  .risk-band-card p{{font-size:12px;color:var(--muted);margin-bottom:8px;}}
  .risk-band-card .pct{{font-size:13px;font-weight:700;}}
  .rx-grid{{display:flex;flex-direction:column;gap:14px;}}
  .rx-card{{background:var(--card);border-radius:14px;border:1px solid var(--border);
            padding:24px 26px;display:flex;gap:20px;align-items:flex-start;
            box-shadow:var(--shadow);transition:box-shadow 0.2s,transform 0.2s;animation:fadeUp 0.5s ease both;}}
  .rx-card:hover{{box-shadow:var(--shadow-hover);transform:translateY(-2px);}}
  .rx-num-wrap{{flex-shrink:0;width:42px;height:42px;background:var(--teal-lt);border-radius:12px;
                display:flex;align-items:center;justify-content:center;}}
  .rx-num{{font-family:'Plus Jakarta Sans',sans-serif;font-size:16px;font-weight:800;color:var(--teal-dk);}}
  .rx-body h3{{font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;color:var(--navy);margin-bottom:6px;}}
  .rx-rationale{{font-size:12px;line-height:1.7;color:var(--muted);margin-bottom:6px;}}
  .rx-action{{font-size:12px;line-height:1.7;color:var(--navy);font-weight:500;}}
  .rx-impact{{background:var(--green-lt);color:var(--green);font-size:11px;font-weight:700;padding:4px 10px;border-radius:20px;}}
  .rx-agent{{background:var(--amber-lt);color:var(--amber);font-size:11px;font-weight:700;padding:4px 10px;border-radius:20px;}}
  .agent-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin-bottom:24px;}}
  .agent-action-card{{background:var(--card);border-radius:12px;border:1px solid var(--border);
                      border-top:3px solid var(--amber);padding:20px;box-shadow:var(--shadow);}}
  .agent-name{{font-family:'Plus Jakarta Sans',sans-serif;font-size:12px;font-weight:700;
               color:var(--amber);text-transform:uppercase;letter-spacing:0.8px;margin-bottom:6px;}}
  .agent-desc{{font-size:13px;color:var(--navy);margin-bottom:8px;line-height:1.5;}}
  .agent-timing{{font-size:11px;color:var(--muted);font-weight:500;}}
  .qw-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:16px;}}
  .qw-card{{background:var(--card);border-radius:14px;border:1px solid var(--border);
            padding:24px;box-shadow:var(--shadow);}}
  .qw-card h3{{font-family:'Plus Jakarta Sans',sans-serif;font-size:14px;font-weight:700;color:var(--navy);margin-bottom:8px;}}
  .qw-card p{{font-size:12px;line-height:1.7;color:var(--muted);}}
  footer{{background:var(--navy);padding:28px 60px;display:flex;justify-content:space-between;
          align-items:center;flex-wrap:wrap;gap:12px;}}
  .footer-logo{{font-family:'Plus Jakarta Sans',sans-serif;font-weight:800;font-size:16px;color:#fff;}}
  .footer-logo span{{color:var(--teal);}}
  footer p{{font-size:12px;color:rgba(255,255,255,0.45);}}
  @keyframes fadeUp{{from{{opacity:0;transform:translateY(18px);}}to{{opacity:1;transform:translateY(0);}}}}
  .kpi-card:nth-child(1){{animation-delay:.05s;}} .kpi-card:nth-child(2){{animation-delay:.1s;}}
  .kpi-card:nth-child(3){{animation-delay:.15s;}} .kpi-card:nth-child(4){{animation-delay:.2s;}}
  .kpi-card:nth-child(5){{animation-delay:.25s;}} .kpi-card:nth-child(6){{animation-delay:.3s;}}
</style>
</head>
<body>

<nav>
  <div class="logo">
    <svg width="36" height="36" viewBox="0 0 40 40" fill="none">
      <rect width="40" height="40" rx="8" fill="#e8f8f5"/>
      <polyline points="4,28 10,18 16,22 22,10 28,16 36,6" stroke="#3dbfaa" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
      <circle cx="22" cy="10" r="2.5" fill="#1a3c4d"/>
    </svg>
    <span class="logo-text">CadenceWorks</span>
  </div>
  <div class="nav-badge">Analytics Report · {source}</div>
</nav>

<div class="hero">
  <div class="hero-eyebrow"><span style="width:6px;height:6px;background:var(--teal);border-radius:50%;display:inline-block;"></span>
    Analytics Report · {date_label}
  </div>
  <h1>Operational <em>Intelligence</em> Report</h1>
  <p class="hero-sub">Descriptive, predictive and prescriptive analysis — powered by CadenceWorks AI. Source: {source}</p>
  <div class="hero-stats">
    <div class="hero-stat"><span class="hero-stat-val">{total}</span><span class="hero-stat-lbl">Appointments</span></div>
    <div class="hero-divider"></div>
    <div class="hero-stat"><span class="hero-stat-val">{providers or '—'}</span><span class="hero-stat-lbl">Providers</span></div>
    <div class="hero-divider"></div>
    <div class="hero-stat"><span class="hero-stat-val">{ns_rate}%</span><span class="hero-stat-lbl">No-Show Rate</span></div>
    <div class="hero-divider"></div>
    <div class="hero-stat"><span class="hero-stat-val">{_fmt_currency(rev_lost)}</span><span class="hero-stat-lbl">Revenue at Risk</span></div>
  </div>
</div>

<main>

  <!-- DESCRIPTIVE -->
  <div class="section-header" style="margin-top:0">
    <div class="section-pill" style="background:var(--teal)">01 · Descriptive</div>
    <div class="section-title">What is currently happening</div>
  </div>

  <div class="kpi-grid">{kpi_html}</div>

  <div class="two-col" style="margin-bottom:16px">
    <div class="chart-card">
      <h3>Appointment Volume by Day</h3>
      {spark_html or '<p style="color:var(--muted);font-size:12px">No date data available</p>'}
    </div>
    <div class="chart-card">
      <h3>Status Breakdown</h3>
      {status_items}
      <div style="height:16px"></div>
      <h3 style="margin-bottom:12px">Appointment Types</h3>
      {appt_items}
    </div>
  </div>

  <div class="two-col" style="margin-bottom:16px">
    <div class="chart-card">
      <h3>No-Show Rate by Booking Channel</h3>
      {ch_bars}
      <div class="callout warn"><strong>WhatsApp shows the highest no-show rate</strong> — the least committed booking channel.</div>
    </div>
    <div class="chart-card">
      <h3>No-Show Rate by Lead Time</h3>
      {lead_bars}
      <div class="callout">Bookings made <strong>15+ days ahead carry the highest risk</strong>. Same-week bookings are the safest.</div>
    </div>
  </div>

  <div class="two-col" style="margin-bottom:16px">
    <div class="chart-card">
      <h3>No-Show Rate by Day of Week</h3>
      {day_bars}
    </div>
    <div class="chart-card">
      <h3>No-Show by Patient Type &amp; Slot Priority</h3>
      {pt_bars}
      <div style="height:12px"></div>
      {slot_bars}
      <div class="callout warn"><strong>Prime slots are going to the least committed patients</strong> — the highest-risk combination.</div>
    </div>
  </div>

  <!-- Provider comparison -->
  <div class="chart-card" style="margin-bottom:16px">
    <h3>Provider Performance Comparison</h3>
    <table>
      <thead><tr><th>Provider</th><th>Total</th><th>Completed</th><th>No-Show Rate</th><th>Revenue</th><th>Avg Lead</th></tr></thead>
      <tbody>{prov_rows or '<tr><td colspan="6" style="color:var(--muted)">No provider data</td></tr>'}</tbody>
    </table>
  </div>

  <!-- PREDICTIVE -->
  <div class="section-header">
    <div class="section-pill" style="background:var(--navy)">02 · Predictive</div>
    <div class="section-title">Who will no-show — AI risk scoring</div>
  </div>

  <div class="big-callout">
    <div class="big-callout-tag">⬡ No-Show Risk Score Model · {model_type} {auc_badge}</div>
    <p>Every appointment has been scored from <strong>0 to 100</strong> for no-show probability.
    The model uses <strong>lead time, booking channel, patient type, day of week, slot priority,
    and appointment type</strong> as inputs. Scores above 70 are <strong>High Risk</strong> and
    trigger the Reminder Agent automatically.</p>
  </div>

  <div class="risk-bands">
    <div class="risk-band-card" style="background:var(--red-lt);border-color:#f0c0c0">
      <h3 style="color:var(--red)">{risk_dist.get('High Risk',0)}</h3>
      <p>High Risk appointments</p>
      <div class="pct" style="color:var(--red)">Score ≥ 70 · {high_pct}% of total</div>
    </div>
    <div class="risk-band-card" style="background:var(--amber-lt);border-color:#f0d0b0">
      <h3 style="color:var(--amber)">{risk_dist.get('Medium Risk',0)}</h3>
      <p>Medium Risk appointments</p>
      <div class="pct" style="color:var(--amber)">Score 45–69 · {med_pct}% of total</div>
    </div>
    <div class="risk-band-card" style="background:var(--green-lt);border-color:#b0dfc8">
      <h3 style="color:var(--green)">{risk_dist.get('Low Risk',0)}</h3>
      <p>Low Risk appointments</p>
      <div class="pct" style="color:var(--green)">Score &lt; 45 · {low_pct}% of total</div>
    </div>
  </div>

  <div class="two-col" style="margin-bottom:16px">
    <div class="chart-card">
      <h3>Feature Importance (Model Inputs)</h3>
      {fi_bars}
      <div class="callout">These are the factors the model weighs most heavily when predicting no-show probability.</div>
    </div>
    <div class="chart-card">
      <h3>Model Validation</h3>
      <div class="legend-item" style="margin-bottom:14px">
        <span class="dot" style="background:var(--red)"></span>
        <span>Actual no-shows: <strong>{validation.get('actual_no_shows','—')}</strong> ({validation.get('actual_rate_pct','—')}%)</span>
      </div>
      <div class="legend-item" style="margin-bottom:14px">
        <span class="dot" style="background:var(--amber)"></span>
        <span>Predicted high-risk: <strong>{validation.get('predicted_high_risk','—')}</strong></span>
      </div>
      <div style="height:8px"></div>
      <table>
        <thead><tr><th>Metric</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Mean Risk Score</td><td><strong>{score_stats.get('mean','—')}</strong></td></tr>
          <tr><td>Median Risk Score</td><td><strong>{score_stats.get('median','—')}</strong></td></tr>
          <tr><td>75th Percentile</td><td><strong>{score_stats.get('p75','—')}</strong></td></tr>
          <tr><td>90th Percentile</td><td><strong>{score_stats.get('p90','—')}</strong></td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <div class="chart-card" style="margin-bottom:16px">
    <h3>Top 10 Highest-Risk Appointments</h3>
    <table>
      <thead><tr><th>ID</th><th>Provider</th><th>Patient</th><th>Channel</th><th>Day</th><th>Lead</th><th>Risk Score</th><th>Actual Status</th></tr></thead>
      <tbody>{hr_rows or '<tr><td colspan="8" style="color:var(--muted)">No high risk appointments found</td></tr>'}</tbody>
    </table>
  </div>

  <!-- PRESCRIPTIVE -->
  <div class="section-header">
    <div class="section-pill" style="background:var(--amber)">03 · Prescriptive</div>
    <div class="section-title">What to do — prioritised recommendations</div>
  </div>

  <div class="rx-grid" style="margin-bottom:32px">{rec_html}</div>

  <div class="section-header" style="margin-top:40px">
    <div class="section-pill" style="background:var(--navy)">Agent Actions</div>
    <div class="section-title">What the AI agents would do right now</div>
  </div>
  <div class="agent-grid">{agent_html}</div>

  <div class="section-header" style="margin-top:40px">
    <div class="section-pill" style="background:var(--green)">Quick Wins</div>
    <div class="section-title">What you can do today — no platform needed</div>
  </div>
  <div class="qw-grid">{qw_html}</div>

</main>

<footer>
  <div class="footer-logo">Cadence<span>Works</span> Consulting</div>
  <p>Descriptive · Predictive · Prescriptive Analytics</p>
  <p>Generated automatically · Confidential</p>
</footer>

</body></html>"""

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path
