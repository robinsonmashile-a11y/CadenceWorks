"""
CadenceWorks Analytics Engine
Streamlit App — run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from engine import ingestor, descriptive, predictive, prescriptive
from engine import live_sync
from engine import reminder_agent

# ── Init live DB
live_sync.init_db()
reminder_agent.init_reminder_tables()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CadenceWorks Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap');

  :root {
    --navy: #1a3c4d;
    --teal: #3dbfaa;
    --teal-dk: #2ea393;
    --teal-lt: #e8f8f5;
    --amber: #e8923a;
    --amber-lt: #fdf5ed;
    --green: #2ea37a;
    --green-lt: #edf7f3;
    --red: #e05252;
    --red-lt: #fdf0f0;
    --muted: #6b8899;
    --border: #e0e6ea;
  }

  /* Global font */
  html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
  }

  /* Hide default Streamlit chrome */
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding-top: 0rem !important; padding-bottom: 2rem; }

  /* Sidebar */
  [data-testid="stSidebar"] {
    background: #1a3c4d !important;
  }
  [data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
  }
  [data-testid="stSidebar"] .stMarkdown h1,
  [data-testid="stSidebar"] .stMarkdown h2,
  [data-testid="stSidebar"] .stMarkdown h3 {
    color: #fff !important;
  }
  [data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.15) !important;
  }

  /* Upload area */
  [data-testid="stFileUploader"] {
    background: var(--teal-lt);
    border: 2px dashed var(--teal) !important;
    border-radius: 12px;
    padding: 12px;
  }

  /* Metric cards */
  [data-testid="metric-container"] {
    background: #fff;
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 16px 20px !important;
    box-shadow: 0 2px 12px rgba(26,60,77,0.08);
  }
  [data-testid="stMetricLabel"] {
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: var(--muted) !important;
  }
  [data-testid="stMetricValue"] {
    font-size: 28px !important;
    font-weight: 800 !important;
    color: var(--navy) !important;
  }

  /* Tabs */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 2px solid var(--border);
  }
  .stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 13px;
    color: var(--muted);
    padding: 8px 20px;
  }
  .stTabs [aria-selected="true"] {
    background: var(--teal-lt) !important;
    color: var(--teal-dk) !important;
    border-bottom: 2px solid var(--teal) !important;
  }

  /* Tables */
  [data-testid="stTable"] table {
    border-collapse: collapse;
    width: 100%;
  }
  [data-testid="stTable"] th {
    background: #f0f2f5;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: var(--muted);
    padding: 10px 12px;
  }
  [data-testid="stTable"] td {
    font-size: 13px;
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
  }

  /* Cards via markdown */
  .cw-card {
    background: #fff;
    border: 1px solid #e0e6ea;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 14px;
    box-shadow: 0 2px 12px rgba(26,60,77,0.08);
  }
  .cw-hero {
    background: linear-gradient(135deg, #1a3c4d 0%, #1e4f66 60%, #24607a 100%);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 24px;
    color: white;
  }
  .cw-badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.5px;
  }
  .badge-teal  { background: #e8f8f5; color: #2ea393; }
  .badge-red   { background: #fdf0f0; color: #e05252; }
  .badge-amber { background: #fdf5ed; color: #e8923a; }
  .badge-green { background: #edf7f3; color: #2ea37a; }
  .badge-navy  { background: #eaf0f3; color: #1a3c4d; }

  .rx-card {
    background: #fff;
    border: 1px solid #e0e6ea;
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 12px;
    border-left: 4px solid #3dbfaa;
    box-shadow: 0 2px 8px rgba(26,60,77,0.06);
  }
  .rx-card.high { border-left-color: #e05252; }
  .rx-card.med  { border-left-color: #e8923a; }

  .agent-card {
    background: #fff;
    border: 1px solid #e0e6ea;
    border-top: 3px solid #e8923a;
    border-radius: 12px;
    padding: 16px 18px;
    box-shadow: 0 2px 8px rgba(26,60,77,0.06);
  }

  .risk-high { color: #e05252; font-weight: 700; }
  .risk-med  { color: #e8923a; font-weight: 700; }
  .risk-low  { color: #2ea37a; font-weight: 700; }

  /* Section headers */
  .section-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 16px;
  }

  /* Progress bars */
  .stProgress > div > div {
    background: var(--teal) !important;
    border-radius: 99px;
  }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 20px 0'>
      <div style='display:flex; align-items:center; gap:10px; margin-bottom:6px'>
        <svg width='32' height='32' viewBox='0 0 40 40' fill='none'>
          <rect width='40' height='40' rx='8' fill='rgba(61,191,170,0.2)'/>
          <polyline points='4,28 10,18 16,22 22,10 28,16 36,6'
            stroke='#3dbfaa' stroke-width='2.5'
            stroke-linecap='round' stroke-linejoin='round' fill='none'/>
          <circle cx='22' cy='10' r='2.5' fill='white'/>
        </svg>
        <span style='font-size:18px; font-weight:800; color:white'>CadenceWorks</span>
      </div>
      <div style='font-size:11px; color:rgba(255,255,255,0.45); letter-spacing:1px; text-transform:uppercase'>
        Analytics Engine v1.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Upload Data")

    uploaded_file = st.file_uploader(
        "Drop your booking file here",
        type=["xlsx", "xls", "csv"],
        help="Supports Excel (.xlsx) or CSV files. The engine auto-detects column names.",
    )

    st.markdown("---")
    st.markdown("### ⚙️ Settings")

    currency_symbol = st.selectbox("Currency", ["R", "$", "£", "€", "KSh"], index=0)
    no_show_threshold = st.slider("No-show alert threshold (%)", 5, 20, 8)

    st.markdown("---")
    st.markdown("### 🤖 AI Narrator")
    anthropic_api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        placeholder="sk-ant-...",
        help="Paste your Anthropic API key to enable plain-English AI narrative. Get one at console.anthropic.com",
    )
    if anthropic_api_key:
        st.markdown("""
        <div style='background:rgba(61,191,170,0.15);border-radius:8px;padding:8px 12px;
                    font-size:11px;color:#3dbfaa;margin-top:4px'>
          ✓ AI Narrator active
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='font-size:11px;color:rgba(255,255,255,0.35);margin-top:4px;line-height:1.6'>
          Add your API key to unlock the AI plain-English summary.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:rgba(255,255,255,0.4); line-height:1.8'>
      <strong style='color:rgba(255,255,255,0.7)'>How it works</strong><br>
      1. Upload any booking Excel/CSV<br>
      2. Engine auto-maps your columns<br>
      3. Runs descriptive, predictive<br>
         &amp; prescriptive analytics<br>
      4. Explore results in each tab
    </div>
    """, unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────
def fmt_currency(val, symbol=None):
    s = symbol or currency_symbol
    if val >= 1_000_000: return f"{s}{val/1_000_000:.1f}M"
    if val >= 1_000:     return f"{s}{val/1_000:.0f}k"
    return f"{s}{val:,.0f}"

def risk_badge(score):
    if score >= 20:  return f'<span class="cw-badge badge-red">High Risk · {score}</span>'
    if score >= 12:  return f'<span class="cw-badge badge-amber">Medium Risk · {score}</span>'
    return f'<span class="cw-badge badge-green">Low Risk · {score}</span>'

def bar_html(label, val, max_val, color="teal", suffix="%"):
    pct = round(val / max_val * 100) if max_val else 0
    colors = {"teal": "#3dbfaa", "red": "#e05252", "amber": "#e8923a", "navy": "#1a3c4d", "green": "#2ea37a"}
    c = colors.get(color, "#3dbfaa")
    return f"""
    <div style='display:flex;align-items:center;gap:10px;margin-bottom:10px;font-size:13px'>
      <span style='width:110px;flex-shrink:0;color:#6b8899;font-weight:500'>{label}</span>
      <div style='flex:1;height:8px;background:#f0f2f5;border-radius:99px;overflow:hidden'>
        <div style='width:{pct}%;height:100%;background:{c};border-radius:99px'></div>
      </div>
      <span style='width:52px;text-align:right;font-weight:700;color:#1a3c4d'>{val}{suffix}</span>
    </div>"""

def noshow_color(val, lo=8, hi=15):
    if val >= hi: return "red"
    if val >= lo: return "amber"
    return "teal"


# ── AI Narrator function ──────────────────────────────────────────────────────
def generate_narrative(kpis, desc, pred, presc, api_key, currency="R"):
    """Call Anthropic API to generate plain-English narrative from analytics data."""
    import urllib.request, json

    noshow_by_day  = desc.get("noshow_by_day", {})
    noshow_by_ch   = desc.get("noshow_by_channel", {})
    noshow_by_lead = desc.get("noshow_by_lead_time", {})
    top_rec        = presc.get("recommendations", [{}])[0]
    risk_dist      = pred.get("risk_distribution", {})
    worst_day      = max(noshow_by_day, key=noshow_by_day.get) if noshow_by_day else "unknown"
    worst_ch       = max(noshow_by_ch,  key=noshow_by_ch.get)  if noshow_by_ch  else "unknown"

    prompt = f"""You are the Insight Narrator for CadenceWorks, an AI analytics platform for service businesses.
You have just analysed a client's booking data. Write a concise, sharp, plain-English executive summary 
in exactly 3 short paragraphs. No bullet points. No headers. No markdown. Just direct, confident prose.

DATA:
- Total appointments: {kpis.get("total_appointments")}
- Completion rate: {kpis.get("completion_rate")}%
- No-show rate: {kpis.get("no_show_rate")}% (industry threshold: 8%)
- Revenue lost to no-shows: {currency}{kpis.get("revenue_lost", 0):,.0f}
- Avg lead time: {kpis.get("avg_lead_time_days")} days
- Worst no-show day: {worst_day} ({noshow_by_day.get(worst_day, 0)}%)
- Worst channel: {worst_ch} ({noshow_by_ch.get(worst_ch, 0)}%)
- No-show rate for 15+ day bookings: {noshow_by_lead.get("15+ days", 0)}%
- No-show rate for 0-3 day bookings: {noshow_by_lead.get("0–3 days", 0)}%
- High risk appointments: {risk_dist.get("High Risk", 0)}
- Top recommendation: {top_rec.get("title", "")}

PARAGRAPH 1: State the single biggest operational problem revealed by this data. Be specific — use the actual numbers. Name the day, channel, or pattern that is costing the most.

PARAGRAPH 2: Explain the root cause in plain language. Why is this happening? What does the data suggest about patient behaviour or booking policy?

PARAGRAPH 3: State the single most important action the practice should take this week. Be direct and specific. End with the estimated financial impact.

Write as if you are a sharp analyst briefing a practice manager over coffee. Confident, clear, no fluff."""

    payload = json.dumps({{
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 500,
        "messages": [{{"role": "user", "content": prompt}}]
    }}).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={{
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    return result["content"][0]["text"]


def section_header(pill_text, title, pill_color="#3dbfaa"):
    st.markdown(f"""
    <div style='display:flex;align-items:center;gap:14px;margin-bottom:20px;margin-top:8px'>
      <span class='section-pill' style='background:{pill_color};color:white'>{pill_text}</span>
      <span style='font-size:20px;font-weight:700;color:#1a3c4d'>{title}</span>
    </div>""", unsafe_allow_html=True)


# ── Landing page (no file uploaded) ──────────────────────────────────────────
if not uploaded_file:
    st.markdown("""
    <div class='cw-hero'>
      <div style='font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
                  color:#3dbfaa;margin-bottom:14px'>● Analytics Engine</div>
      <h1 style='font-size:36px;font-weight:800;color:white;line-height:1.2;margin-bottom:12px'>
        Upload your booking data.<br>Get <em style='color:#3dbfaa;font-style:normal'>instant intelligence.</em>
      </h1>
      <p style='color:rgba(255,255,255,0.65);font-size:15px;line-height:1.7;max-width:560px'>
        Drop any Excel or CSV booking file in the sidebar. The engine auto-detects your columns,
        runs descriptive, predictive and prescriptive analytics, and surfaces actionable recommendations
        — all in seconds.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='cw-card'>
          <div style='font-size:24px;margin-bottom:10px'>📊</div>
          <div style='font-weight:700;color:#1a3c4d;margin-bottom:6px'>Descriptive</div>
          <div style='font-size:13px;color:#6b8899;line-height:1.6'>
            KPIs, no-show rates by channel, day, lead time, patient type and provider.
          </div></div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='cw-card'>
          <div style='font-size:24px;margin-bottom:10px'>🤖</div>
          <div style='font-weight:700;color:#1a3c4d;margin-bottom:6px'>Predictive</div>
          <div style='font-size:13px;color:#6b8899;line-height:1.6'>
            AI risk scores every appointment 0–100. Flags who is likely to no-show before it happens.
          </div></div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='cw-card'>
          <div style='font-size:24px;margin-bottom:10px'>🎯</div>
          <div style='font-weight:700;color:#1a3c4d;margin-bottom:6px'>Prescriptive</div>
          <div style='font-size:13px;color:#6b8899;line-height:1.6'>
            Prioritised recommendations and AI agent actions grounded in your actual data.
          </div></div>""", unsafe_allow_html=True)

    st.info("👈 Upload an Excel or CSV file in the sidebar to get started.")
    st.stop()


# ── Run pipeline ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_bytes, file_name):
    import tempfile, os
    suffix = Path(file_name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        df, meta   = ingestor.ingest(tmp_path)
        desc       = descriptive.run(df)
        pred       = predictive.run(df)
        presc      = prescriptive.run(df, desc, pred)
        return df, meta, desc, pred, presc
    finally:
        os.unlink(tmp_path)

with st.spinner("🔄 Running analytics pipeline..."):
    df, meta, desc, pred, presc = run_pipeline(
        uploaded_file.read(), uploaded_file.name
    )

kpis = desc.get("kpis", {})
dr   = meta.get("date_range", ("—", "—"))


# ── Top hero strip ────────────────────────────────────────────────────────────
ns_rate = kpis.get("no_show_rate", 0)
alert_color = "#e05252" if ns_rate >= no_show_threshold else "#2ea37a"

st.markdown(f"""
<div style='background:linear-gradient(135deg,#1a3c4d 0%,#1e4f66 60%,#24607a 100%);
            border-radius:16px;padding:28px 36px;margin-bottom:24px;position:relative;overflow:hidden'>
  <div style='position:absolute;top:-60px;right:-60px;width:220px;height:220px;
              background:#3dbfaa;border-radius:50%;opacity:0.08'></div>
  <div style='font-size:11px;font-weight:700;letter-spacing:2px;text-transform:uppercase;
              color:#3dbfaa;margin-bottom:8px'>● Operational Intelligence Report</div>
  <div style='font-size:22px;font-weight:800;color:white;margin-bottom:6px'>
    {meta.get("source_file","Uploaded File")}
  </div>
  <div style='font-size:13px;color:rgba(255,255,255,0.55)'>
    {dr[0]} to {dr[1]} &nbsp;·&nbsp; {kpis.get("total_appointments",0)} appointments
    &nbsp;·&nbsp; {", ".join(meta.get("providers",[]))}
    &nbsp;·&nbsp; <span style='color:{alert_color};font-weight:700'>No-show rate: {ns_rate}%</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ── AI Narrative panel ───────────────────────────────────────────────────────
if anthropic_api_key:
    with st.container():
        col_n1, col_n2 = st.columns([11, 1])
        with col_n1:
            st.markdown("""
            <div style='display:flex;align-items:center;gap:10px;margin-bottom:12px'>
              <div style='background:#1a3c4d;border-radius:8px;padding:6px 10px;
                          font-size:11px;font-weight:700;letter-spacing:1px;
                          text-transform:uppercase;color:#3dbfaa'>⬡ AI Narrator</div>
              <span style='font-size:13px;color:#6b8899'>Plain-English summary generated from your data</span>
            </div>""", unsafe_allow_html=True)

        narrative_key = f"narrative_{uploaded_file.name}_{kpis.get('total_appointments')}"
        if narrative_key not in st.session_state:
            with st.spinner("✍ AI Narrator is reading your data..."):
                try:
                    st.session_state[narrative_key] = generate_narrative(
                        kpis, desc, pred, presc,
                        api_key=anthropic_api_key,
                        currency=currency_symbol,
                    )
                except Exception as e:
                    st.session_state[narrative_key] = None
                    st.error(f"AI Narrator error: {e}")

        narrative_text = st.session_state.get(narrative_key)
        if narrative_text:
            paragraphs = [p.strip() for p in narrative_text.strip().split("\n\n") if p.strip()]
            para_html = "".join(
                f"<p style='margin:0 0 14px 0;line-height:1.8;font-size:14px;color:#1a3c4d'>{p}</p>"
                for p in paragraphs
            )
            st.markdown(f"""
            <div style='background:#fff;border:1px solid #e0e6ea;border-left:4px solid #3dbfaa;
                        border-radius:0 14px 14px 0;padding:24px 28px;margin-bottom:24px;
                        box-shadow:0 2px 12px rgba(26,60,77,0.08)'>
              {para_html}
            </div>""", unsafe_allow_html=True)

        with col_n2:
            if st.button("↺", help="Regenerate narrative"):
                if narrative_key in st.session_state:
                    del st.session_state[narrative_key]
                st.rerun()
else:
    st.markdown("""
    <div style='background:#fff;border:1px dashed #e0e6ea;border-radius:14px;
                padding:20px 24px;margin-bottom:24px;text-align:center'>
      <span style='font-size:13px;color:#6b8899'>
        🤖 Add your Anthropic API key in the sidebar to unlock the <strong>AI Narrator</strong>
        — a plain-English summary of your data written fresh every report.
      </span>
    </div>""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Descriptive",
    "🤖  Predictive",
    "🎯  Prescriptive",
    "⚡  Score New Bookings",
    "🔴  Live Monitor",
    "💬  Reminder Agent",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    section_header("01 · Descriptive", "What is currently happening")

    # KPI row
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Total Appointments",  kpis.get("total_appointments", 0))
    c2.metric("Completion Rate",     f"{kpis.get('completion_rate', 0)}%")
    c3.metric("No-Show Rate",        f"{kpis.get('no_show_rate', 0)}%",
              delta=f"{kpis.get('no_show_rate',0) - no_show_threshold:.1f}% vs threshold",
              delta_color="inverse")
    c4.metric("Cancellation Rate",   f"{kpis.get('cancellation_rate', 0)}%")
    c5.metric("Revenue at Risk",     fmt_currency(kpis.get("revenue_lost", 0)))
    c6.metric("Avg Lead Time",       f"{kpis.get('avg_lead_time_days', 0)}d")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Row 1: Channel + Lead Time
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("**No-Show Rate by Booking Channel**")
        ch_data = desc.get("noshow_by_channel", {})
        max_v = max(ch_data.values()) if ch_data else 1
        bars = "".join(bar_html(k, v, max_v, noshow_color(v)) for k, v in ch_data.items())
        st.markdown(bars, unsafe_allow_html=True)
        if "WhatsApp" in ch_data:
            wa = ch_data["WhatsApp"]
            ph = ch_data.get("Phone", 1)
            st.markdown(f"""<div style='background:#fdf5ed;border-left:3px solid #e8923a;
                border-radius:0 8px 8px 0;padding:10px 14px;font-size:12px;color:#7a4f1a;margin-top:12px'>
                WhatsApp no-show rate is <strong>{round(wa/ph,1)}× higher</strong> than phone bookings.</div>""",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("**No-Show Rate by Lead Time**")
        ld_data = desc.get("noshow_by_lead_time", {})
        max_v = max(ld_data.values()) if ld_data else 1
        bars = "".join(bar_html(k, v, max_v, noshow_color(v, 8, 18)) for k, v in ld_data.items())
        st.markdown(bars, unsafe_allow_html=True)
        far = ld_data.get("15+ days", 0)
        near = ld_data.get("0–3 days", 1)
        if far and near:
            st.markdown(f"""<div style='background:#e8f8f5;border-left:3px solid #3dbfaa;
                border-radius:0 8px 8px 0;padding:10px 14px;font-size:12px;color:#2a5f52;margin-top:12px'>
                15+ day bookings are <strong>{round(far/near,1)}× more likely</strong> to no-show than same-week bookings.</div>""",
                unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Row 2: Day of week + Patient/Slot type
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("**No-Show Rate by Day of Week**")
        day_data = desc.get("noshow_by_day", {})
        max_v = max(day_data.values()) if day_data else 1
        bars = "".join(bar_html(k, v, max_v, noshow_color(v, 10, 18)) for k, v in day_data.items())
        st.markdown(bars, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col4:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("**No-Show by Patient Type & Slot Priority**")
        pt_data = desc.get("noshow_by_patient_type", {})
        sl_data = desc.get("noshow_by_slot_type", {})
        all_data = {**pt_data, **sl_data}
        max_v = max(all_data.values()) if all_data else 1
        bars = "".join(bar_html(k, v, max_v, noshow_color(v)) for k, v in all_data.items())
        st.markdown(bars, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Provider comparison table
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    prov = desc.get("provider_comparison", [])
    if prov:
        st.markdown("**Provider Performance Comparison**")
        prov_df = pd.DataFrame(prov).rename(columns={
            "provider": "Provider", "total": "Total", "completed": "Completed",
            "no_show_rate": "No-Show Rate (%)", "revenue": "Revenue",
            "avg_lead": "Avg Lead (days)"
        })
        prov_df["Revenue"] = prov_df["Revenue"].apply(lambda x: fmt_currency(x))
        st.dataframe(prov_df, use_container_width=True, hide_index=True)

    # Status breakdown
    col5, col6 = st.columns(2)
    with col5:
        vol_status = desc.get("volume_by_status", {})
        if vol_status:
            st.markdown("**Appointment Status Breakdown**")
            st.bar_chart(pd.Series(vol_status), use_container_width=True, height=200)
    with col6:
        vol_appt = desc.get("volume_by_appt_type", {})
        if vol_appt:
            st.markdown("**Volume by Appointment Type**")
            st.bar_chart(pd.Series(vol_appt), use_container_width=True, height=200)


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — PREDICTIVE
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    section_header("02 · Predictive", "Who will no-show — AI risk scoring", "#1a3c4d")

    # Model info banner
    model_type = pred.get("model_type", "—")
    auc        = pred.get("model_auc")
    auc_str    = f"AUC: {auc}" if auc else "Rule-Based Scoring"
    st.markdown(f"""
    <div style='background:#1a3c4d;border-radius:14px;padding:24px 28px;margin-bottom:24px'>
      <div style='font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                  color:#3dbfaa;margin-bottom:8px'>⬡ Model Active</div>
      <div style='font-size:15px;color:rgba(255,255,255,0.8);line-height:1.7;max-width:700px'>
        Every appointment has been scored <strong style='color:white'>0–100</strong> for no-show probability
        using <strong style='color:white'>{model_type}</strong>.
        The model uses lead time, channel, patient type, day of week, slot priority and appointment type.
        <span style='background:rgba(61,191,170,0.2);color:#3dbfaa;padding:2px 10px;
              border-radius:20px;font-size:12px;font-weight:700;margin-left:8px'>{auc_str}</span>
      </div>
    </div>""", unsafe_allow_html=True)

    # Risk band cards
    risk_dist = pred.get("risk_distribution", {})
    total_apts = kpis.get("total_appointments", 1)

    rc1, rc2, rc3 = st.columns(3)
    high = risk_dist.get("High Risk", 0)
    med  = risk_dist.get("Medium Risk", 0)
    low  = risk_dist.get("Low Risk", 0)
    rc1.metric("🔴 High Risk",   high, f"{round(high/total_apts*100)}% of appointments")
    rc2.metric("🟠 Medium Risk", med,  f"{round(med/total_apts*100)}% of appointments")
    rc3.metric("🟢 Low Risk",    low,  f"{round(low/total_apts*100)}% of appointments")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("**Feature Importance — What drives the risk score**")
        fi = pred.get("feature_importance", {})
        max_v = max(fi.values()) if fi else 1
        bars = "".join(
            bar_html(k.replace("_", " ").title(), round(v * 100), round(max_v * 100), "teal", "%")
            for k, v in list(fi.items())[:7]
        )
        st.markdown(bars, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='cw-card'>", unsafe_allow_html=True)
        st.markdown("**Model Validation**")
        val = pred.get("validation", {})
        ss  = pred.get("score_stats", {})
        st.markdown(f"""
        <table width='100%' style='font-size:13px;border-collapse:collapse'>
          <tr style='border-bottom:1px solid #e0e6ea'>
            <td style='padding:8px 4px;color:#6b8899'>Actual no-shows</td>
            <td style='padding:8px 4px;font-weight:700;text-align:right'>{val.get('actual_no_shows','—')} ({val.get('actual_rate_pct','—')}%)</td>
          </tr>
          <tr style='border-bottom:1px solid #e0e6ea'>
            <td style='padding:8px 4px;color:#6b8899'>Predicted high-risk</td>
            <td style='padding:8px 4px;font-weight:700;text-align:right'>{val.get('predicted_high_risk','—')}</td>
          </tr>
          <tr style='border-bottom:1px solid #e0e6ea'>
            <td style='padding:8px 4px;color:#6b8899'>Mean risk score</td>
            <td style='padding:8px 4px;font-weight:700;text-align:right'>{ss.get('mean','—')}</td>
          </tr>
          <tr style='border-bottom:1px solid #e0e6ea'>
            <td style='padding:8px 4px;color:#6b8899'>Median risk score</td>
            <td style='padding:8px 4px;font-weight:700;text-align:right'>{ss.get('median','—')}</td>
          </tr>
          <tr style='border-bottom:1px solid #e0e6ea'>
            <td style='padding:8px 4px;color:#6b8899'>75th percentile</td>
            <td style='padding:8px 4px;font-weight:700;text-align:right'>{ss.get('p75','—')}</td>
          </tr>
          <tr>
            <td style='padding:8px 4px;color:#6b8899'>90th percentile</td>
            <td style='padding:8px 4px;font-weight:700;text-align:right'>{ss.get('p90','—')}</td>
          </tr>
        </table>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # High-risk appointments table
    st.markdown("**Top High-Risk Appointments**")
    hr = pred.get("high_risk_appointments", [])
    if hr:
        hr_df = pd.DataFrame(hr)
        # Rename columns nicely
        col_map = {
            "appointment_id": "ID", "provider": "Provider",
            "patient_type": "Patient", "channel": "Channel",
            "day_of_week": "Day", "lead_time_days": "Lead (days)",
            "appointment_type": "Type", "risk_score": "Risk Score",
            "status": "Actual Status"
        }
        hr_df = hr_df.rename(columns={k: v for k, v in col_map.items() if k in hr_df.columns})
        st.dataframe(hr_df, use_container_width=True, hide_index=True)
    else:
        st.info("No high-risk appointments identified.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — PRESCRIPTIVE
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    section_header("03 · Prescriptive", "What to do — prioritised recommendations", "#e8923a")

    recs       = presc.get("recommendations", [])
    agent_acts = presc.get("agent_actions", [])
    quick_wins = presc.get("quick_wins", [])
    summary    = presc.get("summary", {})

    # Summary strip
    s1, s2, s3 = st.columns(3)
    s1.metric("Recommendations",   summary.get("total_recommendations", 0))
    s2.metric("High Priority",     summary.get("high_priority", 0))
    s3.metric("Agent Actions",     summary.get("total_agent_actions", 0))

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # Recommendations
    st.markdown("#### Prioritised Recommendations")
    for r in recs:
        priority   = r.get("priority", 1)
        card_class = "high" if priority <= 2 else ("med" if priority <= 4 else "")
        st.markdown(f"""
        <div class='rx-card {card_class}'>
          <div style='display:flex;align-items:flex-start;gap:16px'>
            <div style='background:#e8f8f5;border-radius:10px;width:38px;height:38px;flex-shrink:0;
                        display:flex;align-items:center;justify-content:center;
                        font-size:15px;font-weight:800;color:#2ea393'>0{priority}</div>
            <div style='flex:1'>
              <div style='font-weight:700;font-size:15px;color:#1a3c4d;margin-bottom:6px'>{r['title']}</div>
              <div style='font-size:13px;color:#6b8899;line-height:1.65;margin-bottom:8px'>{r['rationale']}</div>
              <div style='font-size:13px;color:#1a3c4d;font-weight:500;margin-bottom:10px'>→ {r['action']}</div>
              <div style='display:flex;gap:8px;flex-wrap:wrap'>
                <span class='cw-badge badge-green'>↑ {r['impact']}</span>
                <span class='cw-badge badge-amber'>🤖 {r['agent']}</span>
              </div>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    # Agent actions
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### What the AI Agents Would Do Right Now")
    a_cols = st.columns(len(agent_acts) if agent_acts else 1)
    for i, a in enumerate(agent_acts):
        with a_cols[i]:
            st.markdown(f"""
            <div class='agent-card'>
              <div style='font-size:11px;font-weight:700;color:#e8923a;text-transform:uppercase;
                          letter-spacing:0.8px;margin-bottom:6px'>{a['agent']}</div>
              <div style='font-size:13px;color:#1a3c4d;line-height:1.5;margin-bottom:8px'>{a['action']}</div>
              <div style='font-size:11px;color:#6b8899;font-weight:500'>⏱ {a['timing']}</div>
            </div>""", unsafe_allow_html=True)

    # Quick wins
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.markdown("#### Quick Wins — Do These Today, No Platform Needed")
    qw_cols = st.columns(len(quick_wins) if quick_wins else 1)
    badge_map = {"High": "badge-green", "Low": "badge-teal", "Medium": "badge-amber"}
    for i, q in enumerate(quick_wins):
        with qw_cols[i]:
            ec = badge_map.get(q["effort"], "badge-teal")
            ic = badge_map.get(q["impact"], "badge-green")
            st.markdown(f"""
            <div class='cw-card'>
              <div style='font-weight:700;font-size:14px;color:#1a3c4d;margin-bottom:8px'>{q['title']}</div>
              <div style='font-size:12px;color:#6b8899;line-height:1.65;margin-bottom:12px'>{q['detail']}</div>
              <span class='cw-badge {ec}'>Effort: {q['effort']}</span>&nbsp;
              <span class='cw-badge {ic}'>Impact: {q['impact']}</span>
            </div>""", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='margin-top:48px;border-top:1px solid #e0e6ea;padding-top:20px;
            display:flex;justify-content:space-between;align-items:center;
            font-size:12px;color:#6b8899'>
  <span><strong style='color:#1a3c4d'>CadenceWorks</strong> Consulting &nbsp;·&nbsp;
        Descriptive · Predictive · Prescriptive Analytics</span>
  <span>Generated automatically · Confidential</span>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — SCORE NEW BOOKINGS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    section_header("04 · Score New Bookings", "Paste or upload tomorrow's bookings — get instant risk scores", "#1a3c4d")

    st.markdown("""
    <div style='background:#1a3c4d;border-radius:14px;padding:24px 28px;margin-bottom:24px'>
      <div style='font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                  color:#3dbfaa;margin-bottom:8px'>⚡ How this works</div>
      <div style='font-size:14px;color:rgba(255,255,255,0.75);line-height:1.75;max-width:700px'>
        The model trained on your historical data is now applied to <strong style='color:white'>new bookings</strong>
        before they happen. Enter upcoming appointments manually or upload a new file —
        each gets an instant <strong style='color:white'>Risk Score (0–100)</strong> and a recommended action.
        High-risk bookings should trigger your reminder sequence immediately.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input method toggle ───────────────────────────────────────────────
    input_method = st.radio(
        "How do you want to enter new bookings?",
        ["📋  Manual entry (fill in the form)", "📁  Upload a new bookings file"],
        horizontal=True,
    )

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    new_bookings_df = None

    # ── OPTION A: Manual entry form ───────────────────────────────────────
    if "Manual" in input_method:
        st.markdown("**Enter upcoming appointment details:**")

        with st.form("new_booking_form"):
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

            # Row 1
            fc1, fc2, fc3 = st.columns(3)
            with fc1:
                providers_list = meta.get("providers", ["Dr Smith"])
                provider = st.selectbox("Provider", providers_list)
            with fc2:
                patient_type = st.selectbox("Patient Type", ["Returning", "New"])
            with fc3:
                channel = st.selectbox("Booking Channel", ["Phone", "Online", "WhatsApp"])

            # Row 2
            fc4, fc5, fc6 = st.columns(3)
            with fc4:
                day_of_week = st.selectbox("Day of Week", ["Mon", "Tue", "Wed", "Thu", "Fri"])
            with fc5:
                lead_time = st.number_input("Days until appointment", min_value=0, max_value=90, value=5)
            with fc6:
                appt_type = st.selectbox("Appointment Type", ["Consult", "Follow-up", "Procedure"])

            # Row 3
            fc7, fc8 = st.columns(2)
            with fc7:
                is_prime = st.selectbox("Prime Slot?", ["Yes (08:00–09:30 or 13:00–14:30)", "No"])
            with fc8:
                fee = st.number_input(f"Fee ({currency_symbol})", min_value=0, value=900, step=50)

            num_bookings = st.slider("How many bookings to score like this?", 1, 20, 1)
            submitted = st.form_submit_button("⚡ Score this booking", use_container_width=True)

        if submitted:
            rows = []
            for i in range(num_bookings):
                rows.append({
                    "appointment_id": f"NEW-{i+1:03d}",
                    "provider":       provider,
                    "patient_type":   patient_type,
                    "channel":        channel,
                    "day_of_week":    day_of_week,
                    "lead_time_days": int(lead_time),
                    "appointment_type": appt_type,
                    "is_prime_slot":  True if "Yes" in is_prime else False,
                    "fee":            float(fee),
                    "status":         "Pending",
                })
            new_bookings_df = pd.DataFrame(rows)

    # ── OPTION B: Upload new file ─────────────────────────────────────────
    else:
        st.markdown("**Upload your new bookings file** (same format as your historical data):")
        new_file = st.file_uploader(
            "Drop new bookings file here",
            type=["xlsx", "xls", "csv"],
            key="new_bookings_uploader",
            help="The engine will auto-map columns and score each booking."
        )

        if new_file:
            import tempfile, os
            suffix = Path(new_file.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(new_file.read())
                tmp_path = tmp.name
            try:
                new_bookings_df, _ = ingestor.ingest(tmp_path)
                # Add placeholder status for new bookings
                if "status" not in new_bookings_df.columns:
                    new_bookings_df["status"] = "Pending"
                st.success(f"✓ {len(new_bookings_df)} bookings loaded from {new_file.name}")
            except Exception as e:
                st.error(f"Could not load file: {e}")
            finally:
                os.unlink(tmp_path)

    # ── Score and display results ─────────────────────────────────────────
    if new_bookings_df is not None and len(new_bookings_df) > 0:

        # Apply the trained model's feature engineering + rule-based scorer
        # (since we can't guarantee ML model is available as standalone object,
        # we use the full predictive module on the new data)
        from engine.predictive import _build_features, _rule_based_score
        import numpy as np

        features = _build_features(new_bookings_df)
        scores   = features.apply(_rule_based_score, axis=1).values

        def band(s):
            if s >= 70:  return "🔴 High Risk"
            if s >= 45:  return "🟠 Medium Risk"
            return "🟢 Low Risk"

        def action(s):
            if s >= 70:  return "Send reminders at 72hr, 24hr & 4hr"
            if s >= 45:  return "Send reminder at 24hr"
            return "Standard 24hr reminder only"

        scored_new = new_bookings_df.copy()
        scored_new["Risk Score"] = [round(float(s), 1) for s in scores]
        scored_new["Risk Band"]  = [band(s) for s in scores]
        scored_new["Recommended Action"] = [action(s) for s in scores]

        # ── Summary metrics ───────────────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        high_n = sum(1 for s in scores if s >= 70)
        med_n  = sum(1 for s in scores if 45 <= s < 70)
        low_n  = sum(1 for s in scores if s < 45)
        total_n = len(scores)
        rev_at_risk = scored_new[scored_new["Risk Score"] >= 70]["fee"].sum() if "fee" in scored_new.columns else 0

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Scored",     total_n)
        sm2.metric("🔴 High Risk",     high_n,  f"{round(high_n/total_n*100)}% need urgent action")
        sm3.metric("🟠 Medium Risk",   med_n,   f"{round(med_n/total_n*100)}% need a reminder")
        sm4.metric("Revenue at Risk",  fmt_currency(rev_at_risk), "From high-risk slots")

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        # ── Alert for high-risk bookings ──────────────────────────────────
        if high_n > 0:
            st.markdown(f"""
            <div style='background:#fdf0f0;border:1px solid #f0c0c0;border-left:4px solid #e05252;
                        border-radius:10px;padding:16px 20px;margin-bottom:20px'>
              <div style='font-weight:700;color:#e05252;margin-bottom:4px'>
                ⚠ {high_n} high-risk appointment{"s" if high_n > 1 else ""} need immediate action
              </div>
              <div style='font-size:13px;color:#7a3a3a;line-height:1.6'>
                These bookings have a high predicted no-show probability.
                Trigger your reminder sequence now — 72hr, 24hr and 4hr before the appointment.
                {'Consider overbooking these slots or activating your waitlist.' if high_n > 2 else ''}
              </div>
            </div>""", unsafe_allow_html=True)

        # ── Scored table ──────────────────────────────────────────────────
        st.markdown("**Scored Bookings — sorted by risk (highest first)**")

        display_cols = [c for c in [
            "appointment_id", "provider", "patient_type", "channel",
            "day_of_week", "lead_time_days", "appointment_type",
            "Risk Score", "Risk Band", "Recommended Action"
        ] if c in scored_new.columns]

        display_df = scored_new[display_cols].sort_values("Risk Score", ascending=False)

        col_rename = {
            "appointment_id": "ID", "provider": "Provider",
            "patient_type": "Patient", "channel": "Channel",
            "day_of_week": "Day", "lead_time_days": "Lead (days)",
            "appointment_type": "Type",
        }
        display_df = display_df.rename(columns=col_rename)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # ── Download scored results ───────────────────────────────────────
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="⬇ Download scored bookings as CSV",
            data=csv,
            file_name="cadenceworks_scored_bookings.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ── Per-booking detail cards for high risk ────────────────────────
        if high_n > 0:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            st.markdown("**High-Risk Booking Detail**")
            high_risk_rows = scored_new[scored_new["Risk Score"] >= 70].sort_values("Risk Score", ascending=False)

            for _, row in high_risk_rows.iterrows():
                score = row["Risk Score"]
                why_parts = []
                if row.get("lead_time_days", 0) >= 15:
                    why_parts.append(f"booked {row['lead_time_days']} days out")
                if str(row.get("channel","")).lower() == "whatsapp":
                    why_parts.append("WhatsApp channel (highest no-show rate)")
                if str(row.get("patient_type","")).lower() == "new":
                    why_parts.append("new patient (no attendance history)")
                if str(row.get("day_of_week","")) in ["Thu", "Mon"]:
                    why_parts.append(f"{row.get('day_of_week')} is a high no-show day")
                if row.get("is_prime_slot") == True:
                    why_parts.append("prime slot at risk")
                why_str = " · ".join(why_parts) if why_parts else "combination of risk factors"

                st.markdown(f"""
                <div class='rx-card high'>
                  <div style='display:flex;align-items:flex-start;gap:16px'>
                    <div style='background:#fdf0f0;border-radius:10px;width:52px;height:52px;flex-shrink:0;
                                display:flex;align-items:center;justify-content:center;
                                font-size:18px;font-weight:800;color:#e05252'>{score}</div>
                    <div style='flex:1'>
                      <div style='font-weight:700;font-size:15px;color:#1a3c4d;margin-bottom:4px'>
                        {row.get('appointment_id','—')} &nbsp;·&nbsp;
                        {row.get('provider','—')} &nbsp;·&nbsp;
                        {row.get('patient_type','—')} patient &nbsp;·&nbsp;
                        {row.get('channel','—')} &nbsp;·&nbsp;
                        {row.get('day_of_week','—')}
                      </div>
                      <div style='font-size:12px;color:#6b8899;margin-bottom:10px'>
                        ⚠ Why high risk: {why_str}
                      </div>
                      <div style='font-size:13px;color:#1a3c4d;font-weight:500'>
                        → Action: Send reminders at 72hr, 24hr and 4hr via {row.get('channel','their channel')}
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

    elif new_bookings_df is not None and len(new_bookings_df) == 0:
        st.warning("No bookings found in the uploaded file.")



# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE MONITOR
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    section_header("05 · Live Monitor", "Real-time booking feed — auto-scored as they arrive", "#e05252")

    # ── Explainer ────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='background:#1a3c4d;border-radius:14px;padding:24px 28px;margin-bottom:24px'>
      <div style='font-size:11px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;
                  color:#3dbfaa;margin-bottom:10px'>⚡ How Live Sync Works</div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:20px'>
        <div>
          <div style='font-weight:700;color:white;font-size:13px;margin-bottom:5px'>1. Drop a file</div>
          <div style='font-size:12px;color:rgba(255,255,255,0.6);line-height:1.7'>
            Place any booking Excel or CSV into the <code style="background:rgba(255,255,255,0.1);
            padding:1px 6px;border-radius:4px">watch_folder/</code> directory.
            Export from your booking system and drop it in.
          </div>
        </div>
        <div>
          <div style='font-weight:700;color:white;font-size:13px;margin-bottom:5px'>2. Engine scores it</div>
          <div style='font-size:12px;color:rgba(255,255,255,0.6);line-height:1.7'>
            Click Sync Now or enable Auto-Sync. Every booking gets a risk score instantly.
            New files are detected automatically — duplicates are skipped.
          </div>
        </div>
        <div>
          <div style='font-weight:700;color:white;font-size:13px;margin-bottom:5px'>3. Act on alerts</div>
          <div style='font-size:12px;color:rgba(255,255,255,0.6);line-height:1.7'>
            High-risk bookings surface immediately with the recommended action.
            Mark them as reminded when done. Revenue at risk is tracked live.
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sync controls ─────────────────────────────────────────────────────────
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([2, 2, 2, 2])

    with ctrl1:
        watch_path = live_sync.WATCH_DIR.absolute()
        st.markdown(f"""
        <div class='cw-card' style='padding:16px 18px'>
          <div style='font-size:11px;font-weight:700;text-transform:uppercase;
                      letter-spacing:1px;color:#6b8899;margin-bottom:6px'>Watch Folder</div>
          <div style='font-size:12px;color:#1a3c4d;font-weight:600;word-break:break-all'>
            📁 {watch_path}
          </div>
          <div style='font-size:11px;color:#6b8899;margin-top:4px'>
            Drop .xlsx or .csv files here
          </div>
        </div>""", unsafe_allow_html=True)

    with ctrl2:
        if st.button("🔄  Sync Now", use_container_width=True, type="primary"):
            with st.spinner("Scanning watch folder..."):
                files_n, bookings_n = live_sync.scan_watch_folder()
            if files_n > 0:
                st.success(f"✓ {files_n} file(s) processed · {bookings_n} new bookings scored")
            else:
                st.info("No new files found. Drop a booking file into the watch folder.")
            st.rerun()

    with ctrl3:
        auto_sync = st.toggle("⚡ Auto-Sync (every 30s)", value=False)

    with ctrl4:
        if st.button("🗑  Clear All Data", use_container_width=True):
            live_sync.clear_all()
            st.warning("All live data cleared.")
            st.rerun()

    # Auto-sync using Streamlit's rerun
    if auto_sync:
        live_sync.scan_watch_folder()
        st.markdown("""
        <div style='background:#edf7f3;border:1px solid #b8e4d0;border-radius:8px;
                    padding:8px 14px;font-size:12px;color:#2ea37a;margin-bottom:16px'>
          ✓ Auto-Sync active — page refreshes every 30 seconds
        </div>""", unsafe_allow_html=True)
        time.sleep(30)
        st.rerun()

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Live stats ────────────────────────────────────────────────────────────
    stats = live_sync.get_live_stats()

    ls1, ls2, ls3, ls4, ls5 = st.columns(5)
    ls1.metric("Total Scored",     stats["total"])
    ls2.metric("🔴 High Risk",     stats["high_risk"])
    ls3.metric("🟠 Medium Risk",   stats["medium_risk"])
    ls4.metric("🟢 Low Risk",      stats["low_risk"])
    ls5.metric("Revenue at Risk",
               f"{currency_symbol}{stats['revenue_at_risk']:,.0f}" if stats['revenue_at_risk'] else "—")

    st.markdown(f"""
    <div style='font-size:11px;color:#6b8899;margin-bottom:20px;margin-top:4px'>
      Last sync: <strong>{stats['last_sync']}</strong>
      &nbsp;·&nbsp; Files processed: <strong>{stats['files']}</strong>
    </div>""", unsafe_allow_html=True)

    # ── Live bookings feed ────────────────────────────────────────────────────
    live_df = live_sync.get_live_bookings(limit=200)

    if live_df.empty:
        st.markdown("""
        <div style='background:#fff;border:2px dashed #e0e6ea;border-radius:14px;
                    padding:48px;text-align:center;'>
          <div style='font-size:32px;margin-bottom:12px'>📂</div>
          <div style='font-weight:700;font-size:16px;color:#1a3c4d;margin-bottom:8px'>
            No live bookings yet
          </div>
          <div style='font-size:13px;color:#6b8899;max-width:400px;margin:0 auto;line-height:1.7'>
            Drop a booking Excel or CSV file into the <strong>watch_folder</strong> directory,
            then click <strong>Sync Now</strong>. Every booking will appear here with a risk score.
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        # Filter controls
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            band_filter = st.multiselect(
                "Filter by risk band",
                ["High Risk", "Medium Risk", "Low Risk"],
                default=["High Risk", "Medium Risk", "Low Risk"]
            )
        with fc2:
            provider_opts = ["All"] + sorted(live_df["provider"].dropna().unique().tolist())
            prov_filter = st.selectbox("Filter by provider", provider_opts)
        with fc3:
            reminded_filter = st.selectbox("Show", ["All bookings", "Not yet reminded", "Already reminded"])

        # Apply filters
        filtered = live_df[live_df["risk_band"].isin(band_filter)]
        if prov_filter != "All":
            filtered = filtered[filtered["provider"] == prov_filter]
        if reminded_filter == "Not yet reminded":
            filtered = filtered[filtered["reminded"] == 0]
        elif reminded_filter == "Already reminded":
            filtered = filtered[filtered["reminded"] == 1]

        st.markdown(f"<div style='font-size:12px;color:#6b8899;margin-bottom:12px'>"
                    f"Showing <strong>{len(filtered)}</strong> bookings</div>", unsafe_allow_html=True)

        # ── High risk alerts (cards) ──────────────────────────────────────────
        high_risk_live = filtered[filtered["risk_band"] == "High Risk"]
        if not high_risk_live.empty:
            st.markdown("#### 🔴 High-Risk Alerts — Action Required")
            for _, row in high_risk_live.head(5).iterrows():
                reminded_badge = (
                    "<span style='background:#edf7f3;color:#2ea37a;font-size:10px;"
                    "font-weight:700;padding:2px 8px;border-radius:20px'>✓ Reminded</span>"
                    if row["reminded"] else
                    "<span style='background:#fdf0f0;color:#e05252;font-size:10px;"
                    "font-weight:700;padding:2px 8px;border-radius:20px'>⚠ Action needed</span>"
                )

                # Build why string
                why = []
                if row.get("lead_time_days", 0) >= 15:
                    why.append(f"booked {row['lead_time_days']} days out")
                if str(row.get("channel","")).lower() == "whatsapp":
                    why.append("WhatsApp channel")
                if str(row.get("patient_type","")).lower() == "new":
                    why.append("new patient")
                if str(row.get("day_of_week","")) in ["Thu","Mon"]:
                    why.append(f"{row.get('day_of_week')} appointment")
                why_str = " · ".join(why) if why else "combination of risk factors"

                st.markdown(f"""
                <div class='rx-card high' style='margin-bottom:10px'>
                  <div style='display:flex;align-items:flex-start;gap:16px'>
                    <div style='background:#fdf0f0;border-radius:10px;min-width:52px;height:52px;
                                display:flex;align-items:center;justify-content:center;
                                font-size:17px;font-weight:800;color:#e05252'>
                      {row['risk_score']}
                    </div>
                    <div style='flex:1'>
                      <div style='display:flex;align-items:center;gap:10px;margin-bottom:4px'>
                        <span style='font-weight:700;font-size:14px;color:#1a3c4d'>
                          {row.get('appointment_id','—')}
                        </span>
                        {reminded_badge}
                        <span style='font-size:11px;color:#6b8899;margin-left:auto'>
                          Scored: {row.get('scored_at','—')[:16]}
                        </span>
                      </div>
                      <div style='font-size:12px;color:#6b8899;margin-bottom:6px'>
                        {row.get('provider','—')} &nbsp;·&nbsp;
                        {row.get('patient_type','—')} patient &nbsp;·&nbsp;
                        {row.get('channel','—')} &nbsp;·&nbsp;
                        {row.get('day_of_week','—')} &nbsp;·&nbsp;
                        Lead: {row.get('lead_time_days','—')}d
                      </div>
                      <div style='font-size:12px;color:#7a3a3a;margin-bottom:6px'>
                        ⚠ Why: {why_str}
                      </div>
                      <div style='font-size:12px;font-weight:600;color:#1a3c4d'>
                        → {row.get('recommended_action','—')}
                      </div>
                    </div>
                  </div>
                </div>""", unsafe_allow_html=True)

                if not row["reminded"]:
                    if st.button(f"✓ Mark as Reminded — {row['appointment_id']}",
                                 key=f"remind_{row['id']}"):
                        live_sync.mark_reminded(row["appointment_id"])
                        st.rerun()

        # ── Full table ────────────────────────────────────────────────────────
        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
        st.markdown("#### All Live Bookings")

        display_cols = [c for c in [
            "scored_at", "appointment_id", "provider", "patient_type",
            "channel", "day_of_week", "lead_time_days", "appointment_type",
            "risk_score", "risk_band", "recommended_action", "source_file"
        ] if c in filtered.columns]

        show_df = filtered[display_cols].rename(columns={
            "scored_at": "Scored At", "appointment_id": "ID",
            "provider": "Provider", "patient_type": "Patient",
            "channel": "Channel", "day_of_week": "Day",
            "lead_time_days": "Lead (d)", "appointment_type": "Type",
            "risk_score": "Risk Score", "risk_band": "Risk Band",
            "recommended_action": "Action", "source_file": "Source"
        })
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # Download
        csv = show_df.to_csv(index=False)
        st.download_button(
            "⬇ Download live feed as CSV",
            data=csv,
            file_name="cadenceworks_live_feed.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # ── Sync log ──────────────────────────────────────────────────────────────
    with st.expander("📋 Sync Activity Log"):
        log_df = live_sync.get_sync_log(limit=30)
        if log_df.empty:
            st.info("No sync activity yet.")
        else:
            st.dataframe(log_df, use_container_width=True, hide_index=True)



# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — REMINDER AGENT
# ════════════════════════════════════════════════════════════════════════════
with tab6:
    section_header("💬 Reminder Agent", "Automated WhatsApp reminders — powered by Twilio", "#3dbfaa")

    cfg        = reminder_agent.load_config()
    twilio_ok  = reminder_agent.is_twilio_configured(cfg)
    r_stats    = reminder_agent.get_reminder_stats()

    # ── Status banner ─────────────────────────────────────────────────────────
    if twilio_ok:
        st.markdown("""
        <div style='background:#edf7f3;border:1px solid #b8e4d0;border-left:4px solid #2ea37a;
                    border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:24px;
                    display:flex;align-items:center;gap:12px'>
          <span style='font-size:24px'>🟢</span>
          <div>
            <div style='font-weight:700;color:#2ea37a;font-size:14px'>Twilio Connected — Agent is LIVE</div>
            <div style='font-size:12px;color:#2a5f52;margin-top:2px'>
              Real WhatsApp messages will be sent to patients. Make sure your templates are correct before running.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style='background:#fdf5ed;border:1px solid #f0d0b0;border-left:4px solid #e8923a;
                    border-radius:0 12px 12px 0;padding:16px 20px;margin-bottom:24px'>
          <div style='font-weight:700;color:#e8923a;font-size:14px;margin-bottom:6px'>
            🟡 Dry Run Mode — Twilio not yet configured
          </div>
          <div style='font-size:13px;color:#7a4f1a;line-height:1.7'>
            The agent will simulate sending reminders and log everything — but no real WhatsApp messages will go out.
            <br>To go live: open <code>config.ini</code> in the cadenceworks folder and paste your Twilio credentials.
          </div>
        </div>""", unsafe_allow_html=True)

        with st.expander("📋 How to get your Twilio credentials (3 steps)"):
            st.markdown("""
            **Step 1** — Go to [twilio.com](https://www.twilio.com) and create a free account (no credit card needed to start)

            **Step 2** — From your Twilio Console, copy:
            - **Account SID** (starts with `AC...`)
            - **Auth Token** (click the eye icon to reveal)

            **Step 3** — Go to **Messaging → Try it out → Send a WhatsApp message** in the Twilio Console.
            Follow the sandbox setup — scan the QR code with your phone to connect.
            You'll get a sandbox number like `+14155238886`.

            Then open `config.ini` in the cadenceworks folder and paste them in:
            ```
            account_sid = ACxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
            auth_token  = your_auth_token_here
            from_number = whatsapp:+14155238886
            ```
            Save the file and refresh this page — the agent goes live immediately.
            """)

    # ── Stats row ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ra1, ra2, ra3, ra4, ra5 = st.columns(5)
    ra1.metric("Queued",        r_stats["queued"],    help="Scheduled but not yet sent")
    ra2.metric("Total Sent",    r_stats["sent"])
    ra3.metric("Live Messages", r_stats["live_sent"], help="Real WhatsApp messages delivered")
    ra4.metric("Dry Runs",      r_stats["dry_runs"],  help="Simulated sends (Twilio not configured)")
    ra5.metric("Failed",        r_stats["failed"],    delta=None)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    # ── Schedule reminders from live bookings ─────────────────────────────────
    st.markdown("#### Schedule Reminders from Live Bookings")

    live_df = live_sync.get_live_bookings(limit=500)

    if live_df.empty:
        st.info("No live bookings yet. Go to the 🔴 Live Monitor tab and sync your bookings first.")
    else:
        high_risk_df = live_df[live_df["risk_band"].isin(["High Risk", "Medium Risk"])]

        st.markdown(f"""
        <div style='font-size:13px;color:#6b8899;margin-bottom:16px'>
          <strong style='color:#1a3c4d'>{len(high_risk_df)}</strong> high/medium risk bookings eligible for reminders
          out of <strong style='color:#1a3c4d'>{len(live_df)}</strong> total live bookings.
        </div>""", unsafe_allow_html=True)

        # Patient number input — in real system this comes from patient DB
        st.markdown("""
        <div style='background:#f0f2f5;border-radius:10px;padding:14px 18px;margin-bottom:16px;font-size:12px;color:#6b8899;line-height:1.7'>
          ℹ In production, patient phone numbers come from your booking system automatically.
          For now, enter a test number below to see the agent in action.
        </div>""", unsafe_allow_html=True)

        test_number = st.text_input(
            "Test patient WhatsApp number",
            placeholder="e.g. 0821234567",
            help="This number will receive all test reminders. Use your own number to test."
        )

        col_sched1, col_sched2 = st.columns(2)
        with col_sched1:
            schedule_btn = st.button(
                "📅 Schedule Reminders for All High-Risk Bookings",
                use_container_width=True,
                type="primary"
            )
        with col_sched2:
            run_now_btn = st.button(
                "▶ Run Agent Now (send due reminders)",
                use_container_width=True
            )

        if schedule_btn:
            if not test_number:
                st.warning("Enter a test number above first.")
            else:
                total_scheduled = 0
                for _, row in high_risk_df.iterrows():
                    # Build appt dict — in production appt_datetime comes from booking system
                    from datetime import datetime, timedelta
                    # Simulate appointment 2 days from now for demo purposes
                    fake_dt = datetime.now() + timedelta(days=2, hours=9)
                    appt = {
                        "appointment_id":   row.get("appointment_id", ""),
                        "patient_number":   test_number,
                        "patient_name":     "Patient",
                        "provider":         row.get("provider", ""),
                        "appt_datetime":    str(fake_dt),
                        "appointment_type": row.get("appointment_type", "Consult"),
                        "risk_score":       row.get("risk_score", 0),
                        "risk_band":        row.get("risk_band", ""),
                    }
                    n = reminder_agent.schedule_reminders(appt, cfg, dry_run=not twilio_ok)
                    total_scheduled += n

                st.success(f"✓ {total_scheduled} reminders scheduled across {len(high_risk_df)} high-risk bookings.")
                st.rerun()

        if run_now_btn:
            with st.spinner("Running Reminder Agent..."):
                actions = reminder_agent.run_once(verbose=False)
            if actions:
                for a in actions:
                    icon  = "✅" if a["status"] == "sent" else ("📋" if a["status"] == "dry_run" else "❌")
                    label = "SENT" if a["status"] == "sent" else ("DRY RUN" if a["status"] == "dry_run" else "FAILED")
                    st.markdown(f"""
                    <div style='background:#fff;border:1px solid #e0e6ea;border-radius:10px;
                                padding:14px 18px;margin-bottom:8px;font-size:13px'>
                      <div style='display:flex;align-items:center;gap:10px;margin-bottom:8px'>
                        <span>{icon}</span>
                        <strong>{a['appointment_id']}</strong>
                        <span style='background:#e8f8f5;color:#2ea393;padding:2px 8px;
                                     border-radius:20px;font-size:11px;font-weight:700'>{a['reminder_type']}</span>
                        <span style='margin-left:auto;font-size:11px;color:#6b8899'>[{label}]</span>
                      </div>
                      <div style='background:#f0f2f5;border-radius:8px;padding:10px 14px;
                                  font-size:12px;color:#1a3c4d;white-space:pre-line;line-height:1.6'>
{a['message']}
                      </div>
                    </div>""", unsafe_allow_html=True)
            else:
                st.info("No reminders due right now. Schedule some reminders first using the button above.")
            st.rerun()

    # ── Reminder queue ────────────────────────────────────────────────────────
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("#### Reminder Queue")

    queue_df = reminder_agent.get_reminder_queue(limit=100)
    if queue_df.empty:
        st.markdown("""
        <div style='background:#fff;border:1px dashed #e0e6ea;border-radius:12px;
                    padding:28px;text-align:center;font-size:13px;color:#6b8899'>
          No reminders scheduled yet. Sync live bookings and click Schedule Reminders above.
        </div>""", unsafe_allow_html=True)
    else:
        # Colour code sent vs pending
        def style_queue(row):
            if row.sent == 1:
                return ["background-color: #edf7f3"] * len(row)
            return [""] * len(row)

        display_q = queue_df.rename(columns={
            "appointment_id": "ID", "patient_name": "Patient",
            "provider": "Provider", "appt_datetime": "Appt Time",
            "risk_score": "Risk", "risk_band": "Band",
            "reminder_type": "Type", "scheduled_for": "Send At",
            "sent": "Sent", "sent_at": "Sent At", "dry_run": "Dry Run"
        })
        st.dataframe(display_q, use_container_width=True, hide_index=True)

    # ── Message preview ───────────────────────────────────────────────────────
    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    with st.expander("📱 Preview Message Templates"):
        practice_name = cfg.get("practice", "name", fallback="My Medical Practice")
        sample_appt = {
            "patient_name":     "Thabo",
            "provider":         "Dr Naidoo",
            "appt_datetime":    str(datetime.now() + timedelta(days=3, hours=9)),
            "appointment_type": "Consultation",
            "hours_until":      "72",
        }
        for template_key, label in [
            ("reminder_72hr", "72-Hour Reminder"),
            ("reminder_24hr", "24-Hour Reminder"),
            ("reminder_4hr",  "4-Hour Reminder"),
        ]:
            msg = reminder_agent.build_message(template_key, cfg, sample_appt)
            st.markdown(f"**{label}**")
            st.markdown(f"""
            <div style='background:#e8f8f5;border-radius:12px;padding:16px 20px;
                        margin-bottom:16px;font-size:13px;white-space:pre-line;
                        line-height:1.7;color:#1a3c4d;border:1px solid #c0e8e0'>
{msg}
            </div>""", unsafe_allow_html=True)

    # ── Reminder log ──────────────────────────────────────────────────────────
    with st.expander("📋 Reminder Activity Log"):
        log_df = reminder_agent.get_reminder_log(limit=50)
        if log_df.empty:
            st.info("No activity yet.")
        else:
            st.dataframe(log_df, use_container_width=True, hide_index=True)

    # ── Config viewer ─────────────────────────────────────────────────────────
    with st.expander("⚙️ Current Configuration (config.ini)"):
        st.markdown(f"""
        ```
        Practice:     {cfg.get('practice', 'name', fallback='—')}
        Currency:     {cfg.get('practice', 'currency', fallback='R')}
        Country code: {cfg.get('practice', 'country_code', fallback='+27')}
        Twilio:       {'✓ Configured' if twilio_ok else '✗ Not configured (dry run mode)'}
        High risk threshold:   {cfg.get('reminder_agent', 'high_risk_threshold', fallback=70)}
        Medium risk threshold: {cfg.get('reminder_agent', 'medium_risk_threshold', fallback=45)}
        High risk schedule:    {cfg.get('reminder_agent', 'high_risk_schedule', fallback='72, 24, 4')} hours
        Medium risk schedule:  {cfg.get('reminder_agent', 'medium_risk_schedule', fallback='24')} hours
        ```
        Edit <code>config.ini</code> in the cadenceworks folder to change any of these settings.
        """, unsafe_allow_html=True)

