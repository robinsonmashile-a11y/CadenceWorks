"""
CadenceWorks — Practice Intelligence Calculator
A lightweight lead-gen tool. No data upload needed.
5 questions → instant analytics report.

Run with: streamlit run calculator.py
"""

import streamlit as st
import math
from datetime import datetime

st.set_page_config(
    page_title="CadenceWorks — Practice Intelligence",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
  --navy:    #1a3c4d;
  --teal:    #3dbfaa;
  --teal-dk: #2ea393;
  --teal-lt: #e8f8f5;
  --amber:   #e8923a;
  --red:     #e05252;
  --green:   #2ea37a;
  --bg:      #f4f6f8;
  --card:    #ffffff;
  --muted:   #7a8f9a;
  --border:  #dde4e8;
}

* { margin:0; padding:0; box-sizing:border-box; }

html, body, [class*="css"] {
  font-family: 'DM Sans', sans-serif !important;
  background: var(--bg) !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
  max-width: 780px !important;
  padding: 0 24px 80px !important;
}

/* ── Top bar ── */
.top-bar {
  background: var(--navy);
  margin: -1rem -24px 0;
  padding: 16px 40px;
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.top-logo {
  font-family: 'Syne', sans-serif;
  font-weight: 800;
  font-size: 17px;
  color: white;
  letter-spacing: -0.3px;
}
.top-logo span { color: var(--teal); }
.top-tag {
  font-size: 11px;
  font-weight: 500;
  color: rgba(255,255,255,0.4);
  letter-spacing: 1.5px;
  text-transform: uppercase;
}

/* ── Hero ── */
.hero {
  text-align: center;
  padding: 56px 20px 40px;
}
.hero-eyebrow {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: var(--teal-lt);
  color: var(--teal-dk);
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  padding: 6px 14px;
  border-radius: 20px;
  margin-bottom: 20px;
}
.hero h1 {
  font-family: 'Syne', sans-serif;
  font-size: clamp(28px, 5vw, 44px);
  font-weight: 800;
  color: var(--navy);
  line-height: 1.1;
  letter-spacing: -1px;
  margin-bottom: 14px;
}
.hero h1 em {
  font-style: normal;
  color: var(--teal);
}
.hero p {
  font-size: 16px;
  color: var(--muted);
  line-height: 1.7;
  max-width: 500px;
  margin: 0 auto;
}

/* ── Form card ── */
.form-card {
  background: var(--card);
  border-radius: 20px;
  padding: 36px 40px;
  border: 1px solid var(--border);
  box-shadow: 0 4px 24px rgba(26,60,77,0.08);
  margin-bottom: 24px;
}
.form-title {
  font-family: 'Syne', sans-serif;
  font-size: 13px;
  font-weight: 700;
  letter-spacing: 1.5px;
  text-transform: uppercase;
  color: var(--teal-dk);
  margin-bottom: 24px;
  display: flex;
  align-items: center;
  gap: 8px;
}
.q-label {
  font-family: 'Syne', sans-serif;
  font-size: 16px;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 4px;
  margin-top: 20px;
}
.q-sub {
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 10px;
}

/* Streamlit input overrides */
.stSlider > div { padding: 0 !important; }
.stSlider [data-testid="stThumbValue"] {
  background: var(--teal) !important;
  color: white !important;
  font-weight: 700 !important;
}
.stSelectbox > div > div, .stMultiSelect > div > div {
  border-radius: 10px !important;
  border-color: var(--border) !important;
}
.stCheckbox label { font-size: 14px !important; color: var(--navy) !important; }
.stRadio label { font-size: 14px !important; color: var(--navy) !important; }

/* ── Results ── */
.results-header {
  text-align: center;
  padding: 40px 0 28px;
}
.results-header h2 {
  font-family: 'Syne', sans-serif;
  font-size: 28px;
  font-weight: 800;
  color: var(--navy);
  margin-bottom: 6px;
}
.results-header p { font-size: 14px; color: var(--muted); }

.metric-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px;
  margin-bottom: 20px;
}
.metric-card {
  background: var(--card);
  border-radius: 16px;
  padding: 22px 24px;
  border: 1px solid var(--border);
  box-shadow: 0 2px 12px rgba(26,60,77,0.06);
}
.metric-label {
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 1px;
  color: var(--muted);
  margin-bottom: 8px;
}
.metric-value {
  font-family: 'Syne', sans-serif;
  font-size: 34px;
  font-weight: 800;
  line-height: 1;
  color: var(--navy);
  margin-bottom: 4px;
}
.metric-value.red    { color: var(--red); }
.metric-value.green  { color: var(--green); }
.metric-value.amber  { color: var(--amber); }
.metric-value.teal   { color: var(--teal-dk); }
.metric-sub {
  font-size: 12px;
  color: var(--muted);
}
.metric-badge {
  display: inline-block;
  font-size: 11px;
  font-weight: 700;
  padding: 3px 9px;
  border-radius: 20px;
  margin-top: 6px;
}
.badge-red   { background:#fdf0f0; color:var(--red); }
.badge-green { background:#edf7f3; color:var(--green); }
.badge-amber { background:#fdf5ed; color:var(--amber); }
.badge-teal  { background:var(--teal-lt); color:var(--teal-dk); }

.insight-card {
  background: var(--card);
  border-radius: 16px;
  padding: 22px 26px;
  border: 1px solid var(--border);
  border-left: 4px solid var(--teal);
  box-shadow: 0 2px 12px rgba(26,60,77,0.06);
  margin-bottom: 12px;
}
.insight-card.warn { border-left-color: var(--amber); }
.insight-card.danger { border-left-color: var(--red); }
.insight-num {
  font-family: 'Syne', sans-serif;
  font-size: 11px;
  font-weight: 700;
  letter-spacing: 1px;
  text-transform: uppercase;
  color: var(--teal-dk);
  margin-bottom: 6px;
}
.insight-num.warn   { color: var(--amber); }
.insight-num.danger { color: var(--red); }
.insight-title {
  font-family: 'Syne', sans-serif;
  font-size: 15px;
  font-weight: 700;
  color: var(--navy);
  margin-bottom: 6px;
}
.insight-body {
  font-size: 13px;
  color: var(--muted);
  line-height: 1.7;
}
.insight-impact {
  display: inline-block;
  margin-top: 10px;
  font-size: 12px;
  font-weight: 600;
  color: var(--green);
  background: #edf7f3;
  padding: 4px 10px;
  border-radius: 20px;
}

.cta-block {
  background: linear-gradient(135deg, var(--navy) 0%, #1e4f66 60%, #24607a 100%);
  border-radius: 20px;
  padding: 40px;
  text-align: center;
  margin-top: 28px;
  position: relative;
  overflow: hidden;
}
.cta-block::before {
  content: '';
  position: absolute;
  top: -60px; right: -60px;
  width: 200px; height: 200px;
  background: var(--teal);
  border-radius: 50%;
  opacity: 0.1;
}
.cta-block h3 {
  font-family: 'Syne', sans-serif;
  font-size: 24px;
  font-weight: 800;
  color: white;
  margin-bottom: 10px;
}
.cta-block p {
  font-size: 14px;
  color: rgba(255,255,255,0.6);
  margin-bottom: 24px;
  line-height: 1.7;
}
.cta-pill {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  background: var(--teal);
  color: white;
  font-family: 'Syne', sans-serif;
  font-size: 14px;
  font-weight: 700;
  padding: 14px 32px;
  border-radius: 99px;
  text-decoration: none;
  letter-spacing: 0.3px;
}

.divider {
  height: 1px;
  background: var(--border);
  margin: 28px 0;
}

.watermark {
  text-align: center;
  font-size: 11px;
  color: var(--muted);
  padding: 20px 0;
}
.watermark strong { color: var(--navy); }

/* Button */
.stButton > button {
  background: var(--teal) !important;
  color: white !important;
  font-family: 'Syne', sans-serif !important;
  font-weight: 700 !important;
  font-size: 15px !important;
  border: none !important;
  border-radius: 12px !important;
  padding: 14px 0 !important;
  width: 100% !important;
  letter-spacing: 0.3px !important;
  transition: all 0.2s !important;
  box-shadow: 0 4px 16px rgba(61,191,170,0.35) !important;
}
.stButton > button:hover {
  background: var(--teal-dk) !important;
  transform: translateY(-1px) !important;
  box-shadow: 0 8px 24px rgba(61,191,170,0.4) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Top bar ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='top-bar'>
  <div class='top-logo'>Cadence<span>Works</span></div>
  <div class='top-tag'>Practice Intelligence Calculator</div>
</div>
""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='hero'>
  <div class='hero-eyebrow'>
    <span style='width:6px;height:6px;background:var(--teal);border-radius:50%;display:inline-block'></span>
    Free Practice Analysis
  </div>
  <h1>See exactly how much<br>revenue your practice<br>is <em>losing to no-shows</em></h1>
  <p>Answer 5 quick questions. Get an instant report showing your revenue at risk, 
  what's driving it, and what to do about it.</p>
</div>
""", unsafe_allow_html=True)


# ── Input form ─────────────────────────────────────────────────────────────────
st.markdown("<div class='form-card'>", unsafe_allow_html=True)
st.markdown("<div class='form-title'>● Tell us about your practice</div>", unsafe_allow_html=True)

# Q1
st.markdown("<div class='q-label'>1. How many doctors are in your practice?</div>", unsafe_allow_html=True)
st.markdown("<div class='q-sub'>Include all full-time and part-time practitioners</div>", unsafe_allow_html=True)
num_doctors = st.slider("", min_value=1, max_value=20, value=2, key="doctors",
                        label_visibility="collapsed")

# Q2
st.markdown("<div class='q-label'>2. How many appointments do you see per day?</div>", unsafe_allow_html=True)
st.markdown("<div class='q-sub'>Across all doctors combined</div>", unsafe_allow_html=True)
appts_per_day = st.slider("", min_value=5, max_value=150, value=24, key="appts",
                           label_visibility="collapsed")

# Q3
st.markdown("<div class='q-label'>3. How do patients book their appointments?</div>", unsafe_allow_html=True)
st.markdown("<div class='q-sub'>Select all that apply</div>", unsafe_allow_html=True)
channels = st.multiselect(
    "",
    ["Phone", "WhatsApp", "Online / App", "Walk-in"],
    default=["Phone", "WhatsApp"],
    key="channels",
    label_visibility="collapsed"
)

# Q4
st.markdown("<div class='q-label'>4. Do you currently send appointment reminders?</div>", unsafe_allow_html=True)
st.markdown("<div class='q-sub'>Any channel — SMS, WhatsApp, phone call</div>", unsafe_allow_html=True)
sends_reminders = st.radio(
    "",
    ["No — we don't send reminders", "Sometimes — but not consistently", "Yes — we always send reminders"],
    key="reminders",
    label_visibility="collapsed"
)

# Q5
st.markdown("<div class='q-label'>5. What's your approximate no-show rate?</div>", unsafe_allow_html=True)
st.markdown("<div class='q-sub'>Patients who booked but didn't arrive and didn't cancel</div>", unsafe_allow_html=True)
noshow_options = {
    "Less than 5% — very low":     3,
    "Around 5–10% — below average": 7,
    "Around 10–15% — typical":     12,
    "Around 15–20% — high":        17,
    "More than 20% — very high":   23,
    "I honestly don't know":       11,
}
noshow_label = st.selectbox("", list(noshow_options.keys()), index=2,
                             key="noshow", label_visibility="collapsed")
noshow_rate = noshow_options[noshow_label] / 100

st.markdown("</div>", unsafe_allow_html=True)  # close form-card

# ── Generate button ────────────────────────────────────────────────────────────
generate = st.button("Generate My Practice Report →")


# ── Results ───────────────────────────────────────────────────────────────────
if generate or st.session_state.get("report_generated"):
    st.session_state["report_generated"] = True

    # ── Calculations ──────────────────────────────────────────────────────────
    working_days_month = 22
    working_days_week  = 5

    appts_month  = appts_per_day * working_days_month
    appts_week   = appts_per_day * working_days_week

    # Average fee estimate — GP context
    avg_fee = 900

    noshow_month    = round(appts_month * noshow_rate)
    revenue_lost    = noshow_month * avg_fee
    revenue_lost_yr = revenue_lost * 12

    # Reminder impact estimate
    reminder_recovery_pct = 0.12 if "No" in sends_reminders else \
                            0.22 if "Sometimes" in sends_reminders else 0.0
    # CadenceWorks improvement over current state
    cw_recovery_pct = 0.35 if "No" in sends_reminders else \
                      0.25 if "Sometimes" in sends_reminders else 0.18

    recoverable_monthly = round(revenue_lost * cw_recovery_pct)
    recoverable_yearly  = recoverable_monthly * 12

    # Channel risk flag
    has_whatsapp = "WhatsApp" in channels
    whatsapp_risk = "WhatsApp bookings typically have 40–70% higher no-show rates than phone bookings." if has_whatsapp else ""

    # Benchmark
    industry_avg = 0.085
    vs_benchmark = noshow_rate - industry_avg
    benchmark_label = "above" if vs_benchmark > 0 else "below"
    benchmark_color = "red" if vs_benchmark > 0.03 else ("amber" if vs_benchmark > 0 else "green")

    # Risk score (simple composite)
    risk_score = 0
    risk_score += min(noshow_rate * 200, 40)
    risk_score += 20 if "No" in sends_reminders else (10 if "Sometimes" in sends_reminders else 0)
    risk_score += 15 if has_whatsapp else 0
    risk_score += 10 if num_doctors >= 3 else 0
    risk_score += 15 if "I honestly don't know" in noshow_label else 0
    risk_score = min(round(risk_score), 100)

    risk_band  = "High" if risk_score >= 65 else ("Medium" if risk_score >= 35 else "Low")
    risk_color = "red"  if risk_score >= 65 else ("amber"  if risk_score >= 35 else "green")

    # ── Results header ─────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='results-header'>
      <h2>Your Practice Intelligence Report</h2>
      <p>Generated {datetime.now().strftime("%-d %B %Y")} &nbsp;·&nbsp;
         {num_doctors} doctor{"s" if num_doctors > 1 else ""} &nbsp;·&nbsp;
         {appts_per_day} appointments/day</p>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI grid ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='metric-grid'>
      <div class='metric-card'>
        <div class='metric-label'>Monthly Revenue at Risk</div>
        <div class='metric-value red'>R{revenue_lost:,.0f}</div>
        <div class='metric-sub'>{noshow_month} no-shows × R{avg_fee:,} avg fee</div>
        <span class='metric-badge badge-red'>Per month</span>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Annual Revenue at Risk</div>
        <div class='metric-value red'>R{revenue_lost_yr:,.0f}</div>
        <div class='metric-sub'>If no-show rate stays at {round(noshow_rate*100)}%</div>
        <span class='metric-badge badge-red'>Per year</span>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Your No-Show Rate</div>
        <div class='metric-value {benchmark_color}'>{round(noshow_rate*100)}%</div>
        <div class='metric-sub'>Industry average: {round(industry_avg*100)}%</div>
        <span class='metric-badge badge-{"red" if benchmark_color == "red" else ("amber" if benchmark_color == "amber" else "green")}'>
          {round(abs(vs_benchmark)*100, 1)}% {benchmark_label} average
        </span>
      </div>
      <div class='metric-card'>
        <div class='metric-label'>Recoverable with CadenceWorks</div>
        <div class='metric-value green'>R{recoverable_monthly:,.0f}</div>
        <div class='metric-sub'>Estimated monthly recovery</div>
        <span class='metric-badge badge-green'>R{recoverable_yearly:,.0f}/year</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Practice risk score ────────────────────────────────────────────────────
    bar_pct = risk_score
    bar_color = "#e05252" if risk_score >= 65 else ("#e8923a" if risk_score >= 35 else "#2ea37a")
    st.markdown(f"""
    <div class='metric-card' style='margin-bottom:20px'>
      <div style='display:flex;justify-content:space-between;align-items:center;margin-bottom:12px'>
        <div>
          <div class='metric-label'>Practice Risk Score</div>
          <div style='font-family:Syne,sans-serif;font-size:13px;color:var(--muted)'>
            Based on your no-show rate, reminder status and booking channels
          </div>
        </div>
        <div style='font-family:Syne,sans-serif;font-size:42px;font-weight:800;color:{bar_color}'>
          {risk_score}
        </div>
      </div>
      <div style='height:10px;background:#f0f2f5;border-radius:99px;overflow:hidden'>
        <div style='width:{bar_pct}%;height:100%;background:{bar_color};border-radius:99px;
                    transition:width 1s ease'></div>
      </div>
      <div style='display:flex;justify-content:space-between;font-size:11px;
                  color:var(--muted);margin-top:6px'>
        <span>Low Risk</span><span>Medium Risk</span><span>High Risk</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Insights ───────────────────────────────────────────────────────────────
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                letter-spacing:1.5px;text-transform:uppercase;color:var(--teal-dk);
                margin-bottom:14px'>What's driving your no-shows</div>
    """, unsafe_allow_html=True)

    # Insight 1 — Reminders
    if "No" in sends_reminders:
        st.markdown(f"""
        <div class='insight-card danger'>
          <div class='insight-num danger'>● Biggest opportunity</div>
          <div class='insight-title'>You're not sending reminders — this is your #1 fix</div>
          <div class='insight-body'>
            Practices without reminders see 30–45% higher no-show rates than those with automated sequences.
            A simple 24-hour WhatsApp reminder alone reduces no-shows by 20–30%.
            You're leaving approximately <strong>R{round(revenue_lost*0.28):,}/month</strong> on the table from this alone.
          </div>
          <span class='insight-impact'>↑ Est. R{round(revenue_lost*0.28):,}/month recovery with reminders</span>
        </div>""", unsafe_allow_html=True)
    elif "Sometimes" in sends_reminders:
        st.markdown(f"""
        <div class='insight-card warn'>
          <div class='insight-num warn'>● High impact</div>
          <div class='insight-title'>Inconsistent reminders are almost as bad as none</div>
          <div class='insight-body'>
            Patients who receive reminders are 3× more likely to attend or give advance notice if they can't.
            Automating your reminder sequence — so every patient gets it every time — is the highest-ROI 
            change you can make right now.
          </div>
          <span class='insight-impact'>↑ Est. R{round(revenue_lost*0.22):,}/month recovery from consistent reminders</span>
        </div>""", unsafe_allow_html=True)

    # Insight 2 — WhatsApp
    if has_whatsapp:
        st.markdown(f"""
        <div class='insight-card warn'>
          <div class='insight-num warn'>● Channel risk</div>
          <div class='insight-title'>WhatsApp bookings carry significantly higher no-show risk</div>
          <div class='insight-body'>
            {whatsapp_risk} WhatsApp patients tend to book impulsively and cancel without notifying the practice.
            Protecting your prime morning and lunchtime slots for phone-booked and returning patients 
            can reduce prime slot no-shows by up to 35%.
          </div>
          <span class='insight-impact'>↑ Protect your highest-value slots from highest-risk bookings</span>
        </div>""", unsafe_allow_html=True)

    # Insight 3 — Scale
    appts_at_risk_week = round(appts_week * noshow_rate)
    st.markdown(f"""
    <div class='insight-card'>
      <div class='insight-num'>● Volume impact</div>
      <div class='insight-title'>At your scale, every 1% improvement is significant</div>
      <div class='insight-body'>
        With {appts_per_day} appointments per day across {num_doctors} doctor{"s" if num_doctors > 1 else ""},
        you have approximately <strong>{appts_at_risk_week} no-shows per week</strong>.
        Reducing your no-show rate by just 3 percentage points recovers
        <strong>R{round(appts_week * 0.03 * avg_fee):,}/week</strong> — 
        R{round(appts_week * 0.03 * avg_fee * 52):,} over a year.
      </div>
      <span class='insight-impact'>↑ R{round(appts_week * 0.03 * avg_fee * 52):,}/year from a 3% improvement</span>
    </div>""", unsafe_allow_html=True)

    # Insight 4 — Benchmark
    if vs_benchmark > 0.02:
        st.markdown(f"""
        <div class='insight-card danger'>
          <div class='insight-num danger'>● Benchmark gap</div>
          <div class='insight-title'>You're {round(vs_benchmark*100, 1)}% above the industry average</div>
          <div class='insight-body'>
            GP practices using structured reminder and waitlist systems average {round(industry_avg*100)}% no-shows.
            Your current rate of {round(noshow_rate*100)}% suggests there are specific operational patterns —
            booking channel mix, slot allocation, or lead time — creating avoidable gaps.
            A full CadenceWorks analysis would identify exactly which ones.
          </div>
          <span class='insight-impact'>↑ Getting to industry average saves R{round((noshow_rate - industry_avg) * appts_month * avg_fee):,}/month</span>
        </div>""", unsafe_allow_html=True)

    # ── What CadenceWorks does ─────────────────────────────────────────────────
    st.markdown("""<div class='divider'></div>""", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                letter-spacing:1.5px;text-transform:uppercase;color:var(--teal-dk);
                margin-bottom:14px'>What CadenceWorks does for your practice</div>
    """, unsafe_allow_html=True)

    cols_txt = [
        ("📊", "Analyses your full booking history", "Identifies every pattern driving your no-shows — by channel, day, patient type and slot."),
        ("🤖", "Scores every new booking for risk", "AI assigns a 0–100 risk score to each appointment before it happens."),
        ("💬", "Sends automated WhatsApp reminders", "High-risk patients get a 72hr, 24hr and 4hr reminder sequence. Automatically."),
        ("🎯", "Fills cancelled slots instantly", "Waitlist patients are contacted the moment a slot opens. No manual work."),
    ]

    for icon, title, body in cols_txt:
        st.markdown(f"""
        <div style='display:flex;gap:14px;align-items:flex-start;margin-bottom:14px;
                    background:var(--card);border-radius:12px;padding:16px 18px;
                    border:1px solid var(--border)'>
          <span style='font-size:22px;flex-shrink:0'>{icon}</span>
          <div>
            <div style='font-family:Syne,sans-serif;font-weight:700;font-size:14px;
                        color:var(--navy);margin-bottom:4px'>{title}</div>
            <div style='font-size:13px;color:var(--muted);line-height:1.6'>{body}</div>
          </div>
        </div>""", unsafe_allow_html=True)

    # ── ROI summary ────────────────────────────────────────────────────────────
    monthly_cost = 12999
    monthly_roi  = recoverable_monthly - monthly_cost
    roi_ratio    = round(recoverable_monthly / monthly_cost, 1)

    st.markdown(f"""
    <div style='background:var(--teal-lt);border-radius:16px;padding:24px 28px;
                border:1px solid #c0e8e0;margin:20px 0'>
      <div style='font-family:Syne,sans-serif;font-size:11px;font-weight:700;
                  letter-spacing:1.5px;text-transform:uppercase;color:var(--teal-dk);
                  margin-bottom:12px'>Your estimated ROI</div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;text-align:center'>
        <div>
          <div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;color:var(--navy)'>
            R{recoverable_monthly:,}
          </div>
          <div style='font-size:12px;color:var(--muted)'>Estimated monthly recovery</div>
        </div>
        <div>
          <div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;color:var(--navy)'>
            R{monthly_cost:,}
          </div>
          <div style='font-size:12px;color:var(--muted)'>CadenceWorks monthly fee</div>
        </div>
        <div>
          <div style='font-family:Syne,sans-serif;font-size:26px;font-weight:800;
                      color:{"var(--green)" if monthly_roi > 0 else "var(--red)"}'>
            {roi_ratio}×
          </div>
          <div style='font-size:12px;color:var(--muted)'>Return on investment</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── CTA ────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='cta-block'>
      <h3>Ready to recover R{recoverable_monthly:,}/month?</h3>
      <p>
        Get a full analysis of your actual booking data — not estimates.<br>
        CadenceWorks connects to your booking system and starts working in under 24 hours.
      </p>
      <a href='https://calendly.com/robinson-mashile-cadenceworksconsulting/30min' target='_blank'
         class='cta-pill'>
        Book a free demo →
      </a>
    </div>
    """, unsafe_allow_html=True)

    # ── Watermark ──────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class='watermark'>
      <strong>CadenceWorks</strong> Consulting &nbsp;·&nbsp;
      These estimates are based on industry averages and your inputs.
      Actual results vary by practice. &nbsp;·&nbsp;
      Generated {datetime.now().strftime("%-d %B %Y")}
    </div>
    """, unsafe_allow_html=True)
