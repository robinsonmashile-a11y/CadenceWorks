
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, time

st.set_page_config(
    page_title="CadenceWorks | Appointment Leakage Intelligence",
    page_icon="📈",
    layout="wide",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    .cw-header {
        padding: 18px 22px;
        border-radius: 16px;
        background: linear-gradient(90deg, #0B2F4A 0%, #1F4E79 55%, #2E86AB 100%);
        color: white;
        margin-bottom: 14px;
    }
    .cw-header h1 { margin: 0; font-size: 28px; }
    .cw-header p { margin: 6px 0 0 0; opacity: 0.9; }
    .cw-card {
        border-radius: 16px;
        padding: 16px 16px 12px 16px;
        background: #ffffff;
        border: 1px solid rgba(0,0,0,0.08);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
        height: 100%;
    }
    .cw-card h3 { margin: 0 0 6px 0; font-size: 14px; color: #0B2F4A; }
    .cw-big { font-size: 28px; font-weight: 700; color: #0B2F4A; margin: 0; }
    .cw-sub { font-size: 12px; color: rgba(0,0,0,0.62); margin-top: 4px; }
    .cw-pill {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 999px;
        font-size: 12px;
        background: rgba(46, 134, 171, 0.12);
        color: #0B2F4A;
        border: 1px solid rgba(46, 134, 171, 0.25);
        margin-right: 6px;
    }
    .cw-section-title {
        font-size: 18px;
        font-weight: 700;
        color: #0B2F4A;
        margin: 8px 0 8px 0;
    }
    .cw-note {
        border-radius: 14px;
        padding: 12px 14px;
        background: rgba(255, 242, 204, 0.65);
        border: 1px solid rgba(245, 158, 11, 0.25);
        color: #3b2f14;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Utilities ----------
FAIL_STATUSES = set(["no-show", "no show", "noshow", "late cancel", "late cancellation", "late_cancel", "latecancel"])

def is_fail(status) -> bool:
    return str(status).strip().lower() in FAIL_STATUSES

def money(x):
    try:
        return f"R{int(round(float(x),0)):,}".replace(",", " ")
    except Exception:
        return "R0"

def pct(x):
    try:
        return f"{float(x)*100:.1f}%"
    except Exception:
        return "0.0%"

def ensure_datetime(series):
    return pd.to_datetime(series, errors="coerce")

def load_excel(xlsx_bytes_or_path):
    appts = pd.read_excel(xlsx_bytes_or_path, sheet_name="Appointments_Raw")
    enq = pd.read_excel(xlsx_bytes_or_path, sheet_name="Enquiries_Unmet")
    # Optional sheets
    impact = None
    try:
        impact = pd.read_excel(xlsx_bytes_or_path, sheet_name="Impact_Model", header=None)
    except Exception:
        impact = None
    return appts, enq, impact

def compute_metrics(appts: pd.DataFrame, enq: pd.DataFrame):
    appts = appts.copy()
    appts["Appt_Datetime"] = ensure_datetime(appts.get("Appt_Datetime"))
    appts["Hour"] = appts["Appt_Datetime"].dt.hour
    appts["Failure"] = appts["Status"].apply(is_fail)
    appts["Fee_Scheduled"] = pd.to_numeric(appts.get("Fee_Scheduled"), errors="coerce").fillna(0)
    appts["Lead_Time_Days"] = pd.to_numeric(appts.get("Lead_Time_Days"), errors="coerce")
    appts["Is_Prime_Slot"] = appts.get("Is_Prime_Slot").astype(str).str.strip().str.upper().isin(["Y","YES","TRUE","1"])
    appts["DayOfWeek"] = appts.get("DayOfWeek").astype(str).str.strip()

    total = len(appts)
    fails = int(appts["Failure"].sum())
    fail_rate = fails / total if total else 0.0
    lost_total = float(appts.loc[appts["Failure"], "Fee_Scheduled"].sum())

    prime = appts[appts["Is_Prime_Slot"]]
    prime_total = len(prime)
    prime_fails = int(prime["Failure"].sum())
    prime_fail_rate = prime_fails / prime_total if prime_total else 0.0
    lost_prime = float(prime.loc[prime["Failure"], "Fee_Scheduled"].sum())

    # Monday morning
    mon = appts[appts["DayOfWeek"].isin(["Mon","Monday"])]
    mon_morn = mon[(mon["Appt_Datetime"].dt.time < time(12,0))]
    mon_morn_total = len(mon_morn)
    mon_morn_fail_rate = (mon_morn["Failure"].sum() / mon_morn_total) if mon_morn_total else 0.0
    lost_mon_morn = float(mon_morn.loc[mon_morn["Failure"], "Fee_Scheduled"].sum())

    # Risk segment: New + lead > 10
    risk = appts[(appts.get("Patient_Type").astype(str).str.lower().str.strip() == "new") & (appts["Lead_Time_Days"] > 10)]
    risk_total = len(risk)
    risk_fail_rate = (risk["Failure"].sum() / risk_total) if risk_total else 0.0
    lost_risk = float(risk.loc[risk["Failure"], "Fee_Scheduled"].sum())

    # Enquiries
    enq = enq.copy()
    unmet_total = len(enq)
    lost_to_comp = int((enq.get("Outcome").astype(str).str.lower().str.strip() == "lost_to_competitor").sum())
    double_loss = int((enq.get("Double_Loss_Flag").astype(str).str.upper().str.strip() == "Y").sum())
    double_loss_rate = double_loss / unmet_total if unmet_total else 0.0

    return {
        "appts": appts,
        "enq": enq,
        "total_appts": total,
        "fails": fails,
        "fail_rate": fail_rate,
        "lost_total": lost_total,
        "prime_total": prime_total,
        "prime_fails": prime_fails,
        "prime_fail_rate": prime_fail_rate,
        "lost_prime": lost_prime,
        "mon_morn_total": mon_morn_total,
        "mon_morn_fail_rate": mon_morn_fail_rate,
        "lost_mon_morn": lost_mon_morn,
        "risk_total": risk_total,
        "risk_fail_rate": risk_fail_rate,
        "lost_risk": lost_risk,
        "unmet_total": unmet_total,
        "lost_to_comp": lost_to_comp,
        "double_loss": double_loss,
        "double_loss_rate": double_loss_rate,
    }

def kpi_card(title, value, subtitle=""):
    st.markdown(
        f"""
        <div class="cw-card">
            <h3>{title}</h3>
            <p class="cw-big">{value}</p>
            <div class="cw-sub">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------- Sidebar ----------
st.sidebar.markdown("## 📁 Data Source")
default_file = "CadenceWorks_Sample_Booking_Month_WITH_ADJUSTMENTS_AND_UPLIFT.xlsx"

uploaded = st.sidebar.file_uploader("Upload your Excel workbook", type=["xlsx"])
use_default = st.sidebar.toggle("Use default sample workbook (if available)", value=True)

xlsx_source = None
if uploaded is not None:
    xlsx_source = uploaded
else:
    if use_default and os.path.exists(default_file):
        xlsx_source = default_file

st.sidebar.markdown("---")
st.sidebar.markdown("## ⚙️ Impact assumptions")
prime_recovery = st.sidebar.slider("Prime leakage recovery %", 0, 70, 35, 5) / 100.0
retain_pct = st.sidebar.slider("Retain lost-to-competitor % (optional)", 0, 50, 20, 5) / 100.0
avg_retain_rev = st.sidebar.number_input("Avg revenue per retained patient (1st visit)", min_value=0, value=900, step=50)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🧭 View")
show_debug = st.sidebar.toggle("Show data diagnostics", value=False)

# ---------- Main ----------
st.markdown(
    """
    <div class="cw-header">
        <h1>CadenceWorks | Appointment Leakage Intelligence</h1>
        <p>Board-ready visibility into no-shows / late cancels, blocked demand, lead-time risk, and recoverable revenue — without replacing the clinic’s booking system.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if xlsx_source is None:
    st.info("Upload your Excel workbook, or toggle 'Use default sample workbook' and place the sample .xlsx next to app.py.")
    st.stop()

appts, enq, impact_raw = load_excel(xlsx_source)
M = compute_metrics(appts, enq)

# KPI Row
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total appointments", f"{M['total_appts']}", "Booked slots in the month")
with c2:
    kpi_card("Overall slot failure", pct(M["fail_rate"]), f"{M['fails']} failed slots")
with c3:
    kpi_card("Prime slot failure", pct(M["prime_fail_rate"]), f"{M['prime_fails']} failed prime slots")
with c4:
    kpi_card("Estimated leakage (fails)", money(M["lost_total"]), f"Prime leakage: {money(M['lost_prime'])}")

tabs = st.tabs(["Executive Summary", "Insights", "Impact & Revenue Potential", "Data"])

# ---------- Executive Summary ----------

# ---------- Executive Summary ----------
with tabs[0]:
    st.markdown('<div class="cw-section-title">Executive Summary</div>', unsafe_allow_html=True)

    st.markdown(
        """
        This pack explains **where revenue leakage is occurring**, **why it is happening**, and **what operational controls will recover it** —
        without replacing the clinic’s booking system.
        """
    )

    left, right = st.columns([1.2, 0.9])

    with left:
        st.markdown("### The story (what’s happening → why → what to do)")

        st.markdown(
            f"""
            **What’s happening:** your calendar can look “fully booked” while still leaking revenue.
            The leakage is not evenly distributed — it concentrates in **prime windows** (high‑demand times)
            and in **predictable risk segments** (e.g., longer lead times).

            **Why this matters:** when a prime slot is booked weeks ahead, it blocks other patients who need that time.
            If the original booking then **no‑shows** or **cancels late**, the clinic takes a **double hit**:
            1) the slot becomes dead time, and 2) the displaced patient often books elsewhere.

            **What the data is saying this month:**  
            - Prime slot failure rate: **{pct(M['prime_fail_rate'])}** (vs overall **{pct(M['fail_rate'])}**)  
            - Monday morning failure rate: **{pct(M['mon_morn_fail_rate'])}**  
            - Estimated leakage from failed appointments (month): **{money(M['lost_total'])}**  
            - Prime leakage portion: **{money(M['lost_prime'])}**  
            - Unmet enquiries lost to competitors: **{M['lost_to_comp']}** (double‑loss events: **{M['double_loss']}**)  
            """
        )

        st.markdown("### Example scenario (why patients are lost even when capacity exists)")
        st.markdown(
            """
            **Monday 09:30** is a high‑value appointment window.  
            A patient books it **12 days in advance**. The schedule now looks full, so a second patient who *only* can do Monday mornings
            is told **“next available is later in the week”** and goes to another practice.

            On Monday, the original patient **does not show** (or cancels late).  
            The clinic loses the slot **and** loses the second patient — even though the clinic *could* have treated them.
            """
        )

    with right:
        st.markdown("### Executive actions (what we recommend — and why it works)")

        with st.expander("1) Peak Slot Confirmation Gate (targeted, not annoying)", expanded=True):
            st.markdown(
                """
                **Why:** longer lead-time bookings have higher plan‑volatility. Prime windows are the most costly place for that volatility to show up.  
                **What to implement:** apply confirmation only to **prime slots** and **high‑risk segments** (e.g., >10‑day lead time and/or new patients).  
                **How it works:** 24h before the appointment, request confirmation. If not confirmed by cutoff → mark **At‑Risk** and trigger backfill.
                **Result:** converts uncertain inventory into certain inventory before the slot expires.
                """
            )

        with st.expander("2) Standby Backfill Playbook (turn risk into recovered revenue)", expanded=True):
            st.markdown(
                """
                **Why:** the goal is not “zero cancellations” — it’s to ensure cancelled/at‑risk slots don’t become dead time.  
                **What to implement:** maintain a standby list of patients who want prime times.  
                **How it works:** At‑Risk slots are flagged the day before (or early morning) → reception activates standby list → slot is filled.
                **Result:** cancellations become **same‑day recovered revenue**, not empty capacity.
                """
            )

        with st.expander("3) Repeat No‑Show Rules (protect prime inventory)", expanded=False):
            st.markdown(
                """
                **Why:** a small subset of patients often drives a disproportionate share of failures. Treating everyone the same harms reliable patients.  
                **What to implement:** after 1–2 failures, restrict access to prime slots (same‑week bookings and/or deposits for prime windows).  
                **Result:** reduces repeat leakage while preserving access for reliable patients.
                """
            )

        st.markdown('<div class="cw-note"><b>Privacy by design:</b> this use case can run on appointment metadata only (no patient notes, diagnoses, or identifiers required). If AI is used, it runs on de‑identified data.</div>', unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("### Visual evidence (where leakage concentrates)")
    v1, v2 = st.columns([1, 1])

    with v1:
        st.markdown("#### Lead-time risk")
        df = M["appts"].copy()
        df = df.dropna(subset=["Lead_Time_Days"])

        def lead_bucket(x):
            if x <= 3:
                return "0–3 days"
            if x <= 10:
                return "4–10 days"
            return ">10 days"

        df["Lead_Bucket"] = df["Lead_Time_Days"].apply(lead_bucket)
        lead = df.groupby("Lead_Bucket").agg(Total=("Failure", "count"), Failures=("Failure", "sum")).reset_index()
        lead["Failure_Rate"] = np.where(lead["Total"] > 0, lead["Failures"] / lead["Total"], 0)
        fig = px.bar(
            lead,
            x="Lead_Bucket",
            y="Failure_Rate",
            text=lead["Failure_Rate"].map(lambda x: f"{x*100:.1f}%"),
            title="Failure Rate by Booking Lead Time",
        )
        fig.update_yaxes(tickformat=".0%")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Interpretation: as lead time increases, plan volatility increases. Use targeted confirmation gates for long‑lead bookings in prime windows."
        )

    with v2:
        st.markdown("#### Day × Hour failure intensity")
        dfh = M["appts"].dropna(subset=["DayOfWeek", "Hour"]).copy()
        heat = (
            dfh.groupby(["DayOfWeek", "Hour"])
            .agg(Total=("Failure", "count"), Failures=("Failure", "sum"))
            .reset_index()
        )
        heat["Failure_Rate"] = np.where(heat["Total"] > 0, heat["Failures"] / heat["Total"], 0)
        pivot = heat.pivot(index="DayOfWeek", columns="Hour", values="Failure_Rate").fillna(0)

        fig2 = go.Figure(data=go.Heatmap(z=pivot.values, x=pivot.columns, y=pivot.index))
        fig2.update_layout(title="Failure Rate Heatmap (Day × Hour)", height=420, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Interpretation: failure clusters create predictable ‘weak windows’ (e.g., Monday mornings). Prioritise controls in these windows first."
        )

# ---------- Insights ----------

with tabs[1]:
    st.markdown('<div class="cw-section-title">Insights</div>', unsafe_allow_html=True)

    a, b = st.columns([1,1])
    with a:
        st.markdown("### Prime vs non-prime")
        dfp = M["appts"].copy()
        dfp["Prime"] = np.where(dfp["Is_Prime_Slot"], "Prime", "Non-prime")
        s = dfp.groupby("Prime").agg(Total=("Failure","count"), Failures=("Failure","sum"), Leakage=("Fee_Scheduled", lambda x: float(x[dfp.loc[x.index, "Failure"]].sum()) )).reset_index()
        s["Failure_Rate"] = np.where(s["Total"]>0, s["Failures"]/s["Total"], 0)
        fig3 = px.bar(s, x="Prime", y="Failure_Rate", text=s["Failure_Rate"].map(lambda x: f"{x*100:.1f}%"),
                      title="Failure Rate: Prime vs Non-prime")
        fig3.update_yaxes(tickformat=".0%")
        fig3.update_traces(textposition="outside")
        st.plotly_chart(fig3, use_container_width=True)

    with b:
        st.markdown("### Risk segment: New + lead time >10 days")
        risk_total = M["risk_total"]
        st.metric("Risk segment volume (appointments)", f"{risk_total}")
        st.metric("Risk segment failure rate", pct(M["risk_fail_rate"]))
        st.metric("Risk segment leakage (fails)", money(M["lost_risk"]))
        st.caption("Use this to justify targeted confirmation policies (avoid burdening all patients).")

    st.markdown("---")
    st.markdown("### Drill-down (filter)")
    f1, f2, f3 = st.columns(3)
    with f1:
        day_sel = st.multiselect("Day of week", sorted(M["appts"]["DayOfWeek"].dropna().unique().tolist()), default=[])
    with f2:
        prime_sel = st.selectbox("Prime filter", ["All","Prime only","Non-prime only"], index=0)
    with f3:
        fail_sel = st.selectbox("Status filter", ["All","Failures only"], index=0)

    dff = M["appts"].copy()
    if day_sel:
        dff = dff[dff["DayOfWeek"].isin(day_sel)]
    if prime_sel == "Prime only":
        dff = dff[dff["Is_Prime_Slot"]]
    elif prime_sel == "Non-prime only":
        dff = dff[~dff["Is_Prime_Slot"]]
    if fail_sel == "Failures only":
        dff = dff[dff["Failure"]]

    st.dataframe(dff.sort_values("Appt_Datetime", ascending=True).head(250), use_container_width=True, height=420)

# ---------- Impact & Revenue Potential ----------
with tabs[2]:
    st.markdown('<div class="cw-section-title">Impact & Revenue Potential</div>', unsafe_allow_html=True)

    st.markdown(
        """
        This is a **recoverable leakage model**, not a growth forecast.
        We estimate month‑level uplift by (1) reducing failed **prime** slots through targeted confirmation/backfill, and (2) optionally retaining
        a conservative portion of demand that would otherwise go to competitors.
        """
    )
    # Baselines
    baseline_prime = M["lost_prime"]
    recovered_prime = baseline_prime * prime_recovery
    retained_rev = M["lost_to_comp"] * retain_pct * float(avg_retain_rev)
    total_uplift = recovered_prime + retained_rev

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Prime leakage baseline", money(baseline_prime), "From failed prime slots (scheduled fees)")
    with k2:
        kpi_card("Recovered (prime)", money(recovered_prime), f"Assumption: {pct(prime_recovery)} recovery")
    with k3:
        kpi_card("Retained demand (optional)", money(retained_rev), f"{pct(retain_pct)} of lost enquiries × avg revenue")
    with k4:
        kpi_card("Total potential uplift", money(total_uplift), "Month-level uplift (modelled)")

    st.markdown("---")
    st.markdown("### Scenario table (editable assumptions in sidebar)")
    scenarios = pd.DataFrame({
        "Scenario": ["Conservative", "Base", "Aggressive"],
        "Prime recovery %": [0.20, 0.35, 0.50],
    })
    scenarios["Recovered prime revenue"] = scenarios["Prime recovery %"] * baseline_prime
    scenarios["Retained-demand revenue (optional)"] = retained_rev
    scenarios["Total uplift"] = scenarios["Recovered prime revenue"] + scenarios["Retained-demand revenue (optional)"]
    scenarios_display = scenarios.copy()
    scenarios_display["Prime recovery %"] = scenarios_display["Prime recovery %"].map(pct)
    for col in ["Recovered prime revenue","Retained-demand revenue (optional)","Total uplift"]:
        scenarios_display[col] = scenarios_display[col].map(money)
    st.dataframe(scenarios_display, use_container_width=True, hide_index=True)

    st.markdown("### Recommended adjustments (operational controls)")
    st.markdown(
        """
        - **Peak Slot Confirmation Gate (targeted):** Apply to prime slots booked far ahead and/or to high-risk segments.  
        - **Standby Backfill Protocol:** Standby list + early activation for At-Risk slots.  
        - **Repeat No-Show Rules:** After repeated non-attendance, constrain peak booking access or require deposit.  
        - **Governance KPIs:** Prime failure rate, Monday morning failure rate, backfilled slots, and recovered revenue.
        """
    )
    st.caption("Note: This model is intentionally conservative. It estimates uplift from reducing failed prime slots and optionally retaining a portion of unmet demand.")

# ---------- Data ----------
with tabs[3]:
    st.markdown('<div class="cw-section-title">Data</div>', unsafe_allow_html=True)

    st.markdown("#### Appointments_Raw")
    st.dataframe(appts.head(200), use_container_width=True, height=320)

    st.markdown("#### Enquiries_Unmet")
    st.dataframe(enq.head(200), use_container_width=True, height=260)

    if show_debug:
        st.markdown("#### Diagnostics")
        st.write("Columns (Appointments):", list(appts.columns))
        st.write("Columns (Enquiries):", list(enq.columns))
        st.write("Nulls (Appointments):")
        st.write(appts.isna().sum())
        st.write("Nulls (Enquiries):")
        st.write(enq.isna().sum())
