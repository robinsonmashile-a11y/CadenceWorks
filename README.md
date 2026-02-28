# CadenceWorks Analytics Engine

AI-powered booking analytics — descriptive, predictive and prescriptive.

## Quick Start

### Step 1 — Install dependencies (once only)
```bash
pip install -r requirements.txt
```

### Step 2 — Run the app
```bash
streamlit run app.py
```
The app opens automatically in your browser at `http://localhost:8501`

### One-click launchers
- **Mac/Linux:** double-click `start.sh` or run `bash start.sh`
- **Windows:** double-click `start.bat`

---

## How it works

1. Upload any Excel (.xlsx) or CSV booking file via the sidebar
2. The engine auto-detects and maps your columns
3. Three analytics layers run automatically:
   - **Descriptive** — KPIs, no-show rates by channel, day, lead time, provider
   - **Predictive** — AI risk scores every appointment 0–100
   - **Prescriptive** — Prioritised recommendations + agent actions
4. Explore results across the three tabs

## Project Structure

```
cadenceworks/
  app.py                  ← Streamlit app (run this)
  requirements.txt        ← Python dependencies
  start.sh                ← Mac/Linux launcher
  start.bat               ← Windows launcher
  engine/
    ingestor.py           ← Data loading & standardisation
    descriptive.py        ← KPIs and breakdowns
    predictive.py         ← ML risk scoring model
    prescriptive.py       ← Recommendations engine
    dashboard.py          ← HTML report generator (optional)
  run.py                  ← CLI runner (alternative to Streamlit)
```

## Supported column names

The engine auto-maps common booking system column names. Supported aliases include:

| Field | Accepted column names |
|---|---|
| Status | Status, outcome, appt_status |
| Provider | Provider, doctor, practitioner |
| Channel | Channel, booking_channel, source |
| Fee | Fee_Scheduled, fee, amount, price |
| Lead Time | Lead_Time_Days, lead_time, days_advance |
| Patient Type | Patient_Type, client_type |

---

Built by **CadenceWorks Consulting** · Descriptive · Predictive · Prescriptive
