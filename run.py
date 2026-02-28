"""
CadenceWorks Analytics Engine
Main Pipeline Runner

Usage:
    python run.py <path_to_excel_or_csv>

Example:
    python run.py bookings.xlsx
    python run.py data/january_bookings.csv

The engine will:
  1. Ingest and standardise the file
  2. Run descriptive analytics
  3. Train and apply the predictive risk model
  4. Generate prescriptive recommendations
  5. Render and save the HTML dashboard
"""

import sys
import json
from pathlib import Path

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent))

from engine import ingestor, descriptive, predictive, prescriptive, dashboard


def run_pipeline(filepath: str, output_dir: str = "output") -> dict:
    """
    Run the full CadenceWorks analytics pipeline on a booking file.
    Returns a summary dict and saves the HTML report to output_dir.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("  CadenceWorks Analytics Engine")
    print(f"{'='*60}")

    # ── Layer 1: Ingest ───────────────────────────────────────────────────────
    print(f"\n[1/4] Ingesting: {filepath}")
    df, meta = ingestor.ingest(filepath)
    print(f"      ✓ {meta['raw_rows']} rows loaded")
    print(f"      ✓ Date range: {meta['date_range']}")
    print(f"      ✓ Providers: {', '.join(meta['providers'])}")
    if meta["missing_udm"]:
        print(f"      ⚠ Missing fields (will be derived or skipped): {meta['missing_udm']}")

    # ── Layer 2: Descriptive ──────────────────────────────────────────────────
    print(f"\n[2/4] Running descriptive analytics...")
    desc = descriptive.run(df)
    kpis = desc.get("kpis", {})
    print(f"      ✓ Total appointments: {kpis.get('total_appointments')}")
    print(f"      ✓ Completion rate: {kpis.get('completion_rate')}%")
    print(f"      ✓ No-show rate: {kpis.get('no_show_rate')}%")
    print(f"      ✓ Revenue at risk: R{kpis.get('revenue_lost', 0):,.0f}")

    # ── Layer 3: Predictive ───────────────────────────────────────────────────
    print(f"\n[3/4] Training predictive model...")
    pred = predictive.run(df)
    print(f"      ✓ Model: {pred.get('model_type')}")
    if pred.get("model_auc"):
        print(f"      ✓ AUC score: {pred.get('model_auc')}")
    risk = pred.get("risk_distribution", {})
    print(f"      ✓ High Risk: {risk.get('High Risk',0)} | Medium: {risk.get('Medium Risk',0)} | Low: {risk.get('Low Risk',0)}")

    # ── Layer 4: Prescriptive ─────────────────────────────────────────────────
    print(f"\n[4/4] Generating recommendations...")
    presc = prescriptive.run(df, desc, pred)
    summary = presc.get("summary", {})
    print(f"      ✓ {summary.get('total_recommendations')} recommendations generated")
    print(f"      ✓ {summary.get('high_priority')} high-priority actions")
    print(f"      ✓ {summary.get('total_agent_actions')} agent actions queued")

    # ── Layer 5: Dashboard ────────────────────────────────────────────────────
    source_stem = Path(filepath).stem
    out_path    = str(Path(output_dir) / f"{source_stem}_analytics_report.html")
    print(f"\n[→]  Rendering dashboard...")
    dashboard.render(meta, desc, pred, presc, output_path=out_path)
    print(f"      ✓ Report saved: {out_path}")

    print(f"\n{'='*60}")
    print(f"  Pipeline complete.")
    print(f"{'='*60}\n")

    return {
        "meta":         meta,
        "descriptive":  desc,
        "predictive":   pred,
        "prescriptive": presc,
        "output":       out_path,
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py <path_to_excel_or_csv>")
        print("Example: python run.py bookings.xlsx")
        sys.exit(1)

    filepath = sys.argv[1]
    if not Path(filepath).exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)

    results = run_pipeline(filepath)
