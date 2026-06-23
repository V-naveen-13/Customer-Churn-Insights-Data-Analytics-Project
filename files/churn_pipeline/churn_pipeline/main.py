"""
main.py — Entry point for the Customer Churn Analysis Pipeline.

Usage:
    python main.py --data path/to/customers.csv
    python main.py --data data/sample_customers.csv --company "Acme Corp"
    python main.py --demo   # generates sample data and runs full pipeline

Options:
    --data       PATH    Path to your CSV or Excel file
    --company    NAME    Company name for the report header (default: from config.py)
    --output     DIR     Output directory (default: outputs/)
    --no-report          Skip HTML report generation
    --demo               Generate synthetic data and run the full pipeline
    --help               Show this message
"""

import argparse
import sys
import time
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Customer Churn Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data',      type=str, help='Path to CSV or Excel file')
    parser.add_argument('--company',   type=str, help='Company name for the report')
    parser.add_argument('--output',    type=str, default='outputs', help='Output directory')
    parser.add_argument('--no-report', action='store_true', help='Skip HTML report generation')
    parser.add_argument('--demo',      action='store_true', help='Run demo with synthetic data')
    return parser.parse_args()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║       Customer Churn Analysis Pipeline v1.0              ║
║       Logistic Regression · Random Forest · XGBoost      ║
╚══════════════════════════════════════════════════════════╝
""")


def main():
    args = parse_args()
    print_banner()

    # ── Demo mode ────────────────────────────────────────────────────────────
    if args.demo:
        print("[main] Demo mode — generating synthetic dataset...")
        import generate_sample_data   # runs on import
        data_path = 'data/sample_customers.csv'
    elif args.data:
        data_path = args.data
    else:
        print("[main] ERROR: Provide --data PATH or use --demo for a demo run.")
        print("       Run: python main.py --help")
        sys.exit(1)

    # ── Apply CLI overrides to config ─────────────────────────────────────────
    import config
    if args.company:
        config.REPORT_CONFIG['company_name'] = args.company
    if args.output:
        config.REPORT_CONFIG['output_dir'] = args.output

    # Ensure output dirs exist
    Path(config.REPORT_CONFIG['output_dir']).mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Step 1: Ingest & validate ─────────────────────────────────────────────
    print("\n" + "─"*55)
    print("STEP 1 / 3 — Data ingestion & validation")
    print("─"*55)
    from ingest import load_and_validate
    df, validation_report = load_and_validate(data_path)

    # ── Step 2: EDA ───────────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("STEP 2 / 3 — Exploratory data analysis")
    print("─"*55)
    from eda import run_eda
    eda_output = run_eda(df)

    # ── Step 3: Modelling ─────────────────────────────────────────────────────
    print("\n" + "─"*55)
    print("STEP 3 / 3 — Predictive modelling")
    print("─"*55)
    from model import run_modelling
    model_output = run_modelling(df)

    # ── Report ────────────────────────────────────────────────────────────────
    report_path = None
    if not args.no_report:
        print("\n" + "─"*55)
        print("GENERATING REPORT")
        print("─"*55)
        from report import generate_report
        report_path = generate_report(eda_output, model_output)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    out_dir = config.REPORT_CONFIG['output_dir']

    print(f"""
╔══════════════════════════════════════════════════════════╗
║  Pipeline complete in {elapsed:.1f}s
╠══════════════════════════════════════════════════════════╣
║  Customers analysed : {eda_output['stats']['n_total']:,}
║  Churn rate         : {eda_output['stats']['churn_rate']:.1%}
║  High-risk accounts : {(model_output['scored']['risk_level'] == 'High').sum():,}
╠══════════════════════════════════════════════════════════╣
║  Outputs saved to: {out_dir}/
║    ├── eda/             EDA charts (PNG)
║    ├── models/          Model charts + SHAP (PNG)
║    ├── at_risk_customers.csv
║    {'└── churn_analysis_report.html' if report_path else ''}
╚══════════════════════════════════════════════════════════╝
""")

    return {
        'df':           df,
        'eda':          eda_output,
        'model':        model_output,
        'report_path':  report_path,
    }


if __name__ == '__main__':
    main()
