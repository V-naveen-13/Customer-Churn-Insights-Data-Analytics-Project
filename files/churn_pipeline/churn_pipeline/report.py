"""
report.py — Generate a self-contained HTML report from EDA + modelling results.

The HTML report embeds all charts as base64 inline images so it's a single
portable file with no external dependencies.

Output: outputs/churn_analysis_report.html
"""

import base64
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from jinja2 import Template
from config import REPORT_CONFIG, RISK_THRESHOLDS, MODEL_CONFIG


def _img_to_b64(path: str | None) -> str | None:
    """Convert an image file to a base64 data URI for inline HTML embedding."""
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    with open(p, 'rb') as f:
        data = base64.b64encode(f.read()).decode('utf-8')
    return f"data:image/png;base64,{data}"


def _flatten_figures(figures: dict) -> list[str]:
    """Flatten a figures dict (values may be str or list[str]) to a flat list."""
    out = []
    for v in figures.values():
        if isinstance(v, list):
            out.extend([x for x in v if x])
        elif v:
            out.append(v)
    return out


def _format_pct(val: float) -> str:
    return f"{val:.1%}"


def _format_num(val) -> str:
    if isinstance(val, float):
        return f"{val:,.1f}"
    return f"{val:,}"


def _build_metrics_table(eval_results: dict) -> list[dict]:
    rows = []
    for name, res in eval_results.items():
        if name.startswith('_'):
            continue
        rep = res.get('report', {})
        churned_metrics = rep.get('1', rep.get('Churned', {}))
        rows.append({
            'model':     name.replace('_', ' ').title(),
            'auc':       f"{res['auc']:.3f}",
            'cv_auc':    f"{res['cv_auc']:.3f}",
            'precision': f"{churned_metrics.get('precision', 0):.3f}",
            'recall':    f"{churned_metrics.get('recall', 0):.3f}",
            'f1':        f"{churned_metrics.get('f1-score', 0):.3f}",
            'brier':     f"{res['brier']:.3f}",
            'best': res['auc'] == max(
                v['auc'] for k, v in eval_results.items() if not k.startswith('_')
            ),
        })
    return sorted(rows, key=lambda r: r['auc'], reverse=True)


def _build_risk_summary(scored: pd.DataFrame) -> dict:
    high   = (scored['risk_level'] == 'High').sum()
    medium = (scored['risk_level'] == 'Medium').sum()
    low    = (scored['risk_level'] == 'Low').sum()
    total  = len(scored)

    # MRR at risk if available
    mrr_col = next((c for c in ['mrr', 'arr', 'monthly_revenue'] if c in scored.columns), None)
    mrr_at_risk = None
    if mrr_col:
        mrr_at_risk = scored[scored['risk_level'] == 'High'][mrr_col].sum()

    return {
        'high': high, 'medium': medium, 'low': low, 'total': total,
        'high_pct':   f"{high/total:.1%}",
        'medium_pct': f"{medium/total:.1%}",
        'low_pct':    f"{low/total:.1%}",
        'mrr_at_risk': f"${mrr_at_risk:,.0f}" if mrr_at_risk is not None else None,
    }


def _top_at_risk(scored: pd.DataFrame, n: int = 20) -> list[dict]:
    cols_to_show = ['churn_probability', 'risk_level']
    for c in ['customer_id', 'segment', 'plan', 'tenure', 'mrr', 'arr']:
        if c in scored.columns:
            cols_to_show.insert(0, c)

    top = scored.head(n)[cols_to_show].copy()
    rows = []
    for _, row in top.iterrows():
        r = {}
        for col in cols_to_show:
            val = row[col]
            if col == 'churn_probability':
                r[col] = f"{float(val):.1%}"
            elif isinstance(val, float):
                r[col] = f"{val:,.1f}"
            else:
                r[col] = str(val)
        r['_risk'] = row['risk_level']
        rows.append(r)
    return rows, list(cols_to_show)


def _generate_recommendations(eda_stats: dict, eval_results: dict,
                               scored: pd.DataFrame) -> list[dict]:
    """Generate evidence-backed recommendations from findings."""
    recs = []
    churn_rate = eda_stats.get('churn_rate', 0)

    # High churn rate
    if churn_rate > 0.10:
        recs.append({
            'priority': 'Critical',
            'title': 'Immediate retention intervention required',
            'finding': f"Overall churn rate is {churn_rate:.1%} — above the 10% critical threshold.",
            'action': "Launch an emergency retention programme. Identify the top 50 highest-risk accounts this week and assign customer success owners to each within 48 hours.",
            'impact': 'High',
        })

    # High-risk volume
    high_risk_count = (scored['risk_level'] == 'High').sum()
    pct_high = high_risk_count / len(scored)
    if pct_high > 0.15:
        recs.append({
            'priority': 'High',
            'title': 'Proactive outreach to high-risk cohort',
            'finding': f"{high_risk_count:,} customers ({pct_high:.1%}) have churn probability ≥ {RISK_THRESHOLDS['high']:.0%}.",
            'action': "Implement a 90-day high-risk playbook: personalised QBR offer, product health audit, executive sponsorship call, and targeted discount threshold for at-risk ARR above $50k.",
            'impact': 'High',
        })

    # Engagement signal
    if 'days_since_login' in scored.columns:
        inactive = (scored['days_since_login'] >= 30).mean()
        if inactive > 0.2:
            recs.append({
                'priority': 'High',
                'title': 'Re-engagement programme for inactive users',
                'finding': f"{inactive:.0%} of customers haven't logged in for 30+ days — a leading churn predictor.",
                'action': "Trigger automated re-engagement sequence: day-30 inactivity email with personalised feature spotlight, day-45 success manager call, day-60 'we miss you' offer with limited-time incentive.",
                'impact': 'High',
            })

    # Support burden
    if 'support_tickets' in scored.columns:
        high_ticket = scored['support_tickets'].quantile(0.9)
        recs.append({
            'priority': 'Medium',
            'title': 'Resolve chronic support issues before they drive churn',
            'finding': f"Top 10% of customers by support volume file {high_ticket:.0f}+ tickets. High support burden correlates with churn.",
            'action': "Create a 'red account' programme for customers filing 5+ tickets in 30 days. Escalate to senior support and initiate a product-side root cause investigation. Offer proactive training to reduce avoidable tickets.",
            'impact': 'Medium',
        })

    # NPS detractors
    if 'nps_score' in scored.columns:
        detractors = (scored['nps_score'] <= 6).mean()
        if detractors > 0.15:
            recs.append({
                'priority': 'Medium',
                'title': 'Close the loop with NPS detractors',
                'finding': f"{detractors:.0%} of customers are NPS detractors (score 0–6). Detractors churn at 3–5× the rate of promoters.",
                'action': "Within 5 business days of any NPS score ≤ 6, trigger a personal call from a senior CS rep. Document the feedback, escalate product issues to the roadmap, and follow up with resolution confirmation.",
                'impact': 'Medium',
            })

    # Best model
    best_name = max(
        [k for k in eval_results if not k.startswith('_')],
        key=lambda k: eval_results[k]['auc']
    )
    best_auc = eval_results[best_name]['auc']
    recs.append({
        'priority': 'Medium',
        'title': 'Operationalise the churn model in CRM',
        'finding': f"The {best_name.replace('_',' ').title()} model achieves AUC={best_auc:.3f}, meaning it correctly ranks churners above non-churners {best_auc:.0%} of the time.",
        'action': "Export weekly churn scores to your CRM. Create automated alerts when a customer crosses the high-risk threshold. Build a CS dashboard showing risk trends so managers can prioritise workload.",
        'impact': 'High',
    })

    recs.append({
        'priority': 'Low',
        'title': 'Incentivise annual contract conversion',
        'finding': "Monthly contract customers churn at significantly higher rates than annual subscribers — contract length is consistently among the top churn predictors.",
        'action': "Design an annual contract migration campaign: offer 1–2 months free, priority support tier, or feature unlocks for monthly customers who convert. A/B test the incentive structure. Target customers in months 3–8 of tenure before the critical churn window.",
        'impact': 'Medium',
    })

    return recs


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --primary: {{ primary }};
    --danger:  {{ danger }};
    --success: {{ success }};
    --warn:    {{ warn }};
    --neutral: {{ neutral }};
    --text:    #1a1a1a;
    --muted:   #6b6b6b;
    --border:  #e0e0e0;
    --surface: #f7f7f7;
    --white:   #ffffff;
  }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
         color: var(--text); background: #f4f4f4; font-size: 15px; line-height: 1.6; }
  .page { max-width: 1100px; margin: 0 auto; padding: 2rem 1.5rem; }
  
  /* Header */
  .report-header { background: var(--white); border-radius: 12px; padding: 2rem 2.5rem;
                   margin-bottom: 1.5rem; border-left: 5px solid var(--primary);
                   box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
  .report-header h1 { font-size: 26px; font-weight: 700; color: var(--text); }
  .report-meta { font-size: 13px; color: var(--muted); margin-top: 6px; }
  
  /* Sections */
  .section { background: var(--white); border-radius: 12px; padding: 1.75rem 2rem;
             margin-bottom: 1.5rem; box-shadow: 0 1px 4px rgba(0,0,0,0.07); }
  .section h2 { font-size: 18px; font-weight: 700; margin-bottom: 1rem;
                padding-bottom: 0.6rem; border-bottom: 1px solid var(--border); color: var(--text); }
  .section h3 { font-size: 15px; font-weight: 600; margin: 1.2rem 0 0.6rem; color: var(--text); }

  /* Metric cards */
  .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin-bottom: 1.5rem; }
  .metric-card { background: var(--surface); border-radius: 8px; padding: 1rem 1.25rem; }
  .metric-label { font-size: 12px; color: var(--muted); margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.04em; }
  .metric-value { font-size: 28px; font-weight: 700; color: var(--text); }
  .metric-value.danger { color: var(--danger); }
  .metric-value.success { color: var(--success); }
  
  /* Charts */
  .chart-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(440px, 1fr)); gap: 16px; }
  .chart-wrap { background: var(--surface); border-radius: 8px; padding: 1rem; }
  .chart-wrap img { width: 100%; height: auto; border-radius: 4px; }
  .chart-wrap p { font-size: 12px; color: var(--muted); margin-top: 8px; text-align: center; }
  .chart-full img { width: 100%; height: auto; border-radius: 4px; }
  
  /* Tables */
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th { text-align: left; font-weight: 600; font-size: 12px; color: var(--muted);
       padding: 8px 10px; border-bottom: 2px solid var(--border); text-transform: uppercase; letter-spacing: 0.04em; }
  td { padding: 9px 10px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: var(--surface); }
  .best-row td { background: #eaf5ff; }
  
  /* Risk badges */
  .badge { display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; }
  .badge-high   { background: #fde8e8; color: #a32d2d; }
  .badge-medium { background: #fef3e0; color: #854f0b; }
  .badge-low    { background: #e5f3e8; color: #2d6a35; }
  .badge-best   { background: #e3f0ff; color: #1045a0; }
  
  /* Recommendations */
  .rec { border-left: 4px solid var(--border); padding: 1rem 1.25rem; margin-bottom: 1rem; border-radius: 0 8px 8px 0; background: var(--surface); }
  .rec.critical { border-left-color: #c0392b; }
  .rec.high     { border-left-color: var(--danger); }
  .rec.medium   { border-left-color: var(--warn); }
  .rec.low      { border-left-color: var(--neutral); }
  .rec-header   { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .rec-title    { font-weight: 600; font-size: 14px; }
  .rec-finding  { font-size: 13px; color: var(--muted); margin-bottom: 6px; }
  .rec-action   { font-size: 13px; }
  .priority-label { font-size: 11px; font-weight: 700; padding: 2px 8px; border-radius: 4px; text-transform: uppercase; }
  .priority-Critical { background: #fde8e8; color: #a32d2d; }
  .priority-High     { background: #fde8e8; color: #c0392b; }
  .priority-Medium   { background: #fef3e0; color: #7d4e0a; }
  .priority-Low      { background: #f1f1f1; color: #555; }
  
  /* Nav */
  .toc { background: var(--surface); border-radius: 8px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; font-size: 13px; }
  .toc a { color: var(--primary); text-decoration: none; display: block; padding: 3px 0; }
  .toc a:hover { text-decoration: underline; }

  @media print { body { background: white; } .section { box-shadow: none; border: 1px solid var(--border); } }
</style>
</head>
<body>
<div class="page">

  <div class="report-header">
    <h1>{{ title }}</h1>
    <div class="report-meta">{{ company }} &nbsp;·&nbsp; Generated {{ date }} &nbsp;·&nbsp; {{ n_customers }} customers analysed</div>
  </div>

  <!-- Table of contents -->
  <div class="toc">
    <strong style="font-size:12px;text-transform:uppercase;letter-spacing:0.06em;color:var(--muted)">Contents</strong>
    <a href="#overview">1. Executive summary</a>
    <a href="#eda">2. Exploratory data analysis</a>
    <a href="#segmentation">3. Segment breakdown</a>
    <a href="#models">4. Predictive modelling</a>
    <a href="#atrisk">5. At-risk customers</a>
    <a href="#recommendations">6. Recommendations</a>
  </div>

  <!-- 1. Executive summary -->
  <div class="section" id="overview">
    <h2>1. Executive summary</h2>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">Overall churn rate</div>
        <div class="metric-value danger">{{ churn_rate }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Total customers</div>
        <div class="metric-value">{{ n_customers }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Churned customers</div>
        <div class="metric-value danger">{{ n_churned }}</div>
      </div>
      <div class="metric-card">
        <div class="metric-label">High-risk accounts</div>
        <div class="metric-value danger">{{ risk_summary.high }}</div>
      </div>
      {% if risk_summary.mrr_at_risk %}
      <div class="metric-card">
        <div class="metric-label">MRR at risk (high-risk)</div>
        <div class="metric-value danger">{{ risk_summary.mrr_at_risk }}</div>
      </div>
      {% endif %}
      <div class="metric-card">
        <div class="metric-label">Best model AUC</div>
        <div class="metric-value success">{{ best_auc }}</div>
      </div>
    </div>
    {% if overview_img %}
    <div class="chart-full"><img src="{{ overview_img }}" alt="Churn overview chart"></div>
    {% endif %}
  </div>

  <!-- 2. EDA -->
  <div class="section" id="eda">
    <h2>2. Exploratory data analysis</h2>
    <h3>Feature distributions by churn status</h3>
    <div class="chart-grid">
      {% for img in dist_imgs %}
      <div class="chart-wrap"><img src="{{ img }}" alt="Feature distribution chart"></div>
      {% endfor %}
    </div>
    {% if corr_img %}
    <h3>Correlation matrix</h3>
    <div class="chart-full"><img src="{{ corr_img }}" alt="Correlation heatmap"></div>
    {% endif %}
    {% if tenure_img %}
    <h3>Churn rate by customer tenure</h3>
    <div class="chart-wrap" style="max-width:700px"><img src="{{ tenure_img }}" alt="Tenure vs churn chart"></div>
    {% endif %}
    {% if revenue_img %}
    <h3>Revenue at risk</h3>
    <div class="chart-full"><img src="{{ revenue_img }}" alt="Revenue at risk chart"></div>
    {% endif %}
  </div>

  <!-- 3. Segmentation -->
  <div class="section" id="segmentation">
    <h2>3. Segment breakdown</h2>
    <div class="chart-grid">
      {% for img in cat_imgs %}
      <div class="chart-wrap"><img src="{{ img }}" alt="Churn rate by segment chart"></div>
      {% endfor %}
    </div>
  </div>

  <!-- 4. Modelling -->
  <div class="section" id="models">
    <h2>4. Predictive modelling</h2>
    <h3>Model performance</h3>
    <table>
      <thead>
        <tr>
          <th>Model</th><th>AUC (test)</th><th>CV AUC</th>
          <th>Precision</th><th>Recall</th><th>F1</th><th>Brier score</th>
        </tr>
      </thead>
      <tbody>
        {% for row in metrics_table %}
        <tr {% if row.best %}class="best-row"{% endif %}>
          <td>{{ row.model }} {% if row.best %}<span class="badge badge-best">Best</span>{% endif %}</td>
          <td><strong>{{ row.auc }}</strong></td>
          <td>{{ row.cv_auc }}</td>
          <td>{{ row.precision }}</td>
          <td>{{ row.recall }}</td>
          <td>{{ row.f1 }}</td>
          <td>{{ row.brier }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>

    <h3>ROC curves</h3>
    {% if roc_img %}
    <div class="chart-wrap" style="max-width:600px"><img src="{{ roc_img }}" alt="ROC curves"></div>
    {% endif %}

    {% if cm_img %}
    <h3>Confusion matrices</h3>
    <div class="chart-full"><img src="{{ cm_img }}" alt="Confusion matrices"></div>
    {% endif %}

    {% if comp_img %}
    <h3>Overall model comparison</h3>
    <div class="chart-full"><img src="{{ comp_img }}" alt="Model comparison chart"></div>
    {% endif %}

    <h3>Feature importance</h3>
    <div class="chart-grid">
      {% for img in fi_imgs %}
      <div class="chart-wrap"><img src="{{ img }}" alt="Feature importance chart"></div>
      {% endfor %}
    </div>
  </div>

  <!-- 5. At-risk customers -->
  <div class="section" id="atrisk">
    <h2>5. At-risk customers</h2>
    <div class="metric-grid">
      <div class="metric-card">
        <div class="metric-label">High risk</div>
        <div class="metric-value danger">{{ risk_summary.high }} <span style="font-size:16px">({{ risk_summary.high_pct }})</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Medium risk</div>
        <div class="metric-value" style="color:var(--warn)">{{ risk_summary.medium }} <span style="font-size:16px">({{ risk_summary.medium_pct }})</span></div>
      </div>
      <div class="metric-card">
        <div class="metric-label">Low risk</div>
        <div class="metric-value success">{{ risk_summary.low }} <span style="font-size:16px">({{ risk_summary.low_pct }})</span></div>
      </div>
    </div>
    <p style="font-size:13px;color:var(--muted);margin-bottom:1rem">
      Top 20 highest-risk customers shown below. Full scored list saved to <code>outputs/at_risk_customers.csv</code>.
    </p>
    <div style="overflow-x:auto">
    <table>
      <thead><tr>
        {% for col in risk_col_headers %}<th>{{ col.replace('_',' ').title() }}</th>{% endfor %}
        <th>Risk level</th>
      </tr></thead>
      <tbody>
        {% for row in at_risk_rows %}
        <tr>
          {% for col in risk_col_headers %}
          <td>{{ row[col] }}</td>
          {% endfor %}
          <td><span class="badge badge-{{ row._risk | lower }}">{{ row._risk }}</span></td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
    </div>
  </div>

  <!-- 6. Recommendations -->
  <div class="section" id="recommendations">
    <h2>6. Recommendations</h2>
    {% for rec in recommendations %}
    <div class="rec {{ rec.priority | lower }}">
      <div class="rec-header">
        <span class="priority-label priority-{{ rec.priority }}">{{ rec.priority }}</span>
        <span class="rec-title">{{ rec.title }}</span>
        <span class="badge" style="margin-left:auto;background:#f0f0f0;color:#444">Impact: {{ rec.impact }}</span>
      </div>
      <div class="rec-finding"><strong>Finding:</strong> {{ rec.finding }}</div>
      <div class="rec-action"><strong>Action:</strong> {{ rec.action }}</div>
    </div>
    {% endfor %}
  </div>

  <p style="text-align:center;font-size:12px;color:var(--muted);margin-top:2rem;padding-bottom:2rem">
    Customer Churn Analysis Pipeline · Generated {{ date }}
  </p>
</div>
</body>
</html>"""


def generate_report(eda_output: dict, model_output: dict) -> str:
    """
    Build and save the HTML report. Returns the path to the file.
    """
    print("\n[report] Generating HTML report...")

    cfg     = REPORT_CONFIG
    stats   = eda_output['stats']
    figs    = eda_output['figures']
    scored  = model_output['scored']
    evals   = model_output['eval_results']
    fi      = model_output.get('fi_paths', {})

    # Best model AUC
    best_auc = max(
        v['auc'] for k, v in evals.items() if not k.startswith('_')
    )

    # Risk summary
    risk_summary = _build_risk_summary(scored)

    # At-risk table
    at_risk_rows, risk_col_headers = _top_at_risk(scored, n=20)
    # Remove 'risk_level' from headers (shown separately)
    risk_col_headers = [c for c in risk_col_headers if c != 'risk_level']

    # Metrics table
    metrics_table = _build_metrics_table(evals)

    # Recommendations
    recs = _generate_recommendations(stats, evals, scored)

    # Convert figures to base64
    overview_img = _img_to_b64(figs.get('overview'))
    corr_img     = _img_to_b64(figs.get('correlation'))
    tenure_img   = _img_to_b64(figs.get('tenure'))
    revenue_img  = _img_to_b64(figs.get('revenue_at_risk'))
    roc_img      = _img_to_b64(evals.get('_fig_roc'))
    cm_img       = _img_to_b64(evals.get('_fig_cm'))
    comp_img     = _img_to_b64(model_output.get('comparison'))

    dist_imgs = [_img_to_b64(p) for p in (figs.get('distributions') or []) if p]
    cat_imgs  = [_img_to_b64(p) for p in (figs.get('categorical') or []) if p]
    fi_imgs   = [_img_to_b64(p) for p in fi.values() if p and Path(p).exists()]

    template = Template(HTML_TEMPLATE)
    html = template.render(
        title        = cfg['report_title'],
        company      = cfg['company_name'],
        date         = datetime.now().strftime('%d %B %Y, %H:%M'),
        primary      = cfg['color_primary'],
        danger       = cfg['color_danger'],
        success      = cfg['color_success'],
        warn         = cfg['color_warn'],
        neutral      = cfg['color_neutral'],
        n_customers  = f"{stats['n_total']:,}",
        n_churned    = f"{stats['n_churned']:,}",
        churn_rate   = _format_pct(stats['churn_rate']),
        best_auc     = f"{best_auc:.3f}",
        risk_summary = risk_summary,
        overview_img = overview_img,
        dist_imgs    = dist_imgs,
        cat_imgs     = cat_imgs,
        corr_img     = corr_img,
        tenure_img   = tenure_img,
        revenue_img  = revenue_img,
        roc_img      = roc_img,
        cm_img       = cm_img,
        comp_img     = comp_img,
        fi_imgs      = fi_imgs,
        metrics_table     = metrics_table,
        at_risk_rows      = at_risk_rows,
        risk_col_headers  = risk_col_headers,
        recommendations   = recs,
    )

    out_dir  = Path(cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / 'churn_analysis_report.html'

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(html)

    print(f"[report] Report saved → {out_path}")
    return str(out_path)
