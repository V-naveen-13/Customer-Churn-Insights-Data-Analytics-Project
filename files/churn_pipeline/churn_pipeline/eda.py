"""
eda.py — Exploratory Data Analysis for the churn pipeline.

Produces:
  - Summary statistics by churn status
  - Distribution plots for numeric features
  - Churn rate by categorical segment
  - Correlation heatmap
  - Tenure cohort analysis
  - All figures saved to outputs/eda/
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from config import COL, NUMERIC_COLS, CATEGORICAL_COLS, REPORT_CONFIG

OUTPUT_DIR = Path(REPORT_CONFIG['output_dir']) / 'eda'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY   = REPORT_CONFIG['color_primary']
DANGER    = REPORT_CONFIG['color_danger']
SUCCESS   = REPORT_CONFIG['color_success']
NEUTRAL   = REPORT_CONFIG['color_neutral']
WARN      = REPORT_CONFIG['color_warn']

PALETTE   = [PRIMARY, DANGER, SUCCESS, WARN, NEUTRAL, '#7f77dd', '#1d9e75', '#ba7517']


def _savefig(name: str):
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [eda] Saved {path}")
    return str(path)


def _numeric_cols_present(df: pd.DataFrame) -> list[str]:
    """Return numeric columns that actually exist in df (using internal names)."""
    present = []
    for c in NUMERIC_COLS:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            present.append(c)
    return present


def _cat_cols_present(df: pd.DataFrame) -> list[str]:
    return [c for c in CATEGORICAL_COLS if c in df.columns]


# ── 1. Summary statistics ────────────────────────────────────────────────────

def summary_stats(df: pd.DataFrame) -> dict:
    """Compute descriptive statistics split by churn status."""
    stats = {}
    churn_col = 'churn'

    stats['n_total']    = len(df)
    stats['n_churned']  = int(df[churn_col].sum())
    stats['n_retained'] = int((df[churn_col] == 0).sum())
    stats['churn_rate'] = df[churn_col].mean()

    numeric = _numeric_cols_present(df)
    if numeric:
        stats['by_churn'] = df.groupby(churn_col)[numeric].agg(['mean', 'median', 'std']).round(2)

    return stats


def print_summary(stats: dict):
    print("\n=== EDA Summary ===")
    print(f"Total customers : {stats['n_total']:,}")
    print(f"Churned         : {stats['n_churned']:,} ({stats['churn_rate']:.1%})")
    print(f"Retained        : {stats['n_retained']:,} ({1 - stats['churn_rate']:.1%})")
    if 'by_churn' in stats:
        print("\nMean values by churn status:")
        by_churn = stats['by_churn']
        means = by_churn.xs('mean', axis=1, level=1) if isinstance(by_churn.columns, pd.MultiIndex) else by_churn
        print(means.T.rename(columns={0: 'Retained', 1: 'Churned'}).to_string())


# ── 2. Overall churn rate chart ──────────────────────────────────────────────

def plot_churn_overview(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    fig.suptitle('Churn Overview', fontsize=14, fontweight='bold', y=1.02)

    # Pie
    counts = df['churn'].value_counts()
    labels = ['Retained', 'Churned']
    colors = [SUCCESS, DANGER]
    axes[0].pie(
        [counts.get(0, 0), counts.get(1, 0)],
        labels=labels, colors=colors, autopct='%1.1f%%',
        startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2}
    )
    axes[0].set_title('Overall churn split')

    # Bar — churn by segment if available
    seg_col = 'segment' if 'segment' in df.columns else 'plan' if 'plan' in df.columns else None
    if seg_col:
        seg_churn = df.groupby(seg_col)['churn'].mean().sort_values(ascending=True)
        bars = axes[1].barh(seg_churn.index, seg_churn.values, color=PRIMARY, edgecolor='white')
        axes[1].xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        axes[1].set_title(f'Churn rate by {seg_col}')
        axes[1].set_xlabel('Churn rate')
        for bar, val in zip(bars, seg_churn.values):
            axes[1].text(val + 0.005, bar.get_y() + bar.get_height()/2,
                         f'{val:.1%}', va='center', fontsize=10)
        axes[1].set_xlim(0, seg_churn.max() * 1.25)
    else:
        axes[1].text(0.5, 0.5, 'No segment column found', ha='center', va='center',
                     transform=axes[1].transAxes, color='gray')
        axes[1].set_title('Churn by segment (unavailable)')

    plt.tight_layout()
    return _savefig('01_churn_overview.png')


# ── 3. Numeric feature distributions ─────────────────────────────────────────

def plot_numeric_distributions(df: pd.DataFrame) -> list[str]:
    numeric = _numeric_cols_present(df)
    numeric = [c for c in numeric if c != 'churn']
    if not numeric:
        return []

    paths = []
    # Plot in rows of 3
    chunk_size = 3
    for chunk_i, chunk in enumerate([numeric[i:i+chunk_size] for i in range(0, len(numeric), chunk_size)]):
        fig, axes = plt.subplots(1, len(chunk), figsize=(5*len(chunk), 4))
        if len(chunk) == 1:
            axes = [axes]
        fig.suptitle('Feature distributions by churn status', fontsize=13, fontweight='bold')

        for ax, col in zip(axes, chunk):
            retained = df[df['churn'] == 0][col].dropna()
            churned  = df[df['churn'] == 1][col].dropna()
            ax.hist(retained, bins=30, alpha=0.55, color=SUCCESS, label='Retained', density=True)
            ax.hist(churned,  bins=30, alpha=0.55, color=DANGER,  label='Churned',  density=True)
            ax.set_title(col.replace('_', ' ').title(), fontsize=11)
            ax.set_xlabel(col)
            ax.set_ylabel('Density')
            ax.legend(fontsize=9)
            ax.spines[['top','right']].set_visible(False)

        plt.tight_layout()
        path = _savefig(f'02_distributions_{chunk_i:02d}.png')
        paths.append(path)

    return paths


# ── 4. Categorical churn rates ────────────────────────────────────────────────

def plot_categorical_churn_rates(df: pd.DataFrame) -> list[str]:
    cats = _cat_cols_present(df)
    cats = [c for c in cats if df[c].nunique() <= 30]  # skip high-cardinality
    if not cats:
        return []

    paths = []
    for col in cats:
        churn_rates = df.groupby(col)['churn'].agg(['mean', 'count']).reset_index()
        churn_rates.columns = [col, 'churn_rate', 'count']
        churn_rates = churn_rates[churn_rates['count'] >= 5]  # skip tiny groups
        churn_rates = churn_rates.sort_values('churn_rate', ascending=True)

        if churn_rates.empty:
            continue

        fig, ax = plt.subplots(figsize=(8, max(3, len(churn_rates) * 0.5)))
        colors = [DANGER if r >= df['churn'].mean() else SUCCESS for r in churn_rates['churn_rate']]
        bars = ax.barh(churn_rates[col].astype(str), churn_rates['churn_rate'],
                       color=colors, edgecolor='white', height=0.6)
        ax.axvline(df['churn'].mean(), color=NEUTRAL, linestyle='--', linewidth=1.2,
                   label=f"Overall avg {df['churn'].mean():.1%}")
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        ax.set_title(f'Churn rate by {col.replace("_"," ").title()}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Churn rate')
        ax.legend(fontsize=9)
        ax.spines[['top','right']].set_visible(False)

        for bar, (_, row) in zip(bars, churn_rates.iterrows()):
            ax.text(row['churn_rate'] + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{row['churn_rate']:.1%} (n={row['count']:,})",
                    va='center', fontsize=9)
        ax.set_xlim(0, churn_rates['churn_rate'].max() * 1.35)
        plt.tight_layout()
        path = _savefig(f'03_churn_by_{col}.png')
        paths.append(path)

    return paths


# ── 5. Correlation heatmap ────────────────────────────────────────────────────

def plot_correlation_heatmap(df: pd.DataFrame) -> str:
    numeric = _numeric_cols_present(df)
    if len(numeric) < 2:
        return None

    corr = df[numeric].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(max(8, len(numeric)), max(6, len(numeric) * 0.8)))
    cmap = sns.diverging_palette(10, 220, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, center=0,
                annot=True, fmt='.2f', annot_kws={'size': 9},
                linewidths=0.5, ax=ax, vmin=-1, vmax=1)
    ax.set_title('Feature correlation matrix', fontsize=13, fontweight='bold', pad=12)
    ax.tick_params(axis='x', rotation=45, labelsize=9)
    ax.tick_params(axis='y', rotation=0, labelsize=9)
    plt.tight_layout()
    return _savefig('04_correlation_heatmap.png')


# ── 6. Tenure cohort analysis ─────────────────────────────────────────────────

def plot_tenure_churn(df: pd.DataFrame) -> str:
    if 'tenure' not in df.columns:
        return None

    df2 = df.copy()
    bins = [0, 3, 6, 12, 24, 36, 60, float('inf')]
    labels = ['0-3mo', '3-6mo', '6-12mo', '1-2yr', '2-3yr', '3-5yr', '5yr+']
    df2['tenure_band'] = pd.cut(df2['tenure'], bins=bins, labels=labels, right=False)

    band_stats = df2.groupby('tenure_band', observed=True)['churn'].agg(['mean', 'count']).reset_index()
    band_stats.columns = ['tenure_band', 'churn_rate', 'count']

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    bars = ax1.bar(band_stats['tenure_band'].astype(str), band_stats['churn_rate'],
                   color=PRIMARY, alpha=0.75, label='Churn rate', width=0.5)
    ax2.plot(band_stats['tenure_band'].astype(str), band_stats['count'],
             color=WARN, marker='o', linewidth=2, label='Customer count', zorder=3)

    ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax1.set_ylabel('Churn rate', color=PRIMARY)
    ax2.set_ylabel('Customer count', color=WARN)
    ax1.set_xlabel('Tenure band')
    ax1.set_title('Churn rate and volume by tenure', fontsize=13, fontweight='bold')
    ax1.tick_params(axis='x', rotation=20)
    ax1.spines[['top']].set_visible(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    plt.tight_layout()
    return _savefig('05_tenure_churn.png')


# ── 7. MRR at risk ───────────────────────────────────────────────────────────

def plot_revenue_at_risk(df: pd.DataFrame) -> str:
    mrr_col = None
    for c in ['mrr', 'arr', 'monthly_revenue']:
        if c in df.columns:
            mrr_col = c
            break
    if not mrr_col:
        return None

    seg_col = 'segment' if 'segment' in df.columns else 'plan' if 'plan' in df.columns else None
    if not seg_col:
        return None

    rev_risk = df.groupby(seg_col).apply(
        lambda g: pd.Series({
            'total_rev': g[mrr_col].sum(),
            'churned_rev': g[g['churn'] == 1][mrr_col].sum(),
            'churn_rate': g['churn'].mean(),
        })
    ).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Revenue at risk by {seg_col}', fontsize=13, fontweight='bold')

    # Stacked bar: retained vs churned revenue
    retained = rev_risk['total_rev'] - rev_risk['churned_rev']
    axes[0].bar(rev_risk[seg_col], retained / 1e3, color=SUCCESS, label='Retained')
    axes[0].bar(rev_risk[seg_col], rev_risk['churned_rev'] / 1e3,
                bottom=retained / 1e3, color=DANGER, label='Churned')
    axes[0].set_ylabel(f'{mrr_col.upper()} (000s)')
    axes[0].set_title(f'{mrr_col.upper()} retained vs. lost')
    axes[0].legend()
    axes[0].tick_params(axis='x', rotation=20)
    axes[0].spines[['top','right']].set_visible(False)

    # Churn rate overlay
    axes[1].bar(rev_risk[seg_col], rev_risk['churn_rate'], color=PRIMARY, alpha=0.8)
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[1].set_title('Churn rate by segment')
    axes[1].set_ylabel('Churn rate')
    axes[1].tick_params(axis='x', rotation=20)
    axes[1].spines[['top','right']].set_visible(False)

    plt.tight_layout()
    return _savefig('06_revenue_at_risk.png')


# ── Main EDA runner ───────────────────────────────────────────────────────────

def run_eda(df: pd.DataFrame) -> dict:
    """
    Run the full EDA suite and return a dict of all figure paths + stats.
    """
    print("\n[eda] Running exploratory data analysis...")

    stats   = summary_stats(df)
    print_summary(stats)

    figures = {}
    figures['overview']        = plot_churn_overview(df)
    figures['distributions']   = plot_numeric_distributions(df)
    figures['categorical']     = plot_categorical_churn_rates(df)
    figures['correlation']     = plot_correlation_heatmap(df)
    figures['tenure']          = plot_tenure_churn(df)
    figures['revenue_at_risk'] = plot_revenue_at_risk(df)

    print(f"\n[eda] Complete. Figures saved to {OUTPUT_DIR}/")
    return {'stats': stats, 'figures': figures}
