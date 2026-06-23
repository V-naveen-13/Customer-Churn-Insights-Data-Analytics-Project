"""
model.py — Churn modelling: preprocessing, training, evaluation, SHAP explainability.

Models trained:
  1. Logistic Regression  (interpretable baseline)
  2. Random Forest        (ensemble, handles non-linearity)
  3. XGBoost              (gradient boosting, typically best performer)

Output:
  - Evaluation metrics (AUC, precision, recall, F1) for each model
  - ROC curves, confusion matrices, calibration plots
  - SHAP feature importance charts
  - Risk-scored customer table saved to outputs/at_risk_customers.csv
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    roc_auc_score, roc_curve, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    precision_recall_curve, average_precision_score,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import shap
import warnings
warnings.filterwarnings('ignore')

from config import MODEL_CONFIG, NUMERIC_COLS, CATEGORICAL_COLS, REPORT_CONFIG, RISK_THRESHOLDS

OUTPUT_DIR = Path(REPORT_CONFIG['output_dir']) / 'models'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRIMARY = REPORT_CONFIG['color_primary']
DANGER  = REPORT_CONFIG['color_danger']
SUCCESS = REPORT_CONFIG['color_success']
NEUTRAL = REPORT_CONFIG['color_neutral']
WARN    = REPORT_CONFIG['color_warn']


def _savefig(name: str) -> str:
    path = OUTPUT_DIR / name
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [model] Saved {path}")
    return str(path)


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features that are strong churn predictors.
    All operations are safe — columns are only created if source columns exist.
    """
    df = df.copy()

    # Engagement ratio: logins per month relative to tenure
    if 'logins_per_month' in df.columns and 'tenure' in df.columns:
        df['login_tenure_ratio'] = df['logins_per_month'] / (df['tenure'].clip(lower=1))

    # Support burden: tickets per month of tenure
    if 'support_tickets' in df.columns and 'tenure' in df.columns:
        df['support_burden'] = df['support_tickets'] / (df['tenure'].clip(lower=1))

    # Revenue tier (log transform to reduce skew)
    for rev_col in ['mrr', 'arr', 'monthly_revenue']:
        if rev_col in df.columns:
            df[f'{rev_col}_log'] = np.log1p(df[rev_col])
            break

    # Recency flag: inactive if no login in 30+ days
    if 'days_since_login' in df.columns:
        df['is_inactive_30d'] = (df['days_since_login'] >= 30).astype(int)
        df['is_inactive_60d'] = (df['days_since_login'] >= 60).astype(int)

    # NPS bucket: Detractor / Passive / Promoter
    if 'nps_score' in df.columns:
        df['nps_bucket'] = pd.cut(
            df['nps_score'], bins=[-1, 6, 8, 10],
            labels=['Detractor', 'Passive', 'Promoter']
        ).astype(str)
        df['is_detractor'] = (df['nps_bucket'] == 'Detractor').astype(int)

    # Tenure bands (numeric encoding)
    if 'tenure' in df.columns:
        df['is_new_customer']  = (df['tenure'] < 3).astype(int)
        df['is_early_tenure']  = ((df['tenure'] >= 3) & (df['tenure'] < 12)).astype(int)
        df['is_mature_tenure'] = (df['tenure'] >= 12).astype(int)

    return df


# ── Preprocessing ─────────────────────────────────────────────────────────────

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list]:
    """
    Build X (features) and y (target) ready for sklearn.
    - Encodes categoricals with LabelEncoder
    - Drops ID/date columns
    - Returns feature names for interpretability
    """
    df = df.copy()
    target = 'churn'

    # Columns to drop
    drop_cols = ['churn', 'customer_id', 'start_date', 'churn_date']
    drop_cols = [c for c in drop_cols if c in df.columns]

    # Encode categorical columns
    cat_cols = [c for c in df.columns if df[c].dtype == object or str(df[c].dtype) == 'category']
    cat_cols = [c for c in cat_cols if c not in drop_cols]

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna('Unknown')
        df[col] = le.fit_transform(df[col])

    # Build X and y
    feature_cols = [c for c in df.columns if c not in drop_cols and c != target]
    X = df[feature_cols].select_dtypes(include=[np.number])
    y = df[target]

    # Fill any remaining NaNs
    X = X.fillna(X.median())

    return X, y, list(X.columns)


# ── SMOTE balancing ───────────────────────────────────────────────────────────

def maybe_apply_smote(X_train, y_train) -> tuple:
    churn_rate = y_train.mean()
    threshold  = MODEL_CONFIG['smote_threshold']
    if MODEL_CONFIG['apply_smote'] and churn_rate < threshold:
        print(f"  [model] Applying SMOTE (churn rate={churn_rate:.1%} < {threshold:.0%} threshold)")
        try:
            smote = SMOTE(random_state=MODEL_CONFIG['random_state'])
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"  [model] Post-SMOTE training set: {len(y_train):,} rows")
        except Exception as e:
            print(f"  [model] SMOTE failed ({e}) — continuing without oversampling.")
    return X_train, y_train


# ── Model training ────────────────────────────────────────────────────────────

def train_models(X_train, y_train) -> dict:
    """Train all three models and return fitted estimators."""
    cfg = MODEL_CONFIG
    rs  = cfg['random_state']

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    models = {}

    # 1. Logistic Regression (needs scaling)
    print("  [model] Training Logistic Regression...")
    lr = LogisticRegression(**cfg['logistic_regression'], random_state=rs)
    lr.fit(X_train_scaled, y_train)
    models['logistic_regression'] = {'model': lr, 'scaler': scaler, 'needs_scaling': True}

    # 2. Random Forest
    print("  [model] Training Random Forest...")
    rf = RandomForestClassifier(**cfg['random_forest'], random_state=rs)
    rf.fit(X_train, y_train)
    models['random_forest'] = {'model': rf, 'scaler': None, 'needs_scaling': False}

    # 3. XGBoost
    print("  [model] Training XGBoost...")
    xgb_params = {k: v for k, v in cfg['xgboost'].items()
                  if k not in ['use_label_encoder']}
    xgb = XGBClassifier(**xgb_params, random_state=rs, verbosity=0)
    xgb.fit(X_train, y_train, eval_set=[(X_train, y_train)], verbose=False)
    models['xgboost'] = {'model': xgb, 'scaler': None, 'needs_scaling': False}

    return models


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_models(models: dict, X_test, y_test, feature_names: list) -> dict:
    """Evaluate all models and produce comparison metrics + figures."""
    results = {}

    fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
    ax_roc.plot([0,1], [0,1], 'k--', lw=1, label='Random (AUC=0.50)')
    colors_roc = [PRIMARY, DANGER, SUCCESS]

    for i, (name, bundle) in enumerate(models.items()):
        model   = bundle['model']
        scaler  = bundle['scaler']
        X_eval  = scaler.transform(X_test) if bundle['needs_scaling'] else X_test

        y_prob  = model.predict_proba(X_eval)[:, 1]
        y_pred  = (y_prob >= 0.5).astype(int)

        auc     = roc_auc_score(y_test, y_prob)
        ap      = average_precision_score(y_test, y_prob)
        brier   = brier_score_loss(y_test, y_prob)
        report  = classification_report(y_test, y_pred, output_dict=True)
        cm      = confusion_matrix(y_test, y_pred)

        # CV AUC
        cv = StratifiedKFold(n_splits=MODEL_CONFIG['cv_folds'], shuffle=True,
                             random_state=MODEL_CONFIG['random_state'])
        X_full  = scaler.transform(X_test) if bundle['needs_scaling'] else X_test
        cv_auc  = cross_val_score(model, X_full, y_test, cv=cv, scoring='roc_auc').mean()

        results[name] = {
            'auc': auc, 'avg_precision': ap, 'brier': brier,
            'cv_auc': cv_auc, 'report': report, 'cm': cm,
            'y_prob': y_prob, 'y_pred': y_pred,
        }

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        ax_roc.plot(fpr, tpr, color=colors_roc[i], lw=2,
                    label=f'{name.replace("_"," ").title()} (AUC={auc:.3f})')

        print(f"\n  [{name}] AUC={auc:.3f} | CV AUC={cv_auc:.3f} | "
              f"AP={ap:.3f} | Brier={brier:.3f}")
        print(classification_report(y_test, y_pred,
                                    target_names=['Retained','Churned']))

    ax_roc.set_xlabel('False positive rate')
    ax_roc.set_ylabel('True positive rate')
    ax_roc.set_title('ROC Curves — Model Comparison', fontsize=13, fontweight='bold')
    ax_roc.legend(loc='lower right', fontsize=10)
    ax_roc.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    results['_fig_roc'] = _savefig('01_roc_curves.png')

    # Confusion matrices
    fig_cm, axes = plt.subplots(1, len(models), figsize=(5*len(models), 4))
    if len(models) == 1: axes = [axes]
    for ax, (name, res) in zip(axes, results.items()):
        if name.startswith('_'): continue
        disp = ConfusionMatrixDisplay(res['cm'], display_labels=['Retained','Churned'])
        disp.plot(ax=ax, colorbar=False, cmap='Blues')
        ax.set_title(name.replace('_',' ').title(), fontsize=11)
    plt.suptitle('Confusion Matrices', fontsize=13, fontweight='bold')
    plt.tight_layout()
    results['_fig_cm'] = _savefig('02_confusion_matrices.png')

    # Calibration plot
    fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
    ax_cal.plot([0,1],[0,1],'k--', lw=1, label='Perfectly calibrated')
    for i, (name, res) in enumerate(results.items()):
        if name.startswith('_'): continue
        prob_true, prob_pred = calibration_curve(y_test, res['y_prob'], n_bins=10)
        ax_cal.plot(prob_pred, prob_true, marker='o', color=colors_roc[i],
                    lw=2, label=name.replace('_',' ').title())
    ax_cal.set_xlabel('Mean predicted probability')
    ax_cal.set_ylabel('Fraction of positives')
    ax_cal.set_title('Calibration curves', fontsize=13, fontweight='bold')
    ax_cal.legend(fontsize=9)
    ax_cal.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    results['_fig_cal'] = _savefig('03_calibration.png')

    return results


# ── Feature importance ────────────────────────────────────────────────────────

def plot_feature_importance(models: dict, X_test, feature_names: list) -> dict:
    """
    For Random Forest: built-in importance.
    For XGBoost: gain-based importance.
    For Logistic Regression: absolute coefficients.
    Also runs SHAP on the best tree model.
    """
    top_n = MODEL_CONFIG['feature_importance_top_n']
    paths = {}

    # --- RF importance ---
    if 'random_forest' in models:
        rf = models['random_forest']['model']
        imp = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
        imp[::-1].plot.barh(color=PRIMARY, ax=ax, edgecolor='white')
        ax.set_title('Random Forest — Feature importances (MDI)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Mean decrease in impurity')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        paths['rf_importance'] = _savefig('04_rf_feature_importance.png')

    # --- XGBoost importance ---
    if 'xgboost' in models:
        xgb = models['xgboost']['model']
        imp = pd.Series(xgb.feature_importances_, index=feature_names).sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
        imp[::-1].plot.barh(color=DANGER, ax=ax, edgecolor='white')
        ax.set_title('XGBoost — Feature importances (gain)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Feature importance score')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        paths['xgb_importance'] = _savefig('05_xgb_feature_importance.png')

    # --- Logistic Regression coefficients ---
    if 'logistic_regression' in models:
        lr = models['logistic_regression']['model']
        coef = pd.Series(np.abs(lr.coef_[0]), index=feature_names).sort_values(ascending=False).head(top_n)
        fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
        coef[::-1].plot.barh(color=SUCCESS, ax=ax, edgecolor='white')
        ax.set_title('Logistic Regression — Absolute coefficients', fontsize=12, fontweight='bold')
        ax.set_xlabel('|Coefficient|')
        ax.spines[['top','right']].set_visible(False)
        plt.tight_layout()
        paths['lr_importance'] = _savefig('06_lr_coefficients.png')

    # --- SHAP (XGBoost) ---
    best_tree = models.get('xgboost') or models.get('random_forest')
    if best_tree:
        model = best_tree['model']
        sample = X_test.iloc[:min(500, len(X_test))]  # cap for speed
        try:
            print("  [model] Computing SHAP values (may take a moment)...")
            explainer   = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

            # For binary classification, shap_values may be list[2] or ndarray
            if isinstance(shap_values, list):
                sv = shap_values[1]
            else:
                sv = shap_values

            fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))
            shap.summary_plot(sv, sample, feature_names=feature_names,
                              max_display=top_n, show=False, plot_type='bar')
            plt.title('SHAP Feature Importance (mean |SHAP value|)', fontsize=12, fontweight='bold')
            plt.tight_layout()
            paths['shap_bar'] = _savefig('07_shap_importance.png')

            # SHAP beeswarm
            fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.45)))
            shap.summary_plot(sv, sample, feature_names=feature_names,
                              max_display=top_n, show=False)
            plt.title('SHAP Values — Impact on churn prediction', fontsize=12, fontweight='bold')
            plt.tight_layout()
            paths['shap_dot'] = _savefig('08_shap_beeswarm.png')

        except Exception as e:
            print(f"  [model] SHAP computation failed: {e} — skipping.")

    return paths


# ── Risk scoring ──────────────────────────────────────────────────────────────

def score_customers(df_original: pd.DataFrame, X: pd.DataFrame,
                    models: dict, eval_results: dict) -> pd.DataFrame:
    """
    Score all customers with churn probability from the best-performing model.
    Returns a DataFrame sorted by risk descending.
    """
    # Pick best model by AUC
    best_name = max(
        [k for k in eval_results if not k.startswith('_')],
        key=lambda k: eval_results[k]['auc']
    )
    bundle = models[best_name]
    model  = bundle['model']
    scaler = bundle['scaler']
    X_eval = scaler.transform(X) if bundle['needs_scaling'] else X

    probs = model.predict_proba(X_eval)[:, 1]

    scored = df_original.copy().reset_index(drop=True)
    scored['churn_probability'] = np.round(probs, 4)
    scored['risk_level'] = np.where(
        probs >= RISK_THRESHOLDS['high'],   'High',
        np.where(probs >= RISK_THRESHOLDS['medium'], 'Medium', 'Low')
    )
    scored['model_used'] = best_name

    scored = scored.sort_values('churn_probability', ascending=False)

    out_path = Path(REPORT_CONFIG['output_dir']) / 'at_risk_customers.csv'
    scored.to_csv(out_path, index=False)
    print(f"\n  [model] Risk scores saved → {out_path}")
    print(f"          Best model: {best_name} (AUC={eval_results[best_name]['auc']:.3f})")
    high_risk = (probs >= RISK_THRESHOLDS['high']).sum()
    print(f"          High-risk customers: {high_risk:,} ({high_risk/len(probs):.1%})")

    return scored


# ── Model comparison summary plot ─────────────────────────────────────────────

def plot_model_comparison(eval_results: dict) -> str:
    metrics = ['auc', 'avg_precision', 'brier', 'cv_auc']
    model_names = [k for k in eval_results if not k.startswith('_')]

    data = {m: [eval_results[mn].get(m, 0) for mn in model_names] for m in metrics}
    labels = [n.replace('_', ' ').title() for n in model_names]

    x = np.arange(len(labels))
    width = 0.2
    colors = [PRIMARY, DANGER, SUCCESS, WARN]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric, values) in enumerate(data.items()):
        bars = ax.bar(x + i*width, values, width, label=metric.upper().replace('_',' '),
                      color=colors[i], alpha=0.85, edgecolor='white')

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title('Model performance comparison', fontsize=13, fontweight='bold')
    ax.set_ylabel('Score')
    ax.legend(fontsize=9)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1, alpha=0.5, label='Baseline')
    ax.spines[['top','right']].set_visible(False)
    plt.tight_layout()
    return _savefig('09_model_comparison.png')


# ── Main runner ───────────────────────────────────────────────────────────────

def run_modelling(df: pd.DataFrame) -> dict:
    """
    Full modelling pipeline.
    Returns dict with models, results, feature importance paths, and scored customers.
    """
    print("\n[model] Starting modelling pipeline...")

    # Feature engineering
    df_feat = engineer_features(df)

    # Build feature matrix
    X, y, feature_names = build_feature_matrix(df_feat)
    print(f"  [model] Feature matrix: {X.shape[0]:,} rows × {X.shape[1]} features")

    # Train/test split
    rs = MODEL_CONFIG['random_state']
    ts = MODEL_CONFIG['test_size']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ts, stratify=y, random_state=rs
    )
    print(f"  [model] Train: {len(X_train):,} | Test: {len(X_test):,}")

    # SMOTE
    X_train_b, y_train_b = maybe_apply_smote(X_train, y_train)

    # Train
    models = train_models(X_train_b, y_train_b)

    # Evaluate
    print("\n[model] Evaluating models on hold-out test set...")
    eval_results = evaluate_models(models, X_test, y_test, feature_names)

    # Feature importance
    print("\n[model] Computing feature importances...")
    fi_paths = plot_feature_importance(models, X_test, feature_names)

    # Model comparison
    comp_path = plot_model_comparison(eval_results)

    # Risk scoring
    scored = score_customers(df_feat, X, models, eval_results)

    return {
        'models':        models,
        'eval_results':  eval_results,
        'feature_names': feature_names,
        'fi_paths':      fi_paths,
        'comparison':    comp_path,
        'scored':        scored,
        'X_test':        X_test,
        'y_test':        y_test,
    }
