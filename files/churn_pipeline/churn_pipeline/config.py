"""
config.py — Central configuration for the churn analysis pipeline.

BEFORE RUNNING: Edit the COL dictionary to map internal names to your
actual CSV column names. Everything else is optional to tune.
"""

# ── Column name mapping ───────────────────────────────────────────────────────
# Keys are internal pipeline names. Values are your CSV column headers.
# Set a value to None if you don't have that field — the pipeline will skip it.

COL = {
    # Required
    'churn':            'Churn',           # 0/1, True/False, Yes/No, Churned/Active

    # Identity / Segmentation
    'customer_id':      'CustomerID',
    'segment':          'Segment',         # e.g. SMB, Mid-Market, Enterprise
    'region':           'Region',
    'industry':         'Industry',
    'plan':             'Plan',            # e.g. Basic, Pro, Enterprise

    # Lifecycle
    'tenure':           'tenure_months',   # numeric, months as customer
    'start_date':       'StartDate',       # date string, used to derive tenure if not present
    'churn_date':       'ChurnDate',       # date string, date of churn (optional)
    'contract_type':    'ContractType',    # Monthly, Annual, Multi-year

    # Financials
    'mrr':              'MRR',             # Monthly recurring revenue
    'arr':              None,              # Annual recurring revenue (or derived from MRR)
    'monthly_revenue':  None,              # Alternative revenue field
    'num_products':     'NumProducts',

    # Engagement / Behaviour
    'logins_per_month': 'LoginsPerMonth',
    'days_since_login': 'DaysSinceLastLogin',
    'features_used':    'FeaturesUsed',
    'usage_score':      'UsageScore',      # 0-100 composite usage metric if available

    # Support
    'support_tickets':  'SupportTickets',
    'nps_score':        'NPSScore',        # Net Promoter Score
    'csat_score':       'CSATScore',       # Customer Satisfaction score

    # Optional demographics
    'age':              None,
    'gender':           None,
    'country':          None,
}

# ── Required vs optional columns ─────────────────────────────────────────────
# Pipeline will error if required cols are missing; warn for optional

REQUIRED_COLS = ['churn']

OPTIONAL_COLS = [
    'customer_id', 'segment', 'region', 'plan', 'tenure',
    'contract_type', 'mrr', 'logins_per_month', 'support_tickets',
]

# ── Type hints for cleaning ───────────────────────────────────────────────────

NUMERIC_COLS = [
    'tenure', 'mrr', 'arr', 'monthly_revenue', 'logins_per_month',
    'days_since_login', 'features_used', 'usage_score',
    'support_tickets', 'nps_score', 'csat_score',
    'num_products', 'age',
]

CATEGORICAL_COLS = [
    'segment', 'region', 'industry', 'plan', 'contract_type',
    'gender', 'country',
]

# ── Modelling settings ────────────────────────────────────────────────────────

MODEL_CONFIG = {
    'test_size':         0.2,
    'random_state':      42,
    'cv_folds':          5,
    'apply_smote':       True,     # Oversample minority class if imbalanced
    'smote_threshold':   0.3,      # Apply SMOTE when churn rate < 30%
    'feature_importance_top_n': 15,

    'logistic_regression': {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'class_weight': 'balanced',
    },

    'random_forest': {
        'n_estimators': 300,
        'max_depth': None,
        'min_samples_leaf': 5,
        'class_weight': 'balanced',
        'n_jobs': -1,
    },

    'xgboost': {
        'n_estimators': 300,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'eval_metric': 'logloss',
        'use_label_encoder': False,
    },
}

# ── Report settings ───────────────────────────────────────────────────────────

REPORT_CONFIG = {
    'company_name':  'Your Company',
    'output_dir':    'outputs',
    'report_title':  'Customer Churn Analysis Report',
    'logo_path':     None,          # Optional: path to logo PNG
    'color_primary': '#378add',
    'color_danger':  '#e24b4a',
    'color_success': '#1d9e75',
    'color_warn':    '#ba7517',
    'color_neutral': '#888780',
}

# ── Risk score thresholds ─────────────────────────────────────────────────────

RISK_THRESHOLDS = {
    'high':   0.70,    # churn probability >= 70% → High risk
    'medium': 0.40,    # 40-70% → Medium risk
    # below 40% → Low risk
}
