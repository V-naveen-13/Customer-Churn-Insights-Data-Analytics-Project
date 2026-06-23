"""
generate_sample_data.py — Creates realistic synthetic churn data for testing.

Run: python generate_sample_data.py
Produces: data/sample_customers.csv (2,000 rows)

The data mirrors a typical B2B SaaS dataset with realistic churn patterns:
  - SMB customers churn more than Enterprise
  - Monthly contracts churn more than annual
  - Low usage and high support tickets predict churn
  - New customers (< 3 months) have elevated churn
"""

import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)
N = 2000

segments       = np.random.choice(['SMB', 'Mid-Market', 'Enterprise'], N, p=[0.55, 0.30, 0.15])
plans          = np.where(segments == 'Enterprise',
                          np.random.choice(['Pro', 'Enterprise'], N, p=[0.3, 0.7]),
                          np.where(segments == 'Mid-Market',
                                   np.random.choice(['Starter', 'Pro', 'Enterprise'], N, p=[0.2, 0.6, 0.2]),
                                   np.random.choice(['Starter', 'Pro'], N, p=[0.7, 0.3])))

contract_types = np.where(segments == 'Enterprise',
                          np.random.choice(['Annual', 'Multi-year'], N, p=[0.5, 0.5]),
                          np.where(segments == 'Mid-Market',
                                   np.random.choice(['Monthly', 'Annual'], N, p=[0.35, 0.65]),
                                   np.random.choice(['Monthly', 'Annual'], N, p=[0.65, 0.35])))

regions   = np.random.choice(['North America', 'Europe', 'APAC', 'LATAM'], N, p=[0.45, 0.30, 0.15, 0.10])
tenure    = np.random.exponential(18, N).clip(0.5, 72).round(1)

# MRR depends on segment
mrr_base  = np.where(segments == 'Enterprise', 2000, np.where(segments == 'Mid-Market', 500, 80))
mrr       = (mrr_base * np.random.lognormal(0, 0.4, N)).clip(20, 25000).round(2)

logins    = np.random.exponential(12, N).clip(0, 60).round(1)
days_last = np.random.exponential(15, N).clip(0, 120).round(0).astype(int)
features  = np.random.randint(1, 20, N)
tickets   = np.random.poisson(1.5, N)
nps       = np.random.choice(range(0, 11), N, p=[0.05,0.03,0.04,0.06,0.06,0.07,0.09,0.12,0.18,0.15,0.15])
csat      = np.random.choice(range(1, 6), N, p=[0.05,0.08,0.15,0.37,0.35])
products  = np.random.randint(1, 6, N)

# Churn probability — driven by realistic predictors
logit = (
    -2.5                                                   # baseline
    + 0.9  * (segments == 'SMB').astype(float)
    - 0.6  * (segments == 'Enterprise').astype(float)
    + 0.8  * (contract_types == 'Monthly').astype(float)
    - 0.7  * (contract_types == 'Multi-year').astype(float)
    - 0.04 * logins
    + 0.03 * days_last
    + 0.15 * tickets
    - 0.15 * nps
    - 0.3  * (tenure > 24).astype(float)
    + 0.5  * (tenure < 3).astype(float)
    - 0.02 * features
    + np.random.normal(0, 0.5, N)         # noise
)

prob_churn = 1 / (1 + np.exp(-logit))
churn = (np.random.uniform(0, 1, N) < prob_churn).astype(int)

start_dates_arr = pd.date_range('2021-01-01', periods=N, freq='6h').to_numpy().copy()
np.random.shuffle(start_dates_arr)
start_dates = start_dates_arr[:N]

df = pd.DataFrame({
    'CustomerID':        [f'CUST-{i:05d}' for i in range(1, N+1)],
    'Segment':           segments,
    'Plan':              plans,
    'ContractType':      contract_types,
    'Region':            regions,
    'tenure_months':     tenure,
    'MRR':               mrr,
    'LoginsPerMonth':    logins,
    'DaysSinceLastLogin': days_last,
    'FeaturesUsed':      features,
    'SupportTickets':    tickets,
    'NPSScore':          nps,
    'CSATScore':         csat,
    'NumProducts':       products,
    'StartDate':         pd.to_datetime(start_dates).strftime('%Y-%m-%d'),
    'Churn':             churn,
})

Path('data').mkdir(exist_ok=True)
df.to_csv('data/sample_customers.csv', index=False)

print(f"Sample data saved → data/sample_customers.csv")
print(f"Rows: {len(df):,} | Churn rate: {df['Churn'].mean():.1%}")
print(df.head())
