# Customer Churn Insights - Data Analytics Project

A data analytics pipeline to identify at-risk customers and generate actionable churn insights.

## Project Overview

This project analyzes customer data to predict churn risk, identify key churn drivers, and generate comprehensive reports with visualizations and recommendations.

## Features

- **Data Ingestion** — Load and validate customer data from CSV
- **Exploratory Data Analysis (EDA)** — Statistical analysis and visualizations
- **Churn Modeling** — Predictive models to identify at-risk customers
- **Report Generation** — HTML reports with insights and recommendations
- **Demo Mode** — Run without data for testing

## Project Structure

```
churn_pipeline/
├── config.py              # Configuration & column mappings
├── ingest.py              # Data loading & validation
├── eda.py                 # Exploratory data analysis
├── model.py               # Churn prediction models
├── report.py              # HTML report generation
├── main.py                # Main pipeline orchestration
├── generate_sample_data.py # Sample data generation
├── requirements.txt       # Python dependencies
├── data/                  # Input customer data
├── outputs/               # Generated reports & predictions
└── reports/               # Report artifacts
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# 1. Navigate to project directory
cd churn_pipeline

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure column mappings
# Edit config.py and map COL{} keys to your CSV column names
```

## Usage

### Run with Your Data

```bash
python main.py --data your_customers.csv --company "Your Company Name"
```

### Run Demo Mode (No Data Required)

```bash
python main.py --demo
```

### Command Line Arguments

- `--data` — Path to customer CSV file
- `--company` — Company name for report header
- `--demo` — Run in demo mode with sample data

## Configuration

Edit `config.py` to map your CSV column names:

```python
COL_CUSTOMER_ID = "customer_id"
COL_TENURE = "tenure_months"
COL_MONTHLY_CHARGE = "monthly_charges"
# ... map other columns
```

## Output

- **at_risk_customers.csv** — List of high-risk churn customers
- **churn_analysis_report.html** — Interactive HTML report with visualizations
- **EDA Plots** — Statistical analysis charts in `outputs/eda/`
- **Model Data** — Model predictions in `outputs/models/`

## Requirements

See `requirements.txt` for full dependencies (pandas, scikit-learn, plotly, etc.)

## License

This project is part of an MSc Data Science program.

## Contact

For questions, reach out to the project owner.