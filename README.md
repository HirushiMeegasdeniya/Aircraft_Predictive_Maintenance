# ✈️ Aircraft Predictive Maintenance & Reliability Analytics

## 📌 Project Overview

This project is an end-to-end Data Science, Reliability Engineering, and Business Intelligence solution that predicts aircraft engine failures before they occur using the NASA CMAPSS Turbofan Engine Dataset.

The platform combines:

- Data Cleaning & Processing
- Feature Engineering
- Exploratory Data Analysis (EDA)
- Remaining Useful Life (RUL) Prediction
- Reliability Analytics
- Failure Risk Assessment
- Maintenance Scheduling
- Interactive Power BI Dashboard

The objective is to help airlines and maintenance organizations proactively identify degrading engines, reduce downtime, optimize maintenance schedules, and improve fleet reliability.

---

# 🎯 Business Problem

Aircraft engine failures can lead to:

- Unexpected downtime
- Flight delays and cancellations
- Increased maintenance costs
- Reduced aircraft availability
- Safety risks
- Inefficient maintenance planning

Traditional maintenance strategies are often reactive or time-based.

Predictive Maintenance enables organizations to:

- Predict failures before they occur
- Schedule maintenance proactively
- Reduce maintenance costs
- Improve reliability
- Increase aircraft availability

---

# 📊 Dataset

## NASA CMAPSS Turbofan Engine Dataset

Source:

https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository

The dataset contains simulated run-to-failure data for aircraft turbofan engines.

Each engine includes:

- Engine ID
- Operational Cycles
- 3 Operational Settings
- 21 Sensor Measurements

### FD001 Subset Used

Characteristics:

- Single Operating Condition
- Single Fault Mode
- Simplest CMAPSS Dataset
- Ideal for predictive maintenance development

---

# 🏗️ System Architecture

```text
NASA CMAPSS Dataset
          │
          ▼
Data Loading & Cleaning
          │
          ▼
Exploratory Data Analysis
          │
          ▼
Feature Engineering
          │
          ▼
Remaining Useful Life Prediction
          │
          ▼
Failure Risk Classification
          │
          ▼
Reliability Analytics
          │
          ▼
Maintenance Scheduling
          │
          ▼
Power BI Dashboard
```

### Architecture Diagram

![Architecture Diagram]("outputs\charts\Architecture diagram.png")

---

# 📂 Project Structure

```text
Aircraft_Predictive_Maintenance/
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── sequences/
│   └── dashboard/
│
├── docs/
│   └── images/
│
├── notebooks/
│
├── models/
│
├── outputs/
│   ├── charts/
│   ├── predictions/
│   └── reports/
│
├── src/
│   ├── 01_load_data.py
│   ├── 02_explore_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_build_rul_model.py
│   ├── 05_reliability_analysis.py
│   ├── 06_predict_failure_risk.py
│   ├── 07_maintenance_scheduler.py
│   └── 08_export_powerbi.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

# 🔬 Exploratory Data Analysis

EDA was performed to identify:

- Sensor degradation patterns
- Correlation with Remaining Useful Life
- Failure-related sensor behavior
- Engine health trends

### Correlation Heatmap

![Correlation Heatmap](outputs\charts\correlation_heatmap.png)

---

# ⚙️ Feature Engineering

Feature Engineering Steps:

- RUL Target Creation
- Sensor Normalization
- Time Window Generation
- Sequence Construction
- Feature Scaling
- Risk Categorization

Configuration:

```text
Window Size = 30 Cycles
Maximum RUL = 125 Cycles
```

---

# 🤖 Machine Learning Models

## Random Forest Regressor

| Metric | Value |
|----------|----------|
| MAE | 11.96 |
| RMSE | 15.87 |
| R² | 0.851 |

---

## XGBoost Regressor

| Metric | Value |
|----------|----------|
| MAE | 10.24 |
| RMSE | 13.94 |
| R² | 0.885 |

---

## Final Model Selection

✅ XGBoost Regressor

Reason:

- Lowest MAE
- Lowest RMSE
- Highest R²
- Best overall predictive performance

---

# 📉 Model Validation

## Actual vs Predicted RUL

![Actual vs Predicted RUL](outputs\charts\actual_vs_predicted_rul.png)

---

## Residual Analysis

![Residual Analysis](outputs\charts\residual_analysis.png)

---

# 📈 Reliability Analytics

The project incorporates Reliability Engineering techniques including:

- Mean Time Between Failures (MTBF)
- Weibull Distribution Analysis
- Hazard Function Analysis
- Kaplan-Meier Survival Analysis
- Reliability Curves

---

## Reliability Curve

![Reliability Curve](outputs\charts\reliability_curve.png)

---

## Kaplan-Meier Survival Curve

![Kaplan-Meier Survival Curve](outputs\charts\kaplan_meier_curve.png)

---

# 🚨 Failure Risk Classification

| Risk Level | RUL Range | Action |
|------------|------------|------------|
| 🟢 Green | > 100 Cycles | Healthy |
| 🟡 Yellow | 30 - 100 Cycles | Monitor |
| 🟠 Orange | 10 - 30 Cycles | Schedule Maintenance |
| 🔴 Red | < 10 Cycles | Immediate Inspection |

---

# 🔧 Maintenance Scheduling

The system automatically generates:

- Maintenance Priority Ranking
- Failure Forecasting
- Downtime Estimation
- Maintenance Recommendations
- Cost Saving Estimates

Outputs:

```text
maintenance_schedule.csv
engine_rul_predictions.csv
reliability_metrics.csv
```

---

# 📊 Power BI Dashboard

The project includes a professional multi-page dashboard for maintenance decision support.

---

## Page 1 — Fleet Overview

Features:

- Total Engines
- Average Fleet RUL
- Fleet Reliability
- Risk Breakdown
- Fleet Health Distribution

![Fleet Overview](docs/images/fleet_overview.png)

---

## Page 2 — Engine Health Monitor

Features:

- Engine Selection
- Sensor Trends
- RUL Tracking
- Degradation Monitoring

![Engine Health Monitor](docs/images/engine_health.png)

---

## Page 3 — Failure Risk Intelligence

Features:

- Risk Distribution
- High-Risk Engines
- Failure Risk Matrix
- Priority Analysis

![Failure Risk Intelligence](docs/images/risk_intelligence.png)

---

## Page 4 — RUL Prediction Analytics

Features:

- Predicted RUL
- Failure Cycle Forecast
- Model Validation
- Top Risk Engines

![RUL Analytics](docs/images/rul_analytics.png)

---

## Page 5 — Maintenance Planner

Features:

- Maintenance Schedule
- Downtime Forecast
- Cost Saving Analysis
- Maintenance Prioritization

![Maintenance Planner](docs/images/maintenance_planner.png)

---

## Page 6 — Reliability Analytics

Features:

- MTBF
- Weibull Analysis
- Reliability Curves
- Survival Analysis

![Reliability Analytics](docs/images/reliability_analytics.png)

---

# 📁 Generated Outputs

## Charts

```text
outputs/charts/
```

Examples:

- Correlation Heatmap
- Sensor Trends
- Actual vs Predicted RUL
- Residual Analysis
- Reliability Curve
- Kaplan-Meier Curve

---

## Predictions

```text
outputs/predictions/
```

Examples:

- Engine RUL Predictions
- Failure Risk Predictions

---

## Reports

```text
outputs/reports/
```

Examples:

- Model Validation Report
- Reliability Report
- Maintenance Schedule

---

# 🚀 How To Run

## Activate Environment

```bash
venv\Scripts\activate
```

## Run Complete Pipeline

```bash
python src/01_load_data.py
python src/02_explore_eda.py
python src/03_feature_engineering.py
python src/04_build_rul_model.py
python src/05_reliability_analysis.py
python src/06_predict_failure_risk.py
python src/07_maintenance_scheduler.py
python src/08_export_powerbi.py
```

---

# 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Seaborn
- Lifelines
- Power BI
- Git
- GitHub

---

# 🔮 Future Improvements

Potential Enhancements:

- LSTM-Based RUL Prediction
- Deep Learning Models
- Real-Time Monitoring Simulation
- Streamlit Web Application
- Isolation Forest Anomaly Detection
- Cost Optimization Algorithms
- Support for FD002–FD004 CMAPSS Datasets

---

# 👩‍💻 Author

**Hirushi Sharanya Meegasdeniya**

Bachelor of Science (Hons) in Information Technology  
Specialization: Data Science  
Sri Lanka Institute of Information Technology (SLIIT)

---

# 📜 License

This project is developed for educational, research, and portfolio purposes.