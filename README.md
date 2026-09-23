# 🏥 Healthcare Operations, Revenue Optimization & Predictive Analytics

**Author:** Sambita Dutta | **Internship Submission**
**Dataset:** [Kaggle – Healthcare Dataset](https://www.kaggle.com/datasets/prasad22/healthcare-dataset)
**Records:** 54,860 patient admissions

---

## 📌 Project Overview

A production-ready Python web application combining **operational performance tracking**, **financial/revenue cycle management**, and **predictive machine learning** for clinical test outcomes and billing estimates. Built with Streamlit for the interactive frontend and Scikit-Learn for ML engines.

---

## 🗂️ Repository Structure

```
healthcare_fraud/
├── app.py                              # Streamlit web application (main entry point)
├── data_pipeline.py                    # Data loading, feature engineering, ML training & inference
├── SambitaDutta_HealthcareAnalytics.ipynb  # Full Jupyter Notebook (EDA + ML + demo)
├── generate_report.py                  # Generates SambitaDutta_ProjectReport.docx
├── cleaned_healthcare_data.csv         # Input dataset (54,860 records)
├── requirements.txt                    # Pinned Python dependencies
└── README.md                           # This file
```

---

## 🛠️ Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| Web UI | Streamlit ≥ 1.35 |
| Data Processing | Pandas, NumPy |
| Machine Learning | Scikit-Learn (Gradient Boosting) |
| Visualisation | Plotly Express / Graph Objects, Matplotlib, Seaborn |
| Report Generation | python-docx |

---

## 📊 Dataset Description

| Column | Description |
|--------|-------------|
| Name, Age, Gender, Blood Type | Patient demographics |
| Medical Condition | Primary diagnosis (Cancer, Diabetes, Asthma, Hypertension, Arthritis, Obesity) |
| Admission Type | Emergency / Elective / Urgent |
| Date of Admission, Discharge Date | Admission window |
| Length_of_Stay | Derived days between admission and discharge |
| Billing Amount, Daily_Cost | Financial fields; Daily_Cost = Billing Amount / Length_of_Stay |
| Insurance Provider | Aetna, Blue Cross, Cigna, Medicare, UnitedHealthcare |
| Test Results | Target for classification: Normal / Abnormal / Inconclusive |
| Age Group | Binned: Child / Adult / Middle-Aged / Senior |

---

## ⚙️ Setup & Installation

### 1. Clone / Download the repository
```bash
git clone <repo-url>
cd healthcare_fraud
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Streamlit app
```bash
streamlit run app.py
```
The app opens automatically at `http://localhost:8501`.

### 5. Run the Jupyter Notebook
```bash
jupyter notebook SambitaDutta_HealthcareAnalytics.ipynb
```

### 6. Generate the Word report
```bash
python generate_report.py
# Output: SambitaDutta_ProjectReport.docx
```

---

## 🏗️ Application Architecture

```
cleaned_healthcare_data.csv
        │
        ▼
data_pipeline.py
  ├── load_data()            – CSV → DataFrame + feature engineering
  ├── apply_filters()        – sidebar filter logic
  ├── train_classifier()     – GBM multi-class (Test Results)
  ├── train_regressor()      – GBM regression (Billing Amount, Daily Cost)
  └── predict_patient()      – single-row inference + anomaly flags
        │
        ▼
app.py  (Streamlit)
  ├── Sidebar: date range, insurance, hospital, admission type, condition
  ├── Tab 1: Executive & Operational Dashboard
  │     ├── KPIs: Patients, Revenue, Avg Billing, Avg LOS, Emergency %
  │     ├── LOS Heatmap (Condition × Admission Type)
  │     ├── Hospital Workload bar charts
  │     └── Admission Trend line chart
  ├── Tab 2: Financial & Revenue Cycle Analytics
  │     ├── KPIs: Avg Daily Cost, Highest Billing Condition, Dominant Insurer
  │     ├── Billing Violin plots by Insurance Provider
  │     ├── High-Cost / LOS Anomaly table & charts
  │     └── Daily Cost grouped bar (Condition × Age Group)
  └── Tab 3: Clinical Outcomes & ML Portal
        ├── Test Result distribution (Age Group, Condition)
        ├── Classifier metrics: accuracy, precision, recall, F1, confusion matrix
        ├── Regressor metrics: RMSE, MAE, R² + feature importances
        └── Patient Inference Form → predicted outcome + billing + risk flags
```

---

## 🤖 Machine Learning Models

### Engine 1 – Test Outcome Classifier
- **Algorithm:** Gradient Boosting Classifier (n_estimators=200, lr=0.08, max_depth=4)
- **Target:** Test Results (Normal / Abnormal / Inconclusive)
- **Features:** Age, Length_of_Stay, Age Group, Gender, Medical Condition, Admission Type, Medication
- **Preprocessing:** OrdinalEncoder (Age Group), OneHotEncoder (nominal), passthrough (numeric)
- **Evaluation:** Confusion Matrix, Precision, Recall, F1-Score per class

### Engine 2 – Billing & Cost Regressors
- **Algorithm:** Gradient Boosting Regressor (same hyperparameters)
- **Targets:** Billing Amount (total), Daily_Cost (per day)
- **Features:** Age, Length_of_Stay, Gender, Medical Condition, Admission Type, Medication, Insurance Provider, Age Group
- **Evaluation:** RMSE, MAE, R²
- **Daily Cost R²:** ~0.68 | Billing Amount R²: ~0.00 *(synthetic dataset — billing is random)*

---

## 💡 Key Business Insights

1. **Uniform Test Result Distribution** – Each outcome class (Normal/Abnormal/Inconclusive) represents exactly 1/3 of records, confirming synthetic data generation; real-world models would benefit from actual clinical signal.
2. **Emergency Share ~33%** – A third of all admissions are emergency, indicating high acute-care demand warranting staffing buffer planning.
3. **Length of Stay** – Average ~15 days; top 10% exceed the 90th-percentile threshold and are automatically flagged for care coordination review.
4. **Daily Cost Predictability** – The GBM regressor achieves R²≈0.68 for Daily_Cost, driven primarily by Length_of_Stay and Medical Condition — actionable for pre-admission cost estimation.
5. **Insurance Parity** – All five insurance providers show nearly equal patient share (~20% each), suggesting no dominant payer concentration risk.
6. **High-Cost Anomalies** – ~10% of patients trigger HighCost_Flag; Cancer and Asthma patients exhibit the highest average daily costs across age groups.

---

## 📄 License

This project is submitted as part of a Data Analytics Internship. Dataset sourced from Kaggle under the original dataset license.
