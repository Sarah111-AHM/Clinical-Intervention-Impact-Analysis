# 🏥 Clinical Intervention Analysis

A fully professional data science project for analysing clinical trial data comparing **Control** and **Intervention** patient groups. Includes end-to-end analysis pipelines, statistical testing, machine learning models, and an interactive Streamlit web application.

---

## 📊 Project Overview

This project analyses a clinical dataset of **120 patients** (67 Control, 53 Intervention) to:

- Compare outcomes between treatment groups
- Identify statistically significant differences
- Predict patient complications and long hospital stays
- Provide interactive visualisations via a web dashboard

### Key Findings

| Metric | Control | Intervention |
|--------|---------|-------------|
| Avg Hospital Stay | 7.3 days | 5.3 days |
| Complication Rate | ~27% | ~25% |
| Sample Size | 67 | 53 |

**The intervention significantly reduces hospital stay duration (p < 0.001) while maintaining comparable complication rates.**

---

## 📁 Project Structure

```
clinical_intervention/
├── data/
│   └── competition_dataset.csv      # Clinical trial dataset
├── notebooks/
│   └── clinical_analysis.ipynb     # Full analysis walkthrough
├── src/
│   ├── data_preprocessing.py       # Data loading, cleaning, feature engineering
│   ├── descriptive_analysis.py     # Stats, visualisations, hypothesis tests
│   ├── ml_models.py               # ML training, evaluation, feature importance
│   └── run_analysis.py            # Main runner script
├── app/
│   └── streamlit_app.py           # Interactive web application
├── reports/
│   ├── age_distribution.png
│   ├── hospital_stay_boxplot.png
│   ├── complication_rates.png
│   ├── correlation_heatmap.png
│   ├── scatter_age_stay.png
│   ├── sex_distribution.png
│   ├── roc_curves_complication.png
│   ├── model_comparison_complication.png
│   ├── feature_importance_complication.png
│   └── ...
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start — Run Locally

### 1. Clone & install

```bash
git clone <your-repo-url>
cd clinical_intervention
pip install -r requirements.txt
```

### 2. Run the full analysis pipeline

```bash
python src/run_analysis.py
```

This will:
- Load and clean the dataset
- Print descriptive statistics and test results
- Save 14+ plots to `/reports/`
- Train 3 ML models for 2 prediction targets

### 3. Launch the Streamlit web app

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501 in your browser.

### 4. Open the Jupyter notebook

```bash
jupyter notebook notebooks/clinical_analysis.ipynb
```

---

## 🧠 Machine Learning Models

Three classifiers are trained with **5-fold stratified cross-validation** and **GridSearchCV** hyperparameter tuning:

| Model | Description |
|-------|-------------|
| **Logistic Regression** | Regularised linear classifier (C tuning) |
| **Random Forest** | Ensemble of 50–200 decision trees |
| **Gradient Boosting** | Sequential boosting (sklearn GBM) |

### Prediction Targets

1. **Complication** — did the patient experience a complication? (Yes/No)
2. **Long Stay** — did the patient stay > 7 days?

### Evaluation Metrics

- Accuracy, Precision, Recall, F1-Score
- ROC-AUC with ROC curve plots
- Confusion matrices
- Feature importance rankings

---

## 🌐 Web Application Features

The Streamlit dashboard includes:

- **📊 Overview** — data quality report, descriptive stats, group comparison table
- **📈 EDA Visualisations** — interactive Plotly histograms, boxplots, scatter plots, heatmap
- **🔬 Statistical Tests** — Welch t-test and chi-square results with significance flags
- **🤖 ML Models** — train, compare, and evaluate models interactively
- **🎯 Patient Predictions** — enter patient data to get a real-time complication risk score

---

## ☁️ Deploy on Streamlit Community Cloud (Free)

> Streamlit Community Cloud is the recommended free deployment option. Vercel is designed for Node.js apps; Streamlit apps deploy best on Streamlit Cloud.

### Steps

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/clinical-intervention
   git push -u origin main
   ```

2. **Go to** https://share.streamlit.io → click **New app**

3. **Settings:**
   - Repository: `<your-username>/clinical-intervention`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`

4. Click **Deploy** — done! App is live within 2 minutes.

### Deploy on Vercel (via Dockerfile)

If you specifically need Vercel, create a `Dockerfile`:

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Then deploy via Vercel's Docker support or use **Railway** / **Render** for the simplest Python app deployment.

---

## 📦 Dependencies

| Library | Version | Purpose |
|---------|---------|---------|
| pandas | ≥2.0 | Data manipulation |
| numpy | ≥1.24 | Numerical computing |
| scipy | ≥1.10 | Statistical tests |
| scikit-learn | ≥1.3 | ML models |
| matplotlib / seaborn | latest | Static plots |
| plotly | ≥5.15 | Interactive plots |
| streamlit | ≥1.28 | Web app |

---

## 📌 Dataset Description

| Column | Type | Description |
|--------|------|-------------|
| `age` | int | Patient age in years |
| `sex` | str | Male / Female |
| `group` | str | Control / Intervention |
| `hospital_stay` | int | Length of stay in days |
| `complication` | str | Yes / No |

**Engineered features:**
- `age_group` — binned age bracket (<40, 40-55, 55-65, 65+)
- `sex_enc` — binary (Male=1)
- `group_enc` — binary (Intervention=1)
- `complication_bin` — binary (Yes=1)
- `stay_long` — binary (stay > 7 days)

---

## 📜 License

MIT — free to use and modify.
