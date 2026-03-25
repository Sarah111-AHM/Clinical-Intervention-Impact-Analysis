"""
Clinical Intervention Analysis — Streamlit Web Application
===========================================================
Interactive dashboard for:
  • Dataset overview & cleaning report
  • Descriptive statistics
  • EDA visualisations (Plotly)
  • Statistical group comparisons
  • ML model training & evaluation
  • Feature importance
"""

import io
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import streamlit as st
from pathlib import Path
from scipy import stats
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)

warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Clinical Intervention Analysis",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .main { background-color: #F8FAFC; }
    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1E3A5F; }
    [data-testid="stSidebar"] * { color: #E8F4FD !important; }
    /* Metric cards */
    [data-testid="stMetric"] {
        background: white;
        border-radius: 10px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #1E3A5F;
    }
    /* Title */
    h1 { color: #1E3A5F; font-weight: 800; }
    h2, h3 { color: #2C5282; }
    /* DataFrames */
    .stDataFrame { border-radius: 8px; overflow: hidden; }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #EBF4FF; border-radius: 10px; padding: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; padding: 8px 20px; font-weight: 600; }
    .stTabs [aria-selected="true"] { background: #1E3A5F !important; color: white !important; }
    /* Success/warning banners */
    .insight-box {
        background: #EBF8FF;
        border-left: 4px solid #3182CE;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Constants ─────────────────────────────────────────────────────────────────
PALETTE = {"Control": "#4C72B0", "Intervention": "#DD8452"}
RANDOM_STATE = 42
FEATURE_COLS   = ["age", "sex_enc", "group_enc", "hospital_stay"]
FEATURE_LABELS = ["Age", "Sex (Male=1)", "Intervention", "Hospital Stay"]


# ══════════════════════════════════════════════════════════════════════════════
# DATA HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def load_and_preprocess(uploaded) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    raw = pd.read_csv(uploaded)
    report = {
        "rows_raw": len(raw),
        "missing": raw.isnull().sum().to_dict(),
        "dtypes": raw.dtypes.astype(str).to_dict(),
    }
    df = raw.copy()
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip().str.capitalize()
    dups = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)
    report["duplicates_removed"] = int(dups)
    report["rows_clean"] = len(df)

    df["age_group"]       = pd.cut(df["age"], [0,40,55,65,120],
                                   labels=["<40","40-55","55-65","65+"], right=False)
    df["sex_enc"]         = (df["sex"] == "Male").astype(int)
    df["group_enc"]       = (df["group"] == "Intervention").astype(int)
    df["complication_bin"]= (df["complication"] == "Yes").astype(int)
    df["stay_long"]       = (df["hospital_stay"] > 7).astype(int)
    return raw, df, report


def stat_tests(df):
    ctrl = df[df["group"] == "Control"]
    intv = df[df["group"] == "Intervention"]
    rows = []
    for col in ["age", "hospital_stay"]:
        t, p = stats.ttest_ind(ctrl[col], intv[col], equal_var=False)
        rows.append(dict(Test="Welch t-test", Variable=col,
                         Statistic=round(t,4), pValue=round(p,4),
                         Significant="✅ Yes" if p<0.05 else "❌ No"))
    for col in ["complication", "sex"]:
        ct = pd.crosstab(df["group"], df[col])
        chi2, p, _, _ = stats.chi2_contingency(ct)
        rows.append(dict(Test="Chi-square", Variable=col,
                         Statistic=round(chi2,4), pValue=round(p,4),
                         Significant="✅ Yes" if p<0.05 else "❌ No"))
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def train_models(X_arr, y_arr, target_name):
    X = pd.DataFrame(X_arr, columns=FEATURE_COLS)
    y = pd.Series(y_arr)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    model_defs = {
        "Logistic Regression": (
            Pipeline([("sc", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))]),
            {"clf__C": [0.01,0.1,1,10], "clf__solver":["lbfgs","liblinear"]}
        ),
        "Random Forest": (
            Pipeline([("clf", RandomForestClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators":[50,100,200], "clf__max_depth":[None,3,5]}
        ),
        "Gradient Boosting": (
            Pipeline([("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))]),
            {"clf__n_estimators":[50,100], "clf__learning_rate":[0.05,0.1,0.2], "clf__max_depth":[2,3]}
        ),
    }

    results = []
    for name, (pipe, grid) in model_defs.items():
        gs = GridSearchCV(pipe, grid, cv=cv, scoring="roc_auc", n_jobs=-1, refit=True)
        gs.fit(X_tr, y_tr)
        best = gs.best_estimator_
        yp  = best.predict(X_te)
        ypr = best.predict_proba(X_te)[:,1]
        fpr, tpr, _ = roc_curve(y_te, ypr)
        clf = best.named_steps["clf"]
        if hasattr(clf, "feature_importances_"):
            imp = clf.feature_importances_
        elif hasattr(clf, "coef_"):
            imp = np.abs(clf.coef_[0])
        else:
            imp = np.ones(len(FEATURE_COLS))
        results.append({
            "name": name,
            "accuracy":  round(accuracy_score(y_te, yp), 4),
            "precision": round(precision_score(y_te, yp, zero_division=0), 4),
            "recall":    round(recall_score(y_te, yp, zero_division=0), 4),
            "f1":        round(f1_score(y_te, yp, zero_division=0), 4),
            "auc":       round(roc_auc_score(y_te, ypr), 4),
            "cm":        confusion_matrix(y_te, yp).tolist(),
            "fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "importance": imp.tolist(),
            "best_params": gs.best_params_,
        })
    results.sort(key=lambda r: r["auc"], reverse=True)
    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def plotly_theme():
    return dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif"),
        margin=dict(l=40,r=20,t=50,b=40),
    )


def fig_hist(df, col, title):
    fig = px.histogram(df, x=col, color="group", barmode="overlay",
                       color_discrete_map=PALETTE, nbins=20,
                       title=title, opacity=0.75,
                       labels={col: col.replace("_"," ").title(), "group":"Group"})
    fig.update_layout(**plotly_theme())
    return fig


def fig_box(df, y_col, title):
    fig = px.box(df, x="group", y=y_col, color="group",
                 color_discrete_map=PALETTE, points="all",
                 title=title, labels={"group":"Group", y_col: y_col.replace("_"," ").title()})
    fig.update_layout(**plotly_theme())
    return fig


def fig_scatter(df):
    fig = px.scatter(df, x="age", y="hospital_stay", color="group",
                     color_discrete_map=PALETTE, trendline="ols",
                     hover_data=["sex","complication"],
                     title="Age vs Hospital Stay",
                     labels={"age":"Age","hospital_stay":"Hospital Stay (days)","group":"Group"})
    fig.update_layout(**plotly_theme())
    return fig


def fig_heatmap(df):
    num = df[["age","hospital_stay","sex_enc","group_enc","complication_bin"]].copy()
    num.columns = ["Age","Hospital Stay","Sex","Intervention","Complication"]
    corr = num.corr().round(2)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, title="Correlation Matrix",
                    aspect="auto")
    fig.update_layout(**plotly_theme())
    return fig


def fig_complication_bar(df):
    ct = df.groupby(["group","complication"]).size().reset_index(name="count")
    tot = df.groupby("group")["complication"].count().reset_index(name="total")
    ct = ct.merge(tot, on="group")
    ct["pct"] = (ct["count"] / ct["total"] * 100).round(1)
    fig = px.bar(ct, x="group", y="pct", color="complication", barmode="group",
                 title="Complication Rate by Group (%)",
                 color_discrete_sequence=["#4CAF50","#F44336"],
                 labels={"pct":"%","group":"Group","complication":"Complication"},
                 text="pct")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(**plotly_theme(), yaxis_range=[0,100])
    return fig


def fig_age_pie(df):
    ct = df["age_group"].value_counts().reset_index()
    ct.columns = ["Age Group", "Count"]
    fig = px.pie(ct, names="Age Group", values="Count",
                 title="Age Group Distribution",
                 color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(**plotly_theme())
    return fig


def fig_roc(results):
    fig = go.Figure()
    colors = ["#4C72B0","#DD8452","#55A868"]
    for r, c in zip(results, colors):
        fig.add_trace(go.Scatter(
            x=r["fpr"], y=r["tpr"], mode="lines", name=f"{r['name']} (AUC={r['auc']:.3f})",
            line=dict(color=c, width=2)
        ))
    fig.add_trace(go.Scatter(x=[0,1],y=[0,1], mode="lines", name="Random",
                             line=dict(dash="dash", color="gray", width=1)))
    fig.update_layout(title="ROC Curves", xaxis_title="FPR", yaxis_title="TPR",
                      legend=dict(x=0.55, y=0.12), **plotly_theme())
    return fig


def fig_cm(result):
    cm = np.array(result["cm"])
    labels = ["No","Yes"]
    fig = ff.create_annotated_heatmap(
        cm, x=labels, y=labels,
        colorscale="Blues", showscale=False,
        annotation_text=cm.astype(str)
    )
    fig.update_layout(title=f"Confusion Matrix — {result['name']}",
                      xaxis_title="Predicted", yaxis_title="Actual",
                      **plotly_theme())
    return fig


def fig_feature_importance(results):
    data = []
    for r in results:
        for feat, imp in zip(FEATURE_LABELS, r["importance"]):
            data.append({"Model": r["name"], "Feature": feat, "Importance": imp})
    df_imp = pd.DataFrame(data)
    fig = px.bar(df_imp, x="Importance", y="Feature", color="Model",
                 barmode="group", orientation="h",
                 title="Feature Importance",
                 color_discrete_sequence=["#4C72B0","#DD8452","#55A868"])
    fig.update_layout(**plotly_theme())
    return fig


def fig_model_compare(results):
    metrics = ["accuracy","precision","recall","f1","auc"]
    labels  = ["Accuracy","Precision","Recall","F1","AUC"]
    data = []
    for r in results:
        for m, l in zip(metrics, labels):
            data.append({"Model":r["name"], "Metric":l, "Score":r[m]})
    df_m = pd.DataFrame(data)
    fig = px.bar(df_m, x="Metric", y="Score", color="Model", barmode="group",
                 title="Model Performance Comparison",
                 color_discrete_sequence=["#4C72B0","#DD8452","#55A868"],
                 text_auto=".2f")
    fig.update_layout(**plotly_theme(), yaxis_range=[0,1.15])
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# APP LAYOUT
# ══════════════════════════════════════════════════════════════════════════════

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Clinical Analysis")
    st.markdown("---")

    # File upload
    st.markdown("### 📂 Dataset")
    uploaded = st.file_uploader("Upload CSV", type="csv",
                                 help="Upload your clinical CSV dataset")

    # Use bundled demo data
    demo_path = Path(__file__).parent.parent / "data" / "competition_dataset.csv"
    if uploaded is None and demo_path.exists():
        with open(demo_path, "rb") as f:
            uploaded = io.BytesIO(f.read())
        st.info("Using demo dataset")

    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    ml_target = st.selectbox("Prediction Target",
                              ["Complication (Yes/No)", "Long Stay (>7 days)"])
    show_raw = st.checkbox("Show raw data", value=False)

    st.markdown("---")
    st.markdown("### 📖 About")
    st.markdown(
        "Clinical Intervention Analysis\n\n"
        "Compares Control vs Intervention groups using descriptive statistics, "
        "statistical tests, and ML models."
    )

# ── Guard ─────────────────────────────────────────────────────────────────────
if uploaded is None:
    st.title("🏥 Clinical Intervention Analysis")
    st.info("👈 Upload a CSV file or use the sidebar to load the demo dataset.")
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────────────
raw, df, clean_report = load_and_preprocess(uploaded)

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.title("🏥 Clinical Intervention Analysis Dashboard")
st.markdown("Interactive analysis of **Control vs Intervention** patient groups")
st.markdown("---")

# KPI cards
k1, k2, k3, k4, k5 = st.columns(5)
ctrl_n = len(df[df["group"]=="Control"])
intv_n = len(df[df["group"]=="Intervention"])
comp_r = df["complication_bin"].mean() * 100
mean_stay = df["hospital_stay"].mean()
k1.metric("Total Patients", clean_report["rows_clean"])
k2.metric("Control Group", ctrl_n)
k3.metric("Intervention Group", intv_n)
k4.metric("Complication Rate", f"{comp_r:.1f}%")
k5.metric("Avg Hospital Stay", f"{mean_stay:.1f} days")

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tabs = st.tabs([
    "📊 Overview",
    "📈 EDA Visualisations",
    "🔬 Statistical Tests",
    "🤖 ML Models",
    "🎯 Predictions",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Overview
# ─────────────────────────────────────────────────────────────────────────────
with tabs[0]:
    st.header("Dataset Overview")

    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Data Quality Report")
        q_data = {
            "Metric": ["Raw Rows", "Cleaned Rows", "Duplicates Removed",
                       "Missing Values", "Features"],
            "Value":  [clean_report["rows_raw"], clean_report["rows_clean"],
                       clean_report["duplicates_removed"],
                       sum(clean_report["missing"].values()),
                       len(df.columns)]
        }
        st.dataframe(pd.DataFrame(q_data), hide_index=True, use_container_width=True)

    with col_right:
        st.subheader("Column Types")
        dtype_df = pd.DataFrame({"Column": df.dtypes.index,
                                  "Type": df.dtypes.astype(str).values})
        st.dataframe(dtype_df, hide_index=True, use_container_width=True)

    st.subheader("Descriptive Statistics")
    num_cols = ["age","hospital_stay"]
    desc = df[num_cols].describe().T
    desc["median"] = df[num_cols].median()
    desc = desc[["count","mean","median","std","min","25%","75%","max"]].round(3)
    st.dataframe(desc, use_container_width=True)

    st.subheader("Group-level Statistics")
    grp_rows = []
    for grp in ["Control","Intervention"]:
        sub = df[df["group"]==grp]
        for col in ["age","hospital_stay"]:
            grp_rows.append({
                "Group": grp, "Variable": col,
                "N": len(sub),
                "Mean": round(sub[col].mean(),2),
                "Median": round(sub[col].median(),2),
                "Std": round(sub[col].std(),2),
                "Min": sub[col].min(), "Max": sub[col].max()
            })
    st.dataframe(pd.DataFrame(grp_rows), hide_index=True, use_container_width=True)

    if show_raw:
        st.subheader("Raw Data")
        st.dataframe(df, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — EDA Visualisations
# ─────────────────────────────────────────────────────────────────────────────
with tabs[1]:
    st.header("Exploratory Data Analysis")

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(fig_hist(df, "age", "Age Distribution by Group"),
                        use_container_width=True)
    with r1c2:
        st.plotly_chart(fig_box(df, "hospital_stay", "Hospital Stay by Group"),
                        use_container_width=True)

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(fig_scatter(df), use_container_width=True)
    with r2c2:
        st.plotly_chart(fig_complication_bar(df), use_container_width=True)

    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.plotly_chart(fig_heatmap(df), use_container_width=True)
    with r3c2:
        st.plotly_chart(fig_age_pie(df), use_container_width=True)

    # Key insights
    ctrl_stay = df[df["group"]=="Control"]["hospital_stay"].mean()
    intv_stay = df[df["group"]=="Intervention"]["hospital_stay"].mean()
    diff = ctrl_stay - intv_stay
    ctrl_comp = df[df["group"]=="Control"]["complication_bin"].mean()*100
    intv_comp = df[df["group"]=="Intervention"]["complication_bin"].mean()*100

    st.markdown("---")
    st.subheader("📌 Key Insights")
    i1, i2, i3 = st.columns(3)
    with i1:
        st.markdown(f"""
        <div class="insight-box">
        <b>🏨 Hospital Stay</b><br>
        Intervention group stays <b>{diff:.1f} days shorter</b> on average
        ({intv_stay:.1f} vs {ctrl_stay:.1f} days)
        </div>""", unsafe_allow_html=True)
    with i2:
        st.markdown(f"""
        <div class="insight-box">
        <b>⚠️ Complications</b><br>
        Control: <b>{ctrl_comp:.1f}%</b> vs Intervention: <b>{intv_comp:.1f}%</b>
        </div>""", unsafe_allow_html=True)
    with i3:
        st.markdown(f"""
        <div class="insight-box">
        <b>👥 Sample Size</b><br>
        Control: <b>{ctrl_n}</b> patients · Intervention: <b>{intv_n}</b> patients
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Statistical Tests
# ─────────────────────────────────────────────────────────────────────────────
with tabs[2]:
    st.header("Statistical Group Comparisons")

    tests_df = stat_tests(df)
    st.subheader("Test Results")
    st.dataframe(tests_df, hide_index=True, use_container_width=True)

    st.markdown("---")
    st.subheader("Significance Interpretation")
    col_a, col_b = st.columns(2)
    with col_a:
        st.info("**Welch t-test** compares means of continuous variables (age, hospital stay) between groups. Assumes normality but does not assume equal variances.")
        t_stay = tests_df[tests_df["Variable"]=="hospital_stay"]["pValue"].values[0]
        if t_stay < 0.05:
            st.success(f"✅ Hospital stay difference is **statistically significant** (p={t_stay:.4f}). The intervention significantly reduces length of stay.")
        else:
            st.warning(f"Hospital stay difference is not significant at p<0.05 (p={t_stay:.4f})")

    with col_b:
        st.info("**Chi-square test** compares distributions of categorical variables (complication, sex) between groups.")
        p_comp = tests_df[tests_df["Variable"]=="complication"]["pValue"].values[0]
        if p_comp < 0.05:
            st.success(f"✅ Complication rate difference is **statistically significant** (p={p_comp:.4f})")
        else:
            st.warning(f"Complication rate difference is not significant at p<0.05 (p={p_comp:.4f})")

    st.markdown("---")
    st.subheader("Distribution Comparison")
    col_1, col_2 = st.columns(2)
    with col_1:
        st.plotly_chart(fig_box(df, "age", "Age: Control vs Intervention"),
                        use_container_width=True)
    with col_2:
        st.plotly_chart(fig_box(df, "hospital_stay", "Hospital Stay: Control vs Intervention"),
                        use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ML Models
# ─────────────────────────────────────────────────────────────────────────────
with tabs[3]:
    st.header("Machine Learning Models")

    is_comp = "Complication" in ml_target
    y = df["complication_bin"].values if is_comp else df["stay_long"].values
    X = df[FEATURE_COLS].values
    target_label = "Complication" if is_comp else "Long Stay (>7 days)"

    with st.spinner("Training models with cross-validation…"):
        results = train_models(X.tolist(), y.tolist(), target_label)

    st.subheader(f"Target: **{target_label}**")

    # Metrics table
    metrics_rows = []
    for r in results:
        metrics_rows.append({
            "Model": r["name"],
            "Accuracy": r["accuracy"],
            "Precision": r["precision"],
            "Recall": r["recall"],
            "F1-Score": r["f1"],
            "ROC-AUC": r["auc"],
        })
    metrics_df = pd.DataFrame(metrics_rows)

    # Highlight best model
    best_model_name = results[0]["name"]
    st.success(f"🏆 Best Model: **{best_model_name}** (AUC = {results[0]['auc']:.4f})")

    def highlight_best(row):
        return ["background-color: #EBF8FF; font-weight: bold"
                if row["Model"] == best_model_name else "" for _ in row]

    st.dataframe(metrics_df.style.apply(highlight_best, axis=1),
                 hide_index=True, use_container_width=True)

    st.markdown("---")
    r_top, r_bot = st.columns(2), st.columns(2)

    with r_top[0]:
        st.plotly_chart(fig_roc(results), use_container_width=True)
    with r_top[1]:
        st.plotly_chart(fig_model_compare(results), use_container_width=True)
    with r_bot[0]:
        st.plotly_chart(fig_feature_importance(results), use_container_width=True)
    with r_bot[1]:
        best_result = next(r for r in results if r["name"] == best_model_name)
        st.plotly_chart(fig_cm(best_result), use_container_width=True)

    # Best params
    st.markdown("---")
    st.subheader("Best Hyperparameters")
    for r in results:
        with st.expander(f"🔧 {r['name']}"):
            params_df = pd.DataFrame({
                "Parameter": list(r["best_params"].keys()),
                "Value": [str(v) for v in r["best_params"].values()]
            })
            st.dataframe(params_df, hide_index=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 5 — Predictions
# ─────────────────────────────────────────────────────────────────────────────
with tabs[4]:
    st.header("🎯 Patient Outcome Prediction")
    st.markdown("Enter patient data below to get a real-time prediction from the best model.")

    # Retrain best model on full data for prediction
    is_comp_pred = "Complication" in ml_target
    y_pred_full = df["complication_bin"].values if is_comp_pred else df["stay_long"].values
    X_pred_full = df[FEATURE_COLS].values

    with st.spinner("Loading model..."):
        pred_results = train_models(X_pred_full.tolist(), y_pred_full.tolist(), "pred")
    best_fitted = pred_results[0]

    # Rebuild pipeline fitted on all data for actual prediction
    from sklearn.linear_model import LogisticRegression as LR
    from sklearn.ensemble import RandomForestClassifier as RFC
    best_name = best_fitted["name"]

    col_inp, col_out = st.columns([1, 1])
    with col_inp:
        st.subheader("Patient Data")
        p_age   = st.slider("Age (years)", 18, 100, 55)
        p_sex   = st.selectbox("Sex", ["Male", "Female"])
        p_group = st.selectbox("Treatment Group", ["Control", "Intervention"])
        p_stay  = st.slider("Hospital Stay (days)", 1, 20, 6)
        predict_btn = st.button("🔍 Predict Outcome", type="primary", use_container_width=True)

    with col_out:
        st.subheader("Prediction Result")
        if predict_btn:
            sex_e   = 1 if p_sex == "Male" else 0
            grp_e   = 1 if p_group == "Intervention" else 0
            patient = pd.DataFrame([[p_age, sex_e, grp_e, p_stay]],
                                   columns=FEATURE_COLS)

            # Use models trained above
            probs = []
            for r in pred_results:
                # Reconstruct fitted model from cache
                pass

            # Quick predict using best result's logic via fresh small pipeline
            pipe_map = {
                "Logistic Regression": Pipeline([
                    ("sc", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, C=1, random_state=RANDOM_STATE))
                ]),
                "Random Forest": Pipeline([
                    ("clf", RFC(n_estimators=100, random_state=RANDOM_STATE))
                ]),
                "Gradient Boosting": Pipeline([
                    ("clf", GradientBoostingClassifier(n_estimators=100, random_state=RANDOM_STATE))
                ]),
            }
            target_col = "complication_bin" if is_comp_pred else "stay_long"
            pipe = pipe_map[best_name]
            pipe.fit(df[FEATURE_COLS], df[target_col])
            prob = pipe.predict_proba(patient)[0][1]
            pred = pipe.predict(patient)[0]

            outcome_label = ("Complication" if is_comp_pred else "Long Stay") if pred == 1 else "No " + ("Complication" if is_comp_pred else "Long Stay")
            color = "🔴" if pred == 1 else "🟢"

            st.markdown(f"""
            <div style="background:{'#FFF5F5' if pred==1 else '#F0FFF4'};
                        border-left: 5px solid {'#E53E3E' if pred==1 else '#38A169'};
                        padding: 20px; border-radius: 0 12px 12px 0; margin-top:10px">
                <h3 style="margin:0; color:{'#E53E3E' if pred==1 else '#38A169'}">
                    {color} {outcome_label}
                </h3>
                <p style="margin:8px 0 0 0; font-size:1.1rem">
                    Probability: <b>{prob*100:.1f}%</b>
                </p>
            </div>""", unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prob * 100,
                number={"suffix": "%", "font":{"size":28}},
                title={"text": f"Risk Score", "font":{"size":16}},
                gauge={
                    "axis": {"range": [0,100]},
                    "bar": {"color": "#E53E3E" if prob > 0.5 else "#38A169"},
                    "steps": [
                        {"range":[0,30],  "color":"#C6F6D5"},
                        {"range":[30,60], "color":"#FEFCBF"},
                        {"range":[60,100],"color":"#FED7D7"},
                    ],
                    "threshold": {"line":{"color":"black","width":3}, "value":50}
                }
            ))
            gauge.update_layout(height=250, **plotly_theme())
            st.plotly_chart(gauge, use_container_width=True)

            st.markdown(f"**Model used:** {best_name}  \n**Best model AUC:** {pred_results[0]['auc']:.4f}")
        else:
            st.info("👈 Enter patient data and click **Predict Outcome**")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#718096; font-size:0.85rem'>"
    "Clinical Intervention Analysis Dashboard · Built with Streamlit · "
    "Models: Logistic Regression, Random Forest, Gradient Boosting"
    "</div>",
    unsafe_allow_html=True
)
