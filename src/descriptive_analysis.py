"""
Clinical Intervention Analysis - Descriptive Analysis Module
=============================================================
Computes summary statistics and generates professional visualisations
saved to /reports/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from pathlib import Path

# ── Palette ──────────────────────────────────────────────────────────────────
CTRL_COLOR  = "#4C72B0"
INTV_COLOR  = "#DD8452"
PALETTE     = {"Control": CTRL_COLOR, "Intervention": INTV_COLOR}
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Descriptive statistics
# ─────────────────────────────────────────────────────────────────────────────

def compute_descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Return a formatted descriptive-statistics table."""
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    stats_df = df[num_cols].describe().T
    stats_df["median"] = df[num_cols].median()
    stats_df["skew"]   = df[num_cols].skew()
    stats_df["kurt"]   = df[num_cols].kurt()
    stats_df = stats_df[["count","mean","median","std","min","25%","75%","max","skew","kurt"]]
    stats_df.columns   = ["N","Mean","Median","Std","Min","Q1","Q3","Max","Skew","Kurt"]
    return stats_df.round(3)


def compute_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats split by Control vs Intervention."""
    num_cols = ["age", "hospital_stay"]
    rows = []
    for grp in ["Control", "Intervention"]:
        sub = df[df["group"] == grp][num_cols]
        for col in num_cols:
            rows.append({
                "Group": grp, "Variable": col,
                "N": len(sub),
                "Mean": round(sub[col].mean(), 2),
                "Median": round(sub[col].median(), 2),
                "Std": round(sub[col].std(), 2),
                "Min": sub[col].min(),
                "Max": sub[col].max(),
            })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Statistical tests
# ─────────────────────────────────────────────────────────────────────────────

def run_statistical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run:
      - Independent t-test for age & hospital_stay (Control vs Intervention)
      - Chi-square test for complication rate by group
      - Chi-square test for sex distribution by group
    """
    results = []

    ctrl = df[df["group"] == "Control"]
    intv = df[df["group"] == "Intervention"]

    for col in ["age", "hospital_stay"]:
        t, p = stats.ttest_ind(ctrl[col], intv[col], equal_var=False)
        results.append({
            "Test": "Welch t-test",
            "Variable": col,
            "Statistic": round(t, 4),
            "p-value": round(p, 4),
            "Significant (p<0.05)": "✓" if p < 0.05 else "✗",
        })

    for cat in ["complication", "sex"]:
        ct = pd.crosstab(df["group"], df[cat])
        chi2, p, dof, _ = stats.chi2_contingency(ct)
        results.append({
            "Test": "Chi-square",
            "Variable": cat,
            "Statistic": round(chi2, 4),
            "p-value": round(p, 4),
            "Significant (p<0.05)": "✓" if p < 0.05 else "✗",
        })

    return pd.DataFrame(results)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path: Path, name: str) -> Path:
    out = path / name
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_age_distribution(df: pd.DataFrame, reports: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Age Distribution by Group", fontsize=15, fontweight="bold")

    for ax, grp, color in zip(axes, ["Control", "Intervention"],
                              [CTRL_COLOR, INTV_COLOR]):
        sub = df[df["group"] == grp]["age"]
        ax.hist(sub, bins=15, color=color, edgecolor="white", alpha=0.85)
        ax.axvline(sub.mean(), color="black", linestyle="--", linewidth=1.5,
                   label=f"Mean = {sub.mean():.1f}")
        ax.set_title(f"{grp} Group (n={len(sub)})")
        ax.set_xlabel("Age (years)")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    return _save(fig, reports, "age_distribution.png")


def plot_hospital_stay_boxplot(df: pd.DataFrame, reports: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8, 6))
    order = ["Control", "Intervention"]
    sns.boxplot(data=df, x="group", y="hospital_stay", order=order,
                palette=PALETTE, width=0.5, ax=ax, flierprops=dict(marker="o", markersize=5))
    sns.stripplot(data=df, x="group", y="hospital_stay", order=order,
                  palette=PALETTE, jitter=True, alpha=0.4, size=4, ax=ax)

    # Annotate means
    for i, grp in enumerate(order):
        m = df[df["group"] == grp]["hospital_stay"].mean()
        ax.text(i, m + 0.3, f"μ={m:.1f}", ha="center", fontsize=10, color="black")

    ax.set_title("Hospital Stay Duration: Control vs Intervention",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Group")
    ax.set_ylabel("Hospital Stay (days)")
    plt.tight_layout()
    return _save(fig, reports, "hospital_stay_boxplot.png")


def plot_complication_rates(df: pd.DataFrame, reports: Path) -> Path:
    ct = df.groupby(["group", "complication"]).size().unstack(fill_value=0)
    pct = ct.div(ct.sum(axis=1), axis=0) * 100

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    fig.suptitle("Complication Rates by Group", fontsize=15, fontweight="bold")

    pct.plot(kind="bar", ax=axes[0], color=["#4CAF50", "#F44336"],
             edgecolor="white", width=0.6)
    axes[0].set_title("Complication Rate (%)")
    axes[0].set_xlabel("Group")
    axes[0].set_ylabel("Percentage (%)")
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(title="Complication")
    for p in axes[0].patches:
        axes[0].annotate(f"{p.get_height():.1f}%",
                         (p.get_x() + p.get_width() / 2, p.get_height() + 0.5),
                         ha="center", fontsize=9)

    ct.plot(kind="bar", ax=axes[1], color=["#4CAF50", "#F44336"],
            edgecolor="white", width=0.6)
    axes[1].set_title("Complication Count")
    axes[1].set_xlabel("Group")
    axes[1].set_ylabel("Count")
    axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=0)
    axes[1].legend(title="Complication")

    plt.tight_layout()
    return _save(fig, reports, "complication_rates.png")


def plot_correlation_heatmap(df: pd.DataFrame, reports: Path) -> Path:
    num_df = df[["age", "hospital_stay", "sex_enc", "group_enc", "complication_bin"]].copy()
    num_df.columns = ["Age", "Hospital Stay", "Sex (Male=1)", "Intervention", "Complication"]

    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix – Clinical Variables",
                 fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    return _save(fig, reports, "correlation_heatmap.png")


def plot_scatter_age_stay(df: pd.DataFrame, reports: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    for grp, color in PALETTE.items():
        sub = df[df["group"] == grp]
        ax.scatter(sub["age"], sub["hospital_stay"], c=color, label=grp,
                   alpha=0.65, s=60, edgecolors="white", linewidths=0.5)
        m, b = np.polyfit(sub["age"], sub["hospital_stay"], 1)
        xs = np.linspace(sub["age"].min(), sub["age"].max(), 100)
        ax.plot(xs, m * xs + b, color=color, linewidth=1.5, linestyle="--")

    ax.set_title("Age vs Hospital Stay (with Trend Lines)",
                 fontsize=14, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Hospital Stay (days)")
    ax.legend(title="Group")
    plt.tight_layout()
    return _save(fig, reports, "scatter_age_stay.png")


def plot_sex_distribution(df: pd.DataFrame, reports: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Sex Distribution by Group", fontsize=15, fontweight="bold")

    for ax, grp, color in zip(axes, ["Control", "Intervention"],
                              [CTRL_COLOR, INTV_COLOR]):
        sub = df[df["group"] == grp]["sex"].value_counts()
        wedge_props = dict(width=0.4, edgecolor="white", linewidth=2)
        ax.pie(sub, labels=sub.index, autopct="%1.1f%%",
               colors=["#5B9BD5", "#ED7D31"], startangle=90,
               wedgeprops=wedge_props)
        ax.set_title(f"{grp} (n={df[df['group']==grp].shape[0]})")

    plt.tight_layout()
    return _save(fig, reports, "sex_distribution.png")


def generate_all_plots(df: pd.DataFrame, reports_dir: str) -> dict:
    """Generate all visualisation files. Returns dict {name: path}."""
    rpath = Path(reports_dir)
    rpath.mkdir(parents=True, exist_ok=True)

    paths = {}
    paths["age_distribution"]       = plot_age_distribution(df, rpath)
    paths["hospital_stay_boxplot"]  = plot_hospital_stay_boxplot(df, rpath)
    paths["complication_rates"]     = plot_complication_rates(df, rpath)
    paths["correlation_heatmap"]    = plot_correlation_heatmap(df, rpath)
    paths["scatter_age_stay"]       = plot_scatter_age_stay(df, rpath)
    paths["sex_distribution"]       = plot_sex_distribution(df, rpath)
    print(f"✔ Saved {len(paths)} plots to {rpath}")
    return {k: str(v) for k, v in paths.items()}


if __name__ == "__main__":
    from data_preprocessing import full_pipeline
    base = Path(__file__).resolve().parent.parent
    _, _, df, _, _, _, _ = full_pipeline(str(base / "data" / "competition_dataset.csv"))
    stats_tbl = compute_descriptive_stats(df)
    print(stats_tbl)
    tests_tbl = run_statistical_tests(df)
    print(tests_tbl)
    generate_all_plots(df, str(base / "reports"))
