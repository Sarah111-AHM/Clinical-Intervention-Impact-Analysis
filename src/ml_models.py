"""
Clinical Intervention Analysis - Machine Learning Models Module
===============================================================
Trains, evaluates, and explains three classifiers:
  • Logistic Regression
  • Random Forest
  • Gradient Boosting (sklearn GradientBoostingClassifier, XGBoost-compatible API)

Each model is trained with GridSearchCV cross-validation.
Evaluation: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
Feature importance visualisations saved to /reports/.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CV_FOLDS     = 5
TEST_SIZE    = 0.25

FEATURE_NAMES = ["age", "sex_enc", "group_enc", "hospital_stay"]
FEATURE_LABELS = ["Age", "Sex (Male=1)", "Intervention", "Hospital Stay"]


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions + hyperparameter grids
# ─────────────────────────────────────────────────────────────────────────────

def build_pipelines() -> dict:
    """Return dict of {name: (pipeline, param_grid)}."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    models = {
        "Logistic Regression": (
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
            ]),
            {
                "clf__C": [0.01, 0.1, 1, 10, 100],
                "clf__solver": ["lbfgs", "liblinear"],
            }
        ),
        "Random Forest": (
            Pipeline([
                ("clf", RandomForestClassifier(random_state=RANDOM_STATE))
            ]),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__max_depth": [None, 3, 5, 7],
                "clf__min_samples_split": [2, 5],
            }
        ),
        "Gradient Boosting": (
            Pipeline([
                ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE))
            ]),
            {
                "clf__n_estimators": [50, 100, 200],
                "clf__learning_rate": [0.05, 0.1, 0.2],
                "clf__max_depth": [2, 3, 4],
            }
        ),
    }
    return models, cv


# ─────────────────────────────────────────────────────────────────────────────
# Training & evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(name: str, best_model, X_test, y_test) -> dict:
    """Compute all evaluation metrics for a fitted model."""
    y_pred  = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, "predict_proba") else None

    metrics = {
        "Model":     name,
        "Accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1-Score":  round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else None,
        "Confusion Matrix": confusion_matrix(y_test, y_pred).tolist(),
        "Classification Report": classification_report(y_test, y_pred),
    }
    return metrics


def train_all_models(X, y, target_name: str = "complication") -> dict:
    """
    Split data, run GridSearchCV for all models, return results dict.

    Returns:
        {
          "X_train", "X_test", "y_train", "y_test",
          "results": [{model, metrics, best_params, cv_scores}, ...],
          "best_model_name": str
        }
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\n[{target_name}] Train={len(X_train)}, Test={len(X_test)}, "
          f"Positive rate={y.mean():.2%}")

    model_defs, cv = build_pipelines()
    results = []

    for name, (pipeline, param_grid) in model_defs.items():
        gs = GridSearchCV(pipeline, param_grid, cv=cv, scoring="roc_auc",
                          n_jobs=-1, refit=True, verbose=0)
        gs.fit(X_train, y_train)

        cv_scores = cross_val_score(gs.best_estimator_, X_train, y_train,
                                    cv=cv, scoring="roc_auc")
        metrics = evaluate_model(name, gs.best_estimator_, X_test, y_test)

        results.append({
            "name":         name,
            "fitted_model": gs.best_estimator_,
            "best_params":  gs.best_params_,
            "cv_mean":      round(cv_scores.mean(), 4),
            "cv_std":       round(cv_scores.std(), 4),
            "metrics":      metrics,
        })
        print(f"  ✔ {name:<25} AUC={metrics['ROC-AUC']:.4f}  "
              f"(CV {cv_scores.mean():.4f}±{cv_scores.std():.4f})")

    # Rank by ROC-AUC
    results.sort(key=lambda r: r["metrics"]["ROC-AUC"] or 0, reverse=True)
    best_name = results[0]["name"]

    return {
        "target":         target_name,
        "X_train":        X_train,
        "X_test":         X_test,
        "y_train":        y_train,
        "y_test":         y_test,
        "results":        results,
        "best_model_name": best_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature importance
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_importance(fitted_pipeline, model_name: str) -> pd.DataFrame:
    """Extract feature importances from a fitted sklearn Pipeline."""
    clf = fitted_pipeline.named_steps["clf"]

    if hasattr(clf, "feature_importances_"):
        imp = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        imp = np.abs(clf.coef_[0])
    else:
        imp = np.ones(len(FEATURE_NAMES))

    df = pd.DataFrame({
        "Feature":    FEATURE_LABELS,
        "Importance": imp,
        "Model":      model_name,
    }).sort_values("Importance", ascending=False)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Visualisations
# ─────────────────────────────────────────────────────────────────────────────

def plot_roc_curves(output: dict, reports_dir: str) -> str:
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for res, color in zip(output["results"], colors):
        model = res["fitted_model"]
        X_test = output["X_test"]
        y_test = output["y_test"]
        if hasattr(model, "predict_proba"):
            RocCurveDisplay.from_estimator(
                model, X_test, y_test, ax=ax,
                name=f"{res['name']} (AUC={res['metrics']['ROC-AUC']:.3f})",
                color=color, lw=2
            )

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.500)")
    ax.set_title(f"ROC Curves — {output['target'].replace('_',' ').title()} Prediction",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = Path(reports_dir) / f"roc_curves_{output['target']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_confusion_matrices(output: dict, reports_dir: str) -> str:
    n = len(output["results"])
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, output["results"]):
        cm = np.array(res["metrics"]["Confusion Matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                    cbar=False, linewidths=0.5)
        ax.set_title(res["name"], fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    fig.suptitle(f"Confusion Matrices — {output['target'].replace('_',' ').title()}",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    path = Path(reports_dir) / f"confusion_matrices_{output['target']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_feature_importance(output: dict, reports_dir: str) -> str:
    """Stacked bar chart of feature importances across all models."""
    frames = [
        extract_feature_importance(r["fitted_model"], r["name"])
        for r in output["results"]
    ]
    all_imp = pd.concat(frames, ignore_index=True)

    pivot = all_imp.pivot(index="Feature", columns="Model", values="Importance").fillna(0)

    fig, ax = plt.subplots(figsize=(9, 5))
    pivot.plot(kind="barh", ax=ax, colormap="tab10", width=0.7, edgecolor="white")
    ax.set_title(f"Feature Importance — {output['target'].replace('_',' ').title()}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Importance Score")
    ax.set_ylabel("")
    ax.legend(title="Model", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = Path(reports_dir) / f"feature_importance_{output['target']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_model_comparison(output: dict, reports_dir: str) -> str:
    """Grouped bar chart comparing metrics across models."""
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    names   = [r["name"] for r in output["results"]]
    data    = {m: [r["metrics"][m] for r in output["results"]] for m in metrics}

    x  = np.arange(len(metrics))
    w  = 0.22
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (name, color) in enumerate(zip(names, colors)):
        vals = [data[m][i] for m in metrics]
        bars = ax.bar(x + i * w, vals, w, label=name, color=color, edgecolor="white")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{bar.get_height():.2f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x + w)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.15)
    ax.set_title(f"Model Performance Comparison — {output['target'].replace('_',' ').title()}",
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.legend(title="Model")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = Path(reports_dir) / f"model_comparison_{output['target']}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run_full_ml_pipeline(X, y_comp, y_stay, reports_dir: str) -> dict:
    """Train models for both targets and save all plots."""
    Path(reports_dir).mkdir(parents=True, exist_ok=True)
    plot_paths = {}

    for target_name, y in [("complication", y_comp), ("long_stay", y_stay)]:
        print(f"\n{'='*60}")
        print(f" Target: {target_name}")
        print(f"{'='*60}")
        output = train_all_models(X, y, target_name)
        plot_paths[f"roc_{target_name}"]         = plot_roc_curves(output, reports_dir)
        plot_paths[f"cm_{target_name}"]          = plot_confusion_matrices(output, reports_dir)
        plot_paths[f"fi_{target_name}"]          = plot_feature_importance(output, reports_dir)
        plot_paths[f"compare_{target_name}"]     = plot_model_comparison(output, reports_dir)

        # Print summary table
        print(f"\n  {'Model':<25} {'AUC':>7} {'F1':>7} {'Acc':>7}")
        for r in output["results"]:
            m = r["metrics"]
            print(f"  {r['name']:<25} {m['ROC-AUC']:>7.4f} {m['F1-Score']:>7.4f} {m['Accuracy']:>7.4f}")

    print(f"\n✔ Saved {len(plot_paths)} ML plots to {reports_dir}")
    return plot_paths


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_preprocessing import full_pipeline
    base = Path(__file__).resolve().parent.parent
    _, _, _, X, y_comp, y_stay, _ = full_pipeline(str(base / "data" / "competition_dataset.csv"))
    run_full_ml_pipeline(X, y_comp, y_stay, str(base / "reports"))
