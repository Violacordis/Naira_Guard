"""
Steps 4-6 of the pipeline: address class imbalance with SMOTE, train Random Forest and
XGBoost classifiers, and visualize how they compare.

Why SMOTE only on the training split: fitting it on the full dataset before the
train/test split would let synthetic minority-class points leak information about test
transactions into training, inflating evaluation scores. SMOTE is fit and applied to the
training split only, after the split; the test split stays untouched and reflects the
real ~0.55% fraud rate from src/explore_data.py.

Why accuracy isn't the headline metric: see results/findings.md, Step 2 - with fraud at
0.55% of transactions, a model that predicts "not fraud" for everything scores >99%
accuracy while catching no fraud. Precision, recall, F1, and PR-AUC are reported instead.

Run with:
    venv/Scripts/python.exe src/train_models.py
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

SEED = 42

# Colors: consistent with the rest of the project (blue = legitimate/RF-ish, red = fraud),
# extended here with a third color to keep XGBoost visually distinct from the fraud-red
# used for confusion-matrix and class charts elsewhere.
COLOR_RF = "#2a78d6"
COLOR_XGB = "#e0a02a"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"

df = pd.read_csv("data/paysim_features.csv")

# --- Prepare features -------------------------------------------------------------
# Dropped: nameOrig/nameDest (identifiers, not predictive on their own), step (a time
# index rather than a behavioral signal), isFlaggedFraud (the naive baseline rule's own
# output - keeping it as a feature would let models lean on the rule this project is
# meant to improve on, rather than learning the underlying behavioral patterns).
feature_cols = [
    "type",
    "amount",
    "oldbalanceOrg",
    "newbalanceOrig",
    "oldbalanceDest",
    "newbalanceDest",
    "tx_velocity",
    "balance_drain_ratio",
    "dest_is_novel",
    "orig_prior_tx_count",
]
X = pd.get_dummies(df[feature_cols], columns=["type"])
y = df["isFraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

# --- Address class imbalance with SMOTE (training split only) ---------------------
smote = SMOTE(random_state=SEED)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(
    f"Training set before SMOTE: {len(y_train):,} rows ({y_train.mean():.3%} fraud)\n"
    f"Training set after SMOTE:  {len(y_train_res):,} rows ({y_train_res.mean():.3%} fraud)"
)

# --- Train models -------------------------------------------------------------------
models = {
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=-1),
    "XGBoost": XGBClassifier(random_state=SEED, eval_metric="logloss", n_jobs=-1),
}

results = {}
for name, model in models.items():
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        "model": model,
        "y_pred": y_pred,
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "pr_auc": average_precision_score(y_test, y_proba),
    }

# --- Report metrics -------------------------------------------------------------------
metrics_df = pd.DataFrame(
    {name: {k: v for k, v in r.items() if k not in ("model", "y_pred")} for name, r in results.items()}
).T
print("\nTest-set metrics (evaluated on the real ~0.55% fraud rate, no SMOTE applied):")
print(metrics_df.round(4))

# --- Visualize: confusion matrices ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor("#fcfcfb")
for ax, (name, r) in zip(axes, results.items()):
    cm = confusion_matrix(y_test, r["y_pred"])
    sns.heatmap(
        cm, annot=True, fmt=",", cmap="Blues", cbar=False, ax=ax,
        xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"],
    )
    ax.set_title(name, color=COLOR_INK, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted", color=COLOR_MUTED)
    ax.set_ylabel("Actual", color=COLOR_MUTED)
fig.tight_layout()
fig.savefig("results/confusion_matrices.png", dpi=200, facecolor=fig.get_facecolor())
print("\nSaved chart to results/confusion_matrices.png")

# --- Visualize: model comparison -----------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor("#fcfcfb")
metric_names = ["precision", "recall", "f1", "pr_auc"]
x = range(len(metric_names))
width = 0.35
for i, (name, color) in enumerate(zip(results, [COLOR_RF, COLOR_XGB])):
    values = [results[name][m] for m in metric_names]
    offset = (i - 0.5) * width
    bars = ax.bar([p + offset for p in x], values, width, label=name, color=color)
    for bar, val in zip(bars, values):
        ax.annotate(
            f"{val:.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center", va="bottom", fontsize=9, color=COLOR_INK,
        )
ax.set_xticks(list(x))
ax.set_xticklabels(["Precision", "Recall", "F1", "PR-AUC"])
ax.set_title("Model comparison", color=COLOR_INK, fontsize=13, fontweight="bold")
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(COLOR_GRID)
ax.tick_params(colors=COLOR_MUTED)
ax.legend(frameon=False)
ax.margins(y=0.15)
fig.tight_layout()
fig.savefig("results/model_comparison.png", dpi=200, facecolor=fig.get_facecolor())
print("Saved chart to results/model_comparison.png")

# --- Visualize: feature importance ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.patch.set_facecolor("#fcfcfb")
for ax, (name, color) in zip(axes, [("Random Forest", COLOR_RF), ("XGBoost", COLOR_XGB)]):
    importances = pd.Series(results[name]["model"].feature_importances_, index=X.columns)
    top = importances.sort_values(ascending=True).tail(10)
    ax.barh(top.index, top.values, color=color)
    ax.set_title(name, color=COLOR_INK, fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_MUTED)
fig.suptitle("Top 10 feature importances", color=COLOR_INK, fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig("results/feature_importance.png", dpi=200, facecolor=fig.get_facecolor())
print("Saved chart to results/feature_importance.png")
