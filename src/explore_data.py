"""
Step 2 of the pipeline: load the generated dataset and explore it.

This is the "Findings" screenshot showing the real-world class imbalance problem the
literature discusses: fraud is a tiny fraction of all transactions, which is exactly
why plain accuracy is a misleading metric later (a model that predicts "not fraud" for
everything would score >99% accuracy while catching zero fraud).

Run with:
    venv/Scripts/python.exe src/explore_data.py
"""

import matplotlib.pyplot as plt
import pandas as pd

# Colors: blue for "legitimate", red for "fraud" - consistent with how the confusion
# matrix and comparison charts later in the pipeline use these same two colors.
COLOR_LEGIT = "#2a78d6"
COLOR_FRAUD = "#e34948"
COLOR_INK = "#0b0b0b"
COLOR_MUTED = "#898781"
COLOR_GRID = "#e1e0d9"

df = pd.read_csv("data/paysim_synthetic.csv")

# --- Basic shape -------------------------------------------------------------
print(f"Dataset shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}\n")

# --- Class imbalance -----------------------------------------------------------
fraud_counts = df["isFraud"].value_counts().sort_index()
n_legit, n_fraud = int(fraud_counts[0]), int(fraud_counts[1])
total = n_legit + n_fraud
print("Class imbalance:")
print(f"  Legitimate: {n_legit:,} ({n_legit / total:.3%})")
print(f"  Fraud:      {n_fraud:,} ({n_fraud / total:.3%})")
print(
    f"  -> A model that always predicts 'not fraud' would score "
    f"{n_legit / total:.2%} accuracy while catching 0% of fraud."
)

# --- Transaction type breakdown, and fraud rate within each type ----------------
type_counts = df["type"].value_counts()
fraud_rate_by_type = df.groupby("type")["isFraud"].mean().sort_values(ascending=False) * 100
print("\nTransaction type breakdown:")
print(type_counts)
print("\nFraud rate by transaction type (%):")
print(fraud_rate_by_type.round(3))

# --- How the naive existing-system rule stacks up -------------------------------
# isFlaggedFraud is a crude "flag any transfer over a fixed amount" rule, meant to
# stand in for a legacy threshold-based fraud check. Comparing it to true fraud shows
# exactly why a smarter model is worth building.
flagged = df[df["isFlaggedFraud"] == 1]
recall_naive = flagged["isFraud"].sum() / n_fraud
precision_naive = (flagged["isFraud"] == 1).mean() if len(flagged) else 0
print(
    f"\nNaive rule-based flag (transfer > threshold): "
    f"caught {int(flagged['isFraud'].sum())}/{n_fraud} fraud cases "
    f"(recall {recall_naive:.1%}), precision {precision_naive:.1%}"
)

# --- Visualize: class imbalance + fraud rate by type ----------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.patch.set_facecolor("#fcfcfb")

# Left panel: legitimate vs fraud counts, linear scale on purpose - the fraud bar
# being nearly invisible next to the legitimate bar *is* the finding.
bars = ax1.bar(
    ["Legitimate", "Fraud"], [n_legit, n_fraud], color=[COLOR_LEGIT, COLOR_FRAUD], width=0.6
)
ax1.set_title("Class imbalance", color=COLOR_INK, fontsize=13, fontweight="bold")
ax1.set_ylabel("Number of transactions", color=COLOR_MUTED)
ax1.spines[["top", "right"]].set_visible(False)
ax1.spines[["left", "bottom"]].set_color(COLOR_GRID)
ax1.tick_params(colors=COLOR_MUTED)
for bar, count in zip(bars, [n_legit, n_fraud]):
    ax1.annotate(
        f"{count:,}\n({count / total:.2%})",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center", va="bottom", fontsize=10, color=COLOR_INK,
    )
ax1.margins(y=0.15)

# Right panel: fraud rate within each transaction type.
bars2 = ax2.bar(fraud_rate_by_type.index, fraud_rate_by_type.values, color=COLOR_FRAUD, width=0.6)
ax2.set_title("Fraud rate by transaction type", color=COLOR_INK, fontsize=13, fontweight="bold")
ax2.set_ylabel("Fraud rate (%)", color=COLOR_MUTED)
ax2.spines[["top", "right"]].set_visible(False)
ax2.spines[["left", "bottom"]].set_color(COLOR_GRID)
ax2.tick_params(colors=COLOR_MUTED)
for bar, rate in zip(bars2, fraud_rate_by_type.values):
    ax2.annotate(
        f"{rate:.2f}%",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center", va="bottom", fontsize=9, color=COLOR_INK,
    )
ax2.margins(y=0.15)

fig.tight_layout()
fig.savefig("results/class_imbalance.png", dpi=200, facecolor=fig.get_facecolor())
print("\nSaved chart to results/class_imbalance.png")
