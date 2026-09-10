"""
Step 3 of the pipeline: engineer behavioral features that mirror the fraud injection
logic directly, so feature-importance results later map cleanly back to the three
typologies from the literature (see src/generate_data.py).

Run with:
    venv/Scripts/python.exe src/build_features.py
"""

import pandas as pd

df = pd.read_csv("data/paysim_synthetic.csv")
df = df.sort_values("step", kind="stable").reset_index(drop=True)

# --- Velocity: transactions by the same origin account within the current step ----
# Velocity fraud (see inject_velocity_fraud) fires 12-20 transactions from one account
# within a single step, so counting same-origin transactions per step captures that
# burst pattern directly.
df["tx_velocity"] = df.groupby(["nameOrig", "step"])["nameOrig"].transform("count")

# --- Balance-drain ratio: how much of the origin's balance this transaction consumes ---
# Account-takeover fraud (see inject_account_takeover) drains 85-99% of the victim's
# balance in one or two transfers. +1 avoids division by zero for empty accounts.
# CASH_IN moves money into the account rather than out of it, so "drain" doesn't apply
# there; left at the amount/oldbalanceOrg ratio it would be unbounded (a small balance
# topped up by a large deposit) and would swamp this feature's signal, so it's fixed at
# 0 for CASH_IN rows.
df["balance_drain_ratio"] = df["amount"] / (df["oldbalanceOrg"] + 1)
df.loc[df["type"] == "CASH_IN", "balance_drain_ratio"] = 0.0

# --- Destination novelty: has this destination ever received a payment before? -------
# Agent/cash-out fraud (see inject_agent_cashout_fraud) always targets a destination
# account with no prior transaction history. `duplicated` over the whole (chronologically
# sorted) dataset is False only on a destination's first-ever appearance, so negating it
# gives "never seen before this row".
df["dest_is_novel"] = (~df.duplicated(subset="nameDest", keep="first")).astype(int)

# --- Prior transaction count: how established is this origin account? -----------------
# inject_account_takeover only targets accounts with >=3 prior transactions, so this
# feature lets a model learn that distinction rather than treating all accounts alike.
df["orig_prior_tx_count"] = df.groupby("nameOrig").cumcount()

df.to_csv("data/paysim_features.csv", index=False)

# --- Quick sanity check: do these features actually separate fraud from legit? --------
print(f"Engineered features for {len(df):,} rows")
print("\nMean feature value, fraud vs. legitimate:")
compare_cols = ["tx_velocity", "balance_drain_ratio", "dest_is_novel", "orig_prior_tx_count"]
print(df.groupby("isFraud")[compare_cols].mean().rename(index={0: "legit", 1: "fraud"}))

print("\nSaved to data/paysim_features.csv")
