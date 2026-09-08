"""
Generates a synthetic mobile-money transaction dataset for the NairaGuard project.

Why synthetic, and why this schema:
- We don't download PaySim from Kaggle. Instead we generate our own dataset that follows
  PaySim's well-known column layout (step, type, amount, nameOrig, oldbalanceOrg,
  newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud), so
  results stay comparable to the literature we're citing (Lopez-Rojas et al.; Lokanan, 2023).
- Fraud is injected using three typologies pulled directly from the Nigerian/African fintech
  literature (see README.md), rather than generic patterns, so feature importance results
  later in the pipeline map cleanly back to the literature review.

Run this script directly to regenerate the dataset:
    venv/Scripts/python.exe src/generate_data.py
"""

import json
import random

import numpy as np
import pandas as pd

# --- Reproducibility -----------------------------------------------------
# Fixing the seed means re-running this script always produces the exact same
# dataset. That matters for the paper: our results need to be reproducible.
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Simulation size -------------------------------------------------------
NUM_CUSTOMERS = 5_000
NUM_MERCHANTS = 1_000
NUM_STEPS = 720  # ~30 days, at roughly one step per hour (mirrors PaySim's convention)
TARGET_ROWS = 300_000

# Roughly matches real-world mobile-money transaction type proportions.
TRANSACTION_TYPES = ["CASH_OUT", "PAYMENT", "CASH_IN", "TRANSFER", "DEBIT"]
TYPE_WEIGHTS = [0.35, 0.34, 0.22, 0.08, 0.01]

# The naive "existing system" rule this project's ML models are meant to improve on:
# flag any single transfer over this amount. It's crude on purpose — real fraud rarely
# shows up as one suspiciously large transfer, which is exactly the point we're making.
FLAG_THRESHOLD_NAIRA = 150_000

customer_ids = [f"C{100000000 + i}" for i in range(NUM_CUSTOMERS)]
merchant_ids = [f"M{100000000 + i}" for i in range(NUM_MERCHANTS)]

# Each customer's current wallet balance, updated as transactions happen.
balances = {cid: float(np.random.lognormal(mean=9.5, sigma=1.0)) for cid in customer_ids}

# Tracks which destination accounts have ever received a transfer before — used both by
# the agent/cash-out fraud injection below and, later, as a feature during modeling.
dest_ever_received = set()

# Tracks how many transactions each customer has made so far — used to make sure
# account-takeover fraud only targets accounts with an established history.
customer_tx_count = {cid: 0 for cid in customer_ids}

rows = []


def sample_amount(tx_type):
    """Draw a realistic Naira amount for a given transaction type (lognormal: many
    small transactions, a long tail of larger ones).

    CASH_IN is deliberately sized larger on average than the outflow types (CASH_OUT,
    PAYMENT, TRANSFER, DEBIT combined). Those outflow types happen far more often
    (~78% of transactions vs. 22% for CASH_IN), so if CASH_IN amounts weren't sized up
    to compensate, every account's balance would drain toward zero well before the
    simulation ends — which would make most "amounts" degenerate to 0 (clipped by the
    can't-spend-more-than-you-have rule) instead of reflecting realistic account activity.
    """
    if tx_type == "DEBIT":
        return float(np.random.lognormal(mean=7.5, sigma=0.6))
    if tx_type == "PAYMENT":
        return float(np.random.lognormal(mean=8.5, sigma=0.9))
    if tx_type == "CASH_IN":
        return float(np.random.lognormal(mean=10.2, sigma=0.8))
    return float(np.random.lognormal(mean=9.0, sigma=1.1))  # CASH_OUT, TRANSFER


def counterparty_for(tx_type):
    """Decide who the other side of the transaction is, per type."""
    if tx_type == "TRANSFER":
        return random.choice(customer_ids), True  # another customer, balance tracked
    return random.choice(merchant_ids), False  # merchant/agent, balance not tracked


def record_transaction(step, tx_type, orig, amount, dest, dest_balance_tracked, is_fraud):
    """Apply a transaction's effect on balances and append the resulting row."""
    old_orig = balances[orig]

    if tx_type == "CASH_IN":
        new_orig = old_orig + amount
    else:
        # CASH_OUT, PAYMENT, DEBIT, TRANSFER all move money out of the customer's wallet.
        amount = min(amount, old_orig)  # can't send more than the balance holds
        new_orig = old_orig - amount

    balances[orig] = new_orig
    customer_tx_count[orig] += 1

    if dest_balance_tracked:
        old_dest = balances[dest]
        new_dest = old_dest + amount
        balances[dest] = new_dest
    else:
        old_dest, new_dest = 0.0, 0.0

    dest_ever_received.add(dest)

    is_flagged = 1 if (tx_type == "TRANSFER" and amount > FLAG_THRESHOLD_NAIRA) else 0

    rows.append(
        {
            "step": step,
            "type": tx_type,
            "amount": round(amount, 2),
            "nameOrig": orig,
            "oldbalanceOrg": round(old_orig, 2),
            "newbalanceOrig": round(new_orig, 2),
            "nameDest": dest,
            "oldbalanceDest": round(old_dest, 2),
            "newbalanceDest": round(new_dest, 2),
            "isFraud": is_fraud,
            "isFlaggedFraud": is_flagged,
        }
    )


LOW_BALANCE_FLOOR = 1_000  # below this, an account tops up instead of spending further


def generate_normal_transaction(step):
    orig = random.choice(customer_ids)
    if balances[orig] < LOW_BALANCE_FLOOR:
        # A near-empty wallet gets topped up rather than attempting to spend from
        # nothing — this is both more realistic and avoids the dataset filling up
        # with degenerate zero-amount rows that carry no real signal.
        tx_type = "CASH_IN"
    else:
        tx_type = random.choices(TRANSACTION_TYPES, weights=TYPE_WEIGHTS, k=1)[0]
    dest, dest_tracked = counterparty_for(tx_type)
    while dest == orig:
        dest, dest_tracked = counterparty_for(tx_type)
    amount = sample_amount(tx_type)
    record_transaction(step, tx_type, orig, amount, dest, dest_tracked, is_fraud=0)


# --- Fraud typology 1: account-takeover / SIM-swap style ------------------
# A sudden large transfer (or two) that drains an established account to near-zero —
# atypical for that account's usual pattern. Modeled on the AJESA Nigeria GSM-fintech paper.
def inject_account_takeover(step):
    eligible = [c for c in customer_ids if customer_tx_count[c] >= 3 and balances[c] > 5_000]
    if not eligible:
        return
    victim = random.choice(eligible)
    remaining_steps = min(2, NUM_STEPS - step)
    num_drains = random.choice([1, 2])
    for i in range(num_drains):
        if balances[victim] < 1_000:
            break
        drain_fraction = random.uniform(0.85, 0.99)
        amount = balances[victim] * drain_fraction
        dest = random.choice(customer_ids)
        while dest == victim:
            dest = random.choice(customer_ids)
        record_transaction(
            step + min(i, remaining_steps), "TRANSFER", victim, amount, dest,
            dest_balance_tracked=True, is_fraud=1,
        )


# --- Fraud typology 2: velocity fraud --------------------------------------
# An abnormal burst of transactions in a very short time window. Modeled on the Rwanda
# mobile-money AML paper's "rolling velocity" feature.
def inject_velocity_fraud(step):
    victim = random.choice(customer_ids)
    burst_size = random.randint(12, 20)
    for _ in range(burst_size):
        if balances[victim] < 500:
            break
        amount = min(balances[victim] * random.uniform(0.05, 0.15), balances[victim])
        tx_type = random.choice(["TRANSFER", "CASH_OUT"])
        dest, dest_tracked = counterparty_for(tx_type)
        while dest == victim:
            dest, dest_tracked = counterparty_for(tx_type)
        record_transaction(step, tx_type, victim, amount, dest, dest_tracked, is_fraud=1)


# --- Fraud typology 3: agent / cash-out fraud ------------------------------
# A disproportionately large CASH_OUT sent to a destination with no prior transaction
# history. Echoes the mobile-money agent-fraud literature.
def inject_agent_cashout_fraud(step):
    eligible = [c for c in customer_ids if balances[c] > 20_000]
    if not eligible:
        return
    victim = random.choice(eligible)
    amount = balances[victim] * random.uniform(0.7, 0.95)
    novel_dest = f"M{900000000 + random.randint(0, 99999)}"  # unlikely to have been used
    record_transaction(step, "CASH_OUT", victim, amount, novel_dest, dest_balance_tracked=False, is_fraud=1)


# --- Run the simulation -----------------------------------------------------
avg_per_step = TARGET_ROWS // NUM_STEPS

# Spread fraud events across the timeline rather than clustering them at the end.
takeover_steps = np.random.choice(NUM_STEPS, size=150, replace=True)
velocity_steps = np.random.choice(NUM_STEPS, size=60, replace=True)
cashout_steps = np.random.choice(NUM_STEPS, size=550, replace=True)

for step in range(1, NUM_STEPS + 1):
    n = np.random.poisson(avg_per_step)
    for _ in range(n):
        generate_normal_transaction(step)

    for s in takeover_steps[takeover_steps == step]:
        inject_account_takeover(step)
    for s in velocity_steps[velocity_steps == step]:
        inject_velocity_fraud(step)
    for s in cashout_steps[cashout_steps == step]:
        inject_agent_cashout_fraud(step)

df = pd.DataFrame(rows)
df = df.sort_values("step", kind="stable").reset_index(drop=True)

# --- Save ------------------------------------------------------------------
df.to_csv("data/paysim_synthetic.csv", index=False)
df.to_json("data/paysim_synthetic.json", orient="records")

# --- Quick summary so we can eyeball the result before moving on -----------
n_rows = len(df)
n_fraud = int(df["isFraud"].sum())
print(f"Generated {n_rows:,} transactions")
print(f"Fraud cases: {n_fraud:,} ({n_fraud / n_rows:.3%})")
print(f"Naive-rule flagged (isFlaggedFraud): {int(df['isFlaggedFraud'].sum()):,}")
print("\nTransaction type breakdown:")
print(df["type"].value_counts())
print("\nSaved to data/paysim_synthetic.csv and data/paysim_synthetic.json")
