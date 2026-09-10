# Findings log

Running notes from each pipeline stage, written to be dropped almost directly into the
paper's Findings/Analysis chapter. Update this file as each stage completes.

## Step 2: Data exploration (class imbalance)

Chart: `results/class_imbalance.png`

- Dataset: 300,789 simulated transactions, of which 1,648 (0.55%) are fraudulent.
- This confirms the class imbalance problem the literature describes: fraud is a small
  minority of transactions, so a model has very few positive examples to learn from.
- **Why accuracy is a misleading metric here:** a model that predicts "not fraud" for
  every single transaction would score 99.45% accuracy while catching 0% of fraud. This
  is the reason the evaluation stage of this project reports precision, recall, and
  PR-AUC rather than leaning on accuracy alone.
- **Fraud rate by transaction type:** fraud appears only in TRANSFER (2.76%) and
  CASH_OUT (0.99%); CASH_IN, DEBIT, and PAYMENT show 0%. This follows directly from how
  the three fraud typologies were modeled (account-takeover and velocity fraud both move
  money out via transfers; agent/cash-out fraud is a cash-out by definition), and is
  itself consistent with the literature's observation that fraud concentrates in
  money-moving-out transaction types rather than routine payments.
- **Naive baseline comparison:** a crude threshold rule ("flag any transfer over
  ₦150,000") catches only 41 of 1,648 fraud cases (2.5% recall), though it is correct
  54.7% of the time it does fire (precision). This motivates the move to ML models: a
  simple rule is precise but nearly blind to the bulk of fraud.

## Step 3: Feature engineering

Four behavioral features were engineered to mirror the three fraud typologies directly
(`src/build_features.py`), each targeting a specific injection pattern:

| feature | what it captures | legit mean | fraud mean |
|---|---|---|---|
| `tx_velocity` | transactions by the same account within one step | 1.08 | 9.10 |
| `balance_drain_ratio` | share of the account's balance a single outflow consumes (0 for CASH_IN, where "drain" doesn't apply) | 0.15 | 0.45 |
| `dest_is_novel` | destination account has never received a payment before | 1.9% | 42.2% |
| `orig_prior_tx_count` | how established the origin account is | 30.1 | 33.4 |

- `tx_velocity` and `dest_is_novel` separate fraud most sharply, directly reflecting the
  velocity-fraud and agent/cash-out-fraud typologies they were built to capture.
- `orig_prior_tx_count` barely differs between classes on its own; account-takeover fraud
  requires an established account (≥3 prior transactions) but doesn't push the count
  much higher than typical, so this feature is expected to matter more in combination
  with others (e.g. alongside a large `balance_drain_ratio`) than alone.
- Early version of `balance_drain_ratio` computed `amount / oldbalanceOrg` for every
  transaction type, which pulled the legit-class mean up to ~1,671 — an artifact of
  CASH_IN deposits (unbounded relative to the pre-deposit balance) rather than a real
  signal. Zeroing it out for CASH_IN rows fixed this.
