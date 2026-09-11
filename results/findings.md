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

## Steps 4-6: SMOTE, model training, and evaluation

Chart: `results/confusion_matrices.png`, `results/model_comparison.png`,
`results/feature_importance.png`

SMOTE was fit on the training split only (240,631 rows, 0.548% fraud), oversampling it
to 478,626 rows at 50/50 before training; the test split (60,158 rows) was left at the
real ~0.55% fraud rate so evaluation reflects real-world conditions.

| metric | Random Forest | XGBoost |
|---|---|---|
| Precision | 0.858 | 0.840 |
| Recall | 0.970 | 0.970 |
| F1 | 0.910 | 0.900 |
| PR-AUC | 0.958 | 0.985 |

- **Both models comfortably beat the naive baseline** from Step 2 (2.5% recall, 54.7%
  precision): both catch ~97% of fraud, correctly, roughly 84-86% of the time they flag
  a transaction. This is the paper's central result — the global methods (RF, XGBoost,
  SMOTE) transfer well onto Nigeria-style mobile money fraud typologies, even though
  they were developed and validated on Western card-transaction data.
- **Confusion matrices**: Random Forest misses 10 of 330 fraud cases in the test set (53
  false positives); XGBoost also misses 10 (61 false positives). Recall is identical;
  Random Forest's lower false-positive count gives it the precision edge, while
  XGBoost's higher PR-AUC (0.985 vs 0.958) means it ranks fraud probability more
  reliably across thresholds than the single default-threshold metrics above show.
- **Feature importance diverges sharply between models.** Random Forest's importances
  spread across the engineered behavioral features as designed: `tx_velocity` and
  `balance_drain_ratio` lead (~0.27 each), then `dest_is_novel` (~0.10) — directly
  reflecting the three fraud typologies they were built to capture. XGBoost instead
  assigns ~0.85 of its importance to a single feature, `type_PAYMENT`. This isn't a bug:
  since fraud never occurs in PAYMENT transactions (Step 2), "is this a PAYMENT" is an
  extremely efficient single split that XGBoost's gain-based importance rewards heavily,
  even though it's an eliminative rule rather than a positive fraud signal. Practically,
  this means XGBoost's feature-importance chart is less useful than Random Forest's for
  the paper's typology-mapping argument, even though XGBoost scores better on PR-AUC.
