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
