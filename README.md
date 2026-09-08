# NairaGuard Fraud Detection

A machine learning fraud detection project built for the MIT8212 seminar paper *"Machine Learning-Based
Fraud Detection in Nigerian Digital Payment Platforms: A Comparative Case Study"* (Miva Open University).
NairaGuard is a fictional Nigerian fintech platform used for illustration only. It is not a real company.

## Background

Global fraud detection research (Random Forest, XGBoost, SMOTE for class imbalance) has proven highly
effective, but has been tested almost entirely on Western card-transaction data. Nigerian and African
fintech research describes real local fraud patterns, such as SIM-swap fraud, agent fraud, account
takeover, and mobile money fraud, but rarely tests the high-performing global algorithms directly against
that context. This project closes that gap on a small scale: it applies the proven global methods to a
Nigeria-style mobile money dataset and evaluates how well they hold up.

## Dataset

The dataset is generated rather than downloaded. `src/generate_data.py` produces a synthetic mobile
money transaction dataset that follows the PaySim column schema (Lopez-Rojas et al.; also used in
Lokanan, 2023, *Applied AI Letters*), so results stay comparable to the literature this paper cites.
Fraud cases are injected using three typologies drawn directly from the Nigerian and African fintech
literature, rather than generic patterns. See [data/README.md](data/README.md) for details, including a
note on why a synthetic dataset was used.

## Method

1. Generate the dataset, injecting fraud through three typologies: account takeover / SIM-swap style (a
   sudden large transfer draining an account to near zero), velocity fraud (an unusually high frequency
   of transactions in a short window), and agent/cash-out fraud (a disproportionately large cash-out to
   a destination account with no prior history).
2. Load and explore the data, quantifying class imbalance and the transaction type breakdown.
3. Engineer behavioral features, including transaction velocity, balance-drain ratio, and destination
   account novelty, that mirror the fraud injection logic directly.
4. Address class imbalance with SMOTE.
5. Train and evaluate Random Forest and XGBoost classifiers, using accuracy, precision, recall, F1, and
   PR-AUC.
6. Visualize the results: confusion matrices, a model comparison chart, and a feature importance chart.

## Findings

To be added once the pipeline has run. Charts are saved to `results/`; the code that produced them is
in `src/`.

## Project structure

```
nairaguard-fraud-detection/
├── README.md
├── requirements.txt
├── data/          # dataset lives here locally (not committed, see data/README.md)
├── src/           # pipeline code
└── results/       # saved charts/images
```
