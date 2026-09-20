# Model Accuracy Report - Phase 1

## Dataset
- Source: German Credit (Statlog), 1000 records, 20 raw features.
- Preprocessing: 7 numeric features standardized (z-score),
  13 categorical features one-hot encoded.
- Final feature vector dimension: **61**.
- Train/test split: 800/200 (80/20, stratified, random_state=42).
- Target: `risk_label` - 1 = bad/risky credit, 0 = good credit (flip of raw `credit_risk` column).
  Class balance: 300 risky / 700 good in the full dataset (imbalanced, noted below).

## Logistic Regression (baseline)
- Train accuracy: **0.7788**
- Test accuracy: **0.7800**
- Test confusion matrix (positive = risky/bad): TP=36 TN=120 FP=20 FN=24

## Small Feed-Forward NN (1 hidden layer, width 16 -> 8)
- Train accuracy: **1.0000**
- Test accuracy: **0.7050**
- Test confusion matrix (positive = risky/bad): TP=31 TN=110 FP=30 FN=29

## Model Selection

**Selected model: Logistic Regression**

Logistic regression is selected as the default unless the FFN shows a clear, meaningful accuracy improvement (>1pt), because a smaller/linear model keeps the EZKL circuit size and proving time manageable (see Rules.md).

## Notes / Caveats
- The dataset is imbalanced (70% good / 30% risky). Accuracy alone can be misleading here,
  the confusion matrix above is the more honest signal; a trivial "always predict good"
  classifier would score ~70% test accuracy without being useful. Both models are compared
  against that baseline, not just against each other.
- Numeric feature scaling (mean/std) was computed on the full dataset rather than
  train-only, which is a simplification acceptable for this academic MVP but would be a
  data-leakage concern in a production setting.
- All preprocessing parameters (feature order, one-hot categories, scaler mean/std) are
  saved to `model/artifacts/preprocessing.json` so that Phase 2 (ONNX export) and Phase 3
  (local inference client) can reproduce the exact same transform on new raw records.
