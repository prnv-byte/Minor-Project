# Memory.md — Living Progress Log

> Single source of truth for "where we are." Update this at the end of every coding session. When starting a new session (even in a new chat/tool), read this file first instead of re-scanning the whole codebase.

---

## Last Updated
20 August 2026 — Day 1, Phase 1 complete.

## Current Phase
**Phase 1 — Data & Baseline Model: DONE.** Ready to start Phase 2 (ONNX export + EZKL circuit compilation).

## Completed
- Planning docs finalized: PRD.md, Architecture.md, Rules.md, Phases.md, Design.md.
- Project folder scaffold created (`model/`, `circuit/`, `client/`, `server/`, `frontend/`, `benchmarks/`, `docs/`).
- German Credit (Statlog) dataset sourced from a public GitHub CSV mirror (1000 records, clean
  column names, no nulls) and saved to `model/data/german_credit.csv`.
- `model/train.py` written and run: preprocesses raw data (7 numeric z-scored, 13 categorical
  one-hot encoded -> 61-dim feature vector), trains logistic regression + a small FFN
  (1 hidden layer, width 16->8) in PyTorch, saves both models' weights to `model/artifacts/`.
- `model/artifacts/preprocessing.json` written — feature order, one-hot categories, and
  numeric scaler mean/std, so Phase 2 (export) and Phase 3 (client) can reproduce the exact
  same transform on new raw records.
- `model/accuracy_report.md` written: **Logistic regression selected** (78.0% test accuracy
  vs. FFN's 70.5% test accuracy — the FFN overfit, hitting 100% train accuracy on only 800
  rows / 61 features). Confusion matrices for both are in the report.
- `model/requirements.txt` and top-level `README.md` (setup + Linux/WSL2 guidance) written.

## Next Steps (Phase 2)
1. Write `model/export_onnx.py` — export the trained logistic regression (`model/artifacts/logreg.pt`)
   to ONNX. Input shape must be the 61-dim preprocessed vector (fixed order from `preprocessing.json`).
2. Install `ezkl` (pip) and write `circuit/compile.py` to compile the ONNX model into a circuit,
   generating `pk.key` / `vk.key`.
3. Verify: circuit output matches the original PyTorch model's inference on a sample of test
   records (this is Phase 2's "done" condition — don't skip it).

## Key Decisions Made
- Dataset: German Credit (Statlog) — chosen over LendingClub for cleaner, smaller tabular data (easier on EZKL circuit size).
- Model: **logistic regression selected** over the small FFN after Phase 1 comparison (see
  `model/accuracy_report.md`) — better test accuracy AND smaller/simpler circuit for EZKL.
- Target framing: `risk_label` = 1 means BAD/risky credit (flipped from the raw `credit_risk`
  column, which is 1=good). This makes the eventual ZK claim read naturally as "predicted risk
  probability is below threshold X" = safe applicant.
- Frontend: React + Vite, no SSR framework.
- Hard constraint carried through every phase: raw data never leaves the client (see `Rules.md`).

## Known Gotchas / Things to Remember
- The dataset is imbalanced (700 good / 300 risky) — a trivial "always predict good" classifier
  would already score ~70% accuracy. Both models were compared against that baseline via
  confusion matrix, not just against each other.
- The FFN overfits badly on this dataset size (61 features, 800 training rows) — don't be
  tempted to "fix" this by growing the FFN; that goes the wrong direction per `Rules.md`
  (bigger model = bigger circuit = slower proving). If NN performance matters later, consider
  regularization/dropout or fewer one-hot dims instead of more width.
- Numeric scaling (mean/std) was computed on the *full* dataset, not train-only — a documented
  simplification (see caveats in `accuracy_report.md`), acceptable for this academic MVP.
- Environment note: EZKL has limited/unreliable Windows support (Lilith doesn't work on
  Windows at all per EZKL's own docs as of this session). Recommended WSL2 (or native Linux/Mac)
  from Phase 2 onward — added to top-level README.

## File Map
Matches `Architecture.md`, with `model/artifacts/` added (not explicitly in the original tree)
to hold trained weights (`logreg.pt`, `ffn.pt`) and `preprocessing.json`.

---

### How to use this file
- At the **start** of a session: read this file top to bottom before touching code.
- At the **end** of a session: update "Last Updated," move finished items from "In Progress" to "Completed," refresh "Next Steps."
- If a decision contradicts `Architecture.md` or `Rules.md`, note it here **and** update that doc — don't let them drift out of sync.
