# Phases.md — Build Plan

Each phase should be fully working and tested before moving to the next. Don't let the AI jump ahead to a later phase's work.

## Phase 1 — Data & Baseline Model
- Load and clean the German Credit (or LendingClub) dataset.
- Preprocess: encode categorical features, train/test split.
- Train a baseline logistic regression model in PyTorch.
- Train a small feed-forward NN for comparison.
- Output: `model/accuracy_report.md` documenting both models' accuracy.
- **Done when:** a model is selected and its accuracy is documented.

## Phase 2 — ONNX Export & EZKL Circuit Compilation
- Export the selected model to ONNX (`model/export_onnx.py`).
- Compile the ONNX model into an EZKL circuit (`circuit/compile.py`).
- Generate proving key (`pk.key`) and verification key (`vk.key`).
- Verify circuit output matches the original model's inference on a sample of test records.
- **Done when:** circuit compiles and its output is proven to match the original model.

## Phase 3 — Local Inference + Proof Generation Client
- Build `client/infer_and_prove.py`: takes a raw record, runs local inference, generates a ZK proof + public threshold claim.
- Test against multiple sample records from `client/sample_records/`.
- Record informal proof-generation timing as a sanity check.
- **Done when:** the client reliably produces a valid proof + claim for arbitrary valid input, entirely locally.

## Phase 4 — Flask Verifier API
- Build `server/app.py` with `/predict-and-prove` and `/verify` routes.
- `/verify` must accept only `{ proof, public_output }` — never raw data.
- Wrap EZKL's verification call in `server/verifier.py`.
- Write basic tests confirming valid proofs pass and tampered/invalid proofs fail verification.
- **Done when:** the API correctly verifies real proofs and rejects invalid ones.

## Phase 5 — React Frontend
- Build the input form (`components/InputForm`).
- Wire it to trigger the local client (not send raw data to the server).
- Build proof-generation status indicator and result display.
- Connect result display to the `/verify` response.
- **Done when:** a user can complete the full flow in the browser: enter details → see proof generate → see verified result.

## Phase 6 — Benchmarking
- Build `benchmarks/run_benchmarks.py`: measures proof generation and verification time across different model sizes and EZKL quantization/precision settings.
- Store results in `benchmarks/results/` (CSV/JSON + a chart).
- **Done when:** there's real, reproducible benchmark data — this is the project's core research contribution, don't skip or rush it.

## Phase 7 — Testing, Edge Cases & Documentation
- Test threshold-comparison logic at boundary values (risk score exactly at / just above / just below threshold) under quantization.
- Fix any correctness bugs found.
- Write/update the README and finalize technical documentation.
- **Done when:** edge cases are verified correct and the codebase is documented well enough for someone else (or an examiner) to follow.

## Phase 8 — Report & Demo Prep
- Consolidate accuracy, proof-correctness, and benchmark results into the minor project report.
- Prepare a demo script (input → local proof → API verify → result) for the viva.
- Final round of testing as a buffer before submission.

---

**Today's starting point:** Phase 1. Do not start Phase 2 work until Phase 1's model + accuracy report is done and reviewed.
