# Memory.md — Living Progress Log

> Single source of truth for "where we are." Update this at the end of every coding session. When starting a new session (even in a new chat/tool), read this file first instead of re-scanning the whole codebase.

---

## Last Updated
20 September 2026 — Phase 2 independently re-verified on Pranav's own machine; added run_phase2.sh orchestrator. Phase 3 not yet started.

## Current Phase
**Phase 2 — ONNX Export & EZKL Circuit Compilation: DONE and independently verified on two machines.** Ready to start Phase 3 (Local Inference + Proof Generation Client).

## Iteration Log (Phase 2 re-run, 19 Sept 2026)
- **Step 1 (ONNX export)**: ran `model/export_onnx.py` against `model/artifacts/logreg.pt`. Wrote `circuit/model.onnx`. Internal sanity check compared the ONNX graph's claim output against the original PyTorch model on 20 held-out test records: **0 mismatches**. Note: torch's exporter auto-upgraded the working opset to 18 internally then converted back down to the requested opset 13 for the file on disk; this is exporter-internal tooling behavior, not a correctness concern, the mismatch check is what actually confirms correctness.
- **Step 2 (EZKL circuit compilation)**: ran `circuit/compile.py --force` from a clean slate (all Phase 2 artifacts deleted first). Two real bugs were found and fixed during this run, not glossed over:
  - `ezkl.get_srs()`'s actual argument order didn't match what the script assumed (`settings_path` first, not `srs_path`) — fixed by calling it with keyword arguments.
  - `ezkl.get_srs()` itself then failed with a field-modulus error: it fetches a pre-generated SRS from EZKL's remote server, which this sandboxed environment has no network route to. Switched to `ezkl.gen_srs()`, which EZKL documents as being for local/testing use (as opposed to a production trusted-setup ceremony), generating the SRS locally instead. This fits the project's own constraints (`PRD.md`: no cloud dependency required for the MVP; `Rules.md`: must run on a standard laptop).
  - Calibration reported a **Numerical Fidelity Report with 0 error across every metric** (mean/median/max error, MSE, percent error), confirming the chosen scale (input_scale=13, param_scale=13) introduces no measurable quantization error for this model.
  - Final circuit: **logrows=15, 382 rows**, same as the original run, confirming these numbers come from the model's actual complexity, not an artifact of a particular run.
  - Output: `settings.json`, `model.compiled`, `kzg.srs` (4,194,564 bytes locally generated), `pk.key` (163,653,739 bytes), `vk.key` (75,271 bytes).
- **Step 3 (circuit correctness verification)**: ran `circuit/verify_circuit.py` against the freshly compiled circuit and freshly generated keys. For 10 held-out test records: generated a real witness, a real proof, ran real `ezkl.verify()`, and decoded the circuit's public output. **Result: 10/10 verified and in exact agreement with the original PyTorch model, 0 mismatches.** Average real proof time: **7.55s/record** (witness generation ~0.02–0.03s, verification ~0.025–0.03s). Slightly faster than the previous run's 9.3s average, ordinary machine-load variance between runs, not a code or circuit change (settings are identical). Report: `circuit/correctness_report.json`.

**Phase 2 status: DONE.** Rebuilt from a completely clean slate (all prior Phase 2 artifacts deleted first), every step re-run for real, two real bugs found in `compile.py` and fixed (not worked around or hidden), and correctness re-confirmed with fresh proofs against the newly compiled circuit.

## Independent verification on Pranav's own machine (20 Sept 2026)
The full Phase 2 pipeline (`export_onnx.py` → `compile.py` → `verify_circuit.py`) was re-run independently on Pranav's own Ubuntu machine, not the sandbox that originally built it. Result: **10/10 proofs verified and in exact agreement with PyTorch**, using the same pre-shipped `pk.key`/`vk.key` from the delivered zip (`compile.py` correctly detected they already existed and skipped regenerating them). Proof generation was noticeably faster on this hardware, **~2.3–2.9s/record** vs. ~7.5–9s in the sandbox, a hardware difference, not a code difference. This is a stronger correctness signal than the original sandbox run alone: it confirms the shipped keys and compiled circuit are genuinely portable across machines, not artifacts of one specific environment.

## Added: run_phase2.sh (one-command pipeline runner)
Added `run_phase2.sh` at the project root, a thin orchestrator that calls `export_onnx.py`, `compile.py`, and `verify_circuit.py` in order from the correct directories. This does NOT replace the three individual scripts, each still works standalone (e.g. re-running just `verify_circuit.py` without re-exporting or re-compiling). The scripts stay separate on purpose: `compile.py` is expensive and idempotent (shouldn't rerun needlessly and would invalidate working keys if it did), `export_onnx.py` only needs rerunning after retraining, and keeping them distinct matches how `Architecture.md`/`Phases.md` describe Phase 2 as three separately-checkable deliverables. `run_phase2.sh` is purely a convenience wrapper on top of that, not a redesign of it. Usage: `./run_phase2.sh` (normal) or `./run_phase2.sh --force` (forces circuit recompilation).

## Bug found and fixed during this iteration: missing accuracy_report.md
While assembling the full codebase for delivery, `model/accuracy_report.md` was found to be missing even though `model/artifacts/*.pt` existed. The current `train.py` on disk turned out to be a rewritten version (its own docstring says "Recreated for a fresh session") that dropped the `write_accuracy_report()` function entirely during that rewrite. This was a real regression, restored the function, reran `train.py` (identical results: 78.0%/70.5%, confirming the fixed seed still reproduces exactly), then re-ran `export_onnx.py` and `verify_circuit.py` end-to-end again to confirm the freshly retrained `logreg.pt` still produces a circuit that agrees with PyTorch on all 10 sampled test records. It does, 10/10, no regressions introduced by the fix.

## Completed
- Planning docs finalized: PRD.md, Architecture.md, Rules.md, Phases.md, Design.md.
- Project folder scaffold created (`model/`, `circuit/`, `client/`, `server/`, `frontend/`, `benchmarks/`, `docs/`).
- **Phase 1**: German Credit (Statlog) dataset sourced and verified (1000 records, 61-dim preprocessed feature vector). Logistic regression selected over a small FFN (78.0% vs. 70.5% test accuracy). Artifacts: `model/artifacts/logreg.pt`, `preprocessing.json`, `X_test.npy`/`y_test.npy` (test split persisted for reuse in Phase 2 verification).
- **Phase 2 (previous run, superseded by the iteration log above; kept here for the record)**:
  - `model/export_onnx.py` — exports the trained logistic regression model to ONNX, with the threshold comparison baked into the graph itself (`circuit/model.onnx`). The graph outputs only a 0/1 safety claim, never the raw risk probability, so EZKL's public output never leaks the applicant's exact score. Verified against the original PyTorch model on 20 test records at export time (0 mismatches).
  - `circuit/compile.py` — reproducible EZKL compile pipeline: generates settings, calibrates quantization scale against real held-out records (not random noise), compiles the circuit, generates the KZG SRS locally, and runs setup to produce `pk.key`/`vk.key`. Idempotent by default (`--force` to regenerate).
  - `circuit/verify_circuit.py` — the actual Phase 2 "done" check per `Phases.md`: generates a **real** witness, a **real** proof, and a **real** `ezkl.verify()` call for 10 held-out test records, then compares each circuit claim against the original PyTorch model's claim. Result: **10/10 verified and in agreement**, 0 mismatches. Report saved to `circuit/correctness_report.json`. No proof result was ever hardcoded (see `Rules.md`).
  - Circuit stats (from `settings.json`): input/param scale 13, logrows 15, 382 rows, single required lookup (Sigmoid). Average real proof generation time on this machine: **~9.3s/record** (witness generation ~0.02–0.13s, verification ~0.03s). This proving time is a first real data point for Phase 6's benchmarking, though Phase 6 itself is still not started.

## Next Steps (Phase 3 — Local Inference + Proof Generation Client)
Per `Phases.md`, Phase 3's job is: "Build `client/infer_and_prove.py`: takes a raw record, runs local inference, generates a ZK proof + public threshold claim." Concretely, this means:
1. **`client/infer_and_prove.py`** — a single script/function that, given a raw applicant record (the kind of input a real form would collect: age, loan amount, housing type, etc., not already-preprocessed numbers):
   - Applies the exact same preprocessing as `model/train.py`, using the saved `preprocessing.json` recipe (numeric mean/std, one-hot categories, feature order), so a brand-new record gets turned into the same 61-dimensional vector the model was trained on.
   - Runs local inference (loads `logreg.pt`, computes the risk probability, applies the threshold), OR skips straight to proof generation, since the ONNX/circuit path already computes inference internally as part of proving; the design decision of whether to also run a plain PyTorch inference locally as a fast sanity check before spending ~3-9 seconds on a full proof is one to make in Phase 3, not assume now.
   - Calls the same `ezkl.gen_witness()` / `ezkl.prove()` sequence already proven correct in `verify_circuit.py`, but packaged as a reusable function instead of a verification loop.
   - Outputs exactly `{ proof, public_output }`, per `Architecture.md`'s API contract, nothing else, no raw record, no intermediate probability.
2. **`client/sample_records/`** — a handful of realistic raw applicant records (not preprocessed vectors) to test the client against, currently this folder is empty.
3. **Informal timing sanity check** — confirm proof generation for arbitrary new records lands in the same ballpark already observed (~2.3–9s/record depending on hardware), not a formal benchmark (that's Phase 6).

**Done when** (per `Phases.md`): "the client reliably produces a valid proof + claim for arbitrary valid input, entirely locally." That means testing against more than one sample record, and confirming no raw data touches disk or network in the process, this is the same hard constraint carried through every phase (`Rules.md`).

## Key Decisions Made
- Dataset: German Credit (Statlog) — chosen over LendingClub for cleaner, smaller tabular data (easier on EZKL circuit size).
- Model: **logistic regression selected** over a small FFN after Phase 1 comparison — better test accuracy AND smaller/simpler circuit for EZKL.
- Target framing: `risk_label` = 1 means BAD/risky credit (flipped from the raw `credit_risk` column).
- **ONNX graph design**: the exported graph computes probability, thresholds it, and outputs only a 0/1 claim — the probability itself never becomes a circuit output, so it stays private even though it's computed as an intermediate value. This was a deliberate call to match `Architecture.md`'s "public output should be a threshold CLAIM, not the risk score itself."
- Frontend: React + Vite, no SSR framework.
- Hard constraint carried through every phase: raw data never leaves the client (see `Rules.md`).

## Known Gotchas / Things to Remember
- The dataset is imbalanced (700 good / 300 risky) — noted in Phase 1's accuracy report; doesn't affect Phase 2 but worth remembering when Phase 7's boundary-value testing looks at threshold edge cases.
- The FFN overfits badly on this dataset size — logistic regression stays the model going forward; do not revisit the FFN without a reason.
- `circuit/compile.py` is intentionally idempotent — it will NOT regenerate `pk.key`/`vk.key` if they already exist and pass verification, since EZKL's trusted setup produces a different (equally valid, but different) key pair each time it runs, and there's no reason to invalidate a working key pair. Use `--force` only if `model.onnx` changes.
- Boundary-value testing (risk score exactly at / near the 0.5 threshold, under quantization) is explicitly **Phase 7** scope per `Phases.md`, not Phase 2 — `verify_circuit.py` intentionally only checks agreement on ordinary held-out records, not edge cases. Don't be tempted to fold that testing into Phase 2 or skip it later.
- Proof generation takes ~7.5–9 seconds per record on this machine's CPU (no GPU used, per `Rules.md`, varies run to run with ordinary machine load). Keep this in mind when designing Phase 5's frontend UX (the "Proof Generation" pipeline step will need to reflect a real multi-second wait, not appear instant).

## File Map
Matches `Architecture.md`, with `model/artifacts/` (trained weights, preprocessing recipe, persisted test split) and populated `circuit/` (ONNX export, compiled circuit, SRS, proving/verification keys, and the correctness report) both now filled in as of Phase 2. `run_phase2.sh` added at the project root as a convenience orchestrator (not part of `Architecture.md`'s original structure, but doesn't change it, just calls the existing scripts in order).

---

### How to use this file
- At the **start** of a session: read this file top to bottom before touching code.
- At the **end** of a session: update "Last Updated," move finished items from "In Progress" to "Completed," refresh "Next Steps."
- If a decision contradicts `Architecture.md` or `Rules.md`, note it here **and** update that doc — don't let them drift out of sync.
