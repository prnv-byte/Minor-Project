# PRD.md — Project Requirements Document

## Project Name
Privacy-Preserving Credit Risk Verification using Zero-Knowledge Machine Learning (ZKML)

## Problem Statement
Credit-scoring and financial-verification systems today require applicants to share their complete raw financial records (bank statements, transaction history, credit history) with third-party verifiers. This creates unnecessary privacy exposure and regulatory risk (e.g. India's DPDP Act). We need a way for a user to *prove* a claim about their financial risk ("my credit risk score is below threshold X") without revealing the underlying raw data to anyone.

## Goal
Build a working end-to-end pipeline where:
1. A user's raw financial record never leaves their device.
2. A machine learning model runs locally and produces a risk-threshold claim.
3. A zero-knowledge proof is generated locally, proving the claim was computed correctly.
4. A verifier service checks the proof's validity — without ever seeing the raw data.

## Target Users / Personas
- **Applicant (data owner):** the person whose financial data is being evaluated. Wants privacy — doesn't want to hand over raw records.
- **Verifier (lender / evaluating party):** wants a trustworthy answer to "is this applicant below the risk threshold?" without taking on liability for storing sensitive data.
- **Project evaluators (Amity panel / examiners):** the immediate audience for the minor project demo — care about correctness, working proof pipeline, and clear benchmarks.

Note: this is an academic minor project / proof-of-concept, not a production fintech product. The "users" above are the personas the demo simulates.

## Core Features (Minor Project MVP Scope)
1. **Model training** — logistic regression / small feed-forward NN trained on a public credit dataset (German Credit or LendingClub), with a documented accuracy report.
2. **ONNX export + EZKL circuit compilation** — convert the trained model into a zero-knowledge circuit (Halo2 proving system via EZKL).
3. **Local inference + proof-generation client** — script/service that runs on the "user's" side: takes a raw record, runs inference locally, and produces a ZK proof + public threshold claim (no raw data leaves this boundary).
4. **Flask verifier API** — `/predict-and-prove` (executed locally) and `/verify` (accepts only proof + public output, never raw data) endpoints.
5. **React frontend** — input form → local inference status → proof generation progress → verification result. Never sends raw data to any server.
6. **Benchmarking** — proof generation time and verification time vs. model size and quantization precision (this is the project's core research contribution).

## Out of Scope (Major Project / Future Extension)
- Model marketplace (registry of multiple domain models — diabetes, loan, vaccine, etc.)
- On-chain verifier (Solidity smart contract, gas-metered verification)
- Multi-domain generalization beyond financial risk

These are explicitly deferred to the major project, but the architecture (see `Architecture.md`) is designed so they can be bolted on without a rewrite.

## Success Criteria
- Proof correctness: circuit output matches original model inference on the full test set.
- End-to-end demo works live: input → local proof → API verification → result, with zero raw data transmitted.
- Benchmark data collected and documented (proof time vs. model size/precision).
- Codebase and report ready for minor project submission and viva.

## Key Constraint (non-negotiable)
Raw user data must never be transmitted off the local device/client. Only the zero-knowledge proof and the public threshold claim are ever sent to the verifier. This constraint should be treated as a hard rule throughout implementation (see `Rules.md`).
