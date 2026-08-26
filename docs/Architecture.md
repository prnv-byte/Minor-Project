# Architecture.md

## High-Level Flow

```
┌─────────────────────────────┐
│   User Device (local)       │
│                              │
│  Raw record → Model + EZKL  │
│  (infer + prove)            │
│                              │
│  Raw data NEVER leaves here │
└──────────────┬───────────────┘
               │ proof + public output (threshold claim)
               ▼
┌─────────────────────────────┐
│   Verifier (Flask API)      │
│                              │
│  /verify → checks proof     │
│  "risk below threshold"?    │
└──────────────┬───────────────┘
               │ result
               ▼
┌─────────────────────────────┐
│   React Frontend             │
│  input form → status →      │
│  proof progress → result    │
└─────────────────────────────┘
```

Reusable abstraction: **Record → Model → Threshold Claim → ZK Proof → Verifier**. This is domain-agnostic by design (financial now, extensible to medical/insurance later without rearchitecting).

## Tech Stack

| Layer | Choice | Notes |
|---|---|---|
| Model training | PyTorch | logistic regression / small feed-forward NN |
| Model export | ONNX | intermediate format required by EZKL |
| ZK circuit + proving | EZKL (Halo2) | Rust-based, installed via pip/cargo |
| Local client | Python | runs inference + proof generation on-device |
| Backend / verifier | Flask | Python, minimal REST API |
| Frontend | React (Vite) | lightweight, no need for Next.js/SSR here |
| Dataset | German Credit (Statlog) or LendingClub | public, tabular |
| Dev tooling | Git, VS Code | version control, no cloud dependency required for MVP |

## Folder Structure

```
zkml-credit-risk/
├── docs/
│   ├── PRD.md
│   ├── Architecture.md
│   ├── Rules.md
│   ├── Phases.md
│   ├── Design.md
│   └── Memory.md
│
├── model/
│   ├── data/                 # raw + processed dataset (gitignored if large)
│   ├── train.py               # training script (logistic regression + small NN)
│   ├── export_onnx.py         # exports trained model to ONNX
│   └── accuracy_report.md     # baseline accuracy documentation
│
├── circuit/
│   ├── model.onnx              # exported model
│   ├── settings.json           # EZKL circuit settings
│   ├── pk.key / vk.key         # proving / verification keys
│   └── compile.py              # EZKL circuit compilation script
│
├── client/
│   ├── infer_and_prove.py     # local inference + proof generation
│   └── sample_records/         # test records for local demo use
│
├── server/
│   ├── app.py                  # Flask app entrypoint
│   ├── routes/
│   │   ├── predict_and_prove.py
│   │   └── verify.py
│   ├── verifier.py             # wraps EZKL's verify() call
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/         # InputForm, ProofStatus, ResultCard
│   │   ├── pages/
│   │   ├── api/                # fetch wrappers to Flask API
│   │   └── App.jsx
│   ├── package.json
│   └── vite.config.js
│
├── benchmarks/
│   ├── run_benchmarks.py       # measures proof/verify time vs model size/precision
│   └── results/                # CSV/JSON outputs + charts
│
└── README.md
```

## Data Flow (step-by-step)

1. User enters financial details in the React frontend (`frontend/`).
2. Frontend calls the **local** client (`client/infer_and_prove.py`, run as a local process/service) — never the network — with the raw record.
3. Client runs model inference locally, then invokes EZKL to generate a zero-knowledge proof + the public threshold claim (e.g. `risk_below_threshold: true`).
4. Only the proof + public claim are sent over HTTP to the Flask server's `/verify` endpoint.
5. Flask calls `verifier.py`, which uses EZKL's verification function against the known verification key (`circuit/vk.key`).
6. Result (`valid: true/false` + the claim) is returned to the frontend and displayed.

## API Contract

### `POST /predict-and-prove` (optional convenience wrapper, still local-only in deployment)
- Input: raw record (financial features)
- Output: `{ proof, public_output }`
- **Must only ever be called from localhost / the user's own device in the demo. Never expose this as a public network endpoint that accepts raw data.**

### `POST /verify`
- Input: `{ proof, public_output }` — no raw data
- Output: `{ valid: boolean, claim: string }`

## Extension Points (for major project, do not build yet)
- `registry/` — model registry supporting multiple domain models (diabetes, loan, vaccine), versioned by circuit hash.
- `onchain/` — Solidity verifier contract generated via EZKL's export, deployed to a testnet, with gas-cost benchmarking.

These plug into the same `proof + public_output` interface without changing `client/` or the core circuit logic.
