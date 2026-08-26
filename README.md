# ZKML Credit Risk — Privacy-Preserving Credit Risk Verification

B.Tech minor project (Amity ASET, batch 2023–2027). See `docs/PRD.md` for full requirements,
`docs/Architecture.md` for system design, `docs/Rules.md` for hard constraints, and
`docs/Memory.md` for current progress / where-we-left-off.

## Setup

### OS: use Linux (native or WSL2 on Windows) — not native Windows

EZKL (the ZK proving library this project depends on from Phase 2 onward) only has partial,
less-supported Windows compatibility, and some of its tooling (e.g. the Lilith component)
does not work on Windows at all. Phase 1 (pure PyTorch) would run fine on native Windows, but
since Phase 2+ requires EZKL, standardize on Linux now to avoid a painful switch later.

- **If you're on Windows:** install WSL2 (Ubuntu) — `wsl --install` in an admin PowerShell,
  then do all work inside the WSL Ubuntu shell (VS Code's "WSL" remote extension makes this
  seamless — you keep your normal Windows editor UI).
- **If you're on Mac:** native is fine.
- **If you're on Linux:** native is fine.

### Python environment

```bash
cd model
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Phase 1 — Data & Baseline Model (done)

```bash
cd model
python3 train.py
```

This loads `model/data/german_credit.csv`, preprocesses it, trains a logistic regression
baseline and a small feed-forward NN, and writes `model/accuracy_report.md`.

Dataset: German Credit (Statlog), sourced from a public GitHub mirror of the UCI dataset
(`model/data/german_credit.csv`, 1000 records, CC BY 4.0 licensed original from UCI).

See `docs/Phases.md` for the full phase-by-phase build plan.
