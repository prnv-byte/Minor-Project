# Rules.md — Boundaries for AI-assisted coding

## Hard Rules (never break these)
1. **Raw data never leaves the local/client boundary.** No raw financial record should ever be sent in an HTTP request body to the Flask server, logged to a file, or written to any database. Only `proof` and `public_output` cross that boundary.
2. **Never fake or mock a proof result to make a demo "work."** If EZKL proof generation or verification fails, surface the actual error — do not hardcode `valid: true` anywhere, even temporarily. This defeats the entire point of the project.
3. **Never commit secrets, API keys, or `.env` files.** Use `.env.example` for documenting required variables.
4. **Do not fabricate benchmark numbers.** All numbers in `benchmarks/results/` must come from an actual run of `run_benchmarks.py`. If asked to "fill in" benchmark data before it exists, refuse and generate it instead.

## Libraries — Use
- **Python:** PyTorch, `ezkl`, `onnx`, `onnxruntime`, Flask, `pytest` for tests
- **Frontend:** React (Vite), plain `fetch`/`axios`, minimal state management (React state/hooks — no Redux needed at this scale)
- **Data:** `pandas`, `scikit-learn` (for preprocessing / baseline comparison only, not for the final provable model)

## Libraries — Avoid
- No Next.js / SSR frameworks for the frontend — adds complexity with no benefit for this scope.
- No large NN architectures (transformers, deep CNNs) — circuit size and proving time will blow up. Stick to logistic regression or a small feed-forward NN (1–2 hidden layers, small width).
- No GPU-dependent proving setup — the project should run on a standard laptop (per `PRD.md` constraints).
- No ORMs / databases unless a real need arises — this project doesn't need to persist raw data, and shouldn't.

## Error Handling Conventions
- **Python (Flask/client):** use explicit `try/except` blocks around EZKL calls and model inference; return meaningful error messages and correct HTTP status codes (400 for bad input, 500 for server-side failure) — never a silent `pass`.
- **React:** every API call must handle the error case in the UI (show a clear error state, not a blank screen or infinite spinner).
- Circuit/proof failures should be treated as first-class errors, not edge cases — log them clearly during development (without logging raw input data).

## Code Style
- Python: PEP8, type hints where reasonable, docstrings on non-trivial functions.
- JS/React: functional components + hooks, consistent naming (`camelCase` for variables/functions, `PascalCase` for components).
- Keep functions small and single-purpose — this codebase will be read by an examiner/panel, clarity matters as much as correctness.

## What the AI Should Do
- Ask before making an architectural decision not already covered in `Architecture.md`.
- Write tests for circuit-correctness (does circuit output match plain model inference?) before considering a phase "done."
- Update `Memory.md` at the end of every work session with what was completed, what's next, and any gotchas discovered.
- Flag anything that looks like it will blow up proof generation time (e.g., accidentally increasing model size) before implementing it.

## What the AI Should NOT Do
- Should not skip straight to frontend polish before the core proof pipeline (`model` → `circuit` → `client` → `server`) is working end-to-end.
- Should not silently change the chosen dataset, model architecture, or tech stack without flagging it first.
- Should not introduce the major-project scope (model marketplace, on-chain verifier) into the minor project codebase.
- Should not reduce or bypass the "raw data never leaves the client" rule for convenience, even in dev/test code — build test fixtures that respect the same boundary.
