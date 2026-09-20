# ZKML Credit Risk — Setup & Run Steps

## 1. Activate the virtual environment

```bash
cd ~/zkml-credit-risk
source .venv/bin/activate
```

You should see `(.venv)` in the terminal prompt.

## 2. Run Phase 1 — train the model

Go to the model directory:

```bash
cd ~/zkml-credit-risk/model
python3 train.py
```

This reads `data/german_credit.csv`, trains both models, and writes:

- `artifacts/logreg.pt`
- `artifacts/ffn.pt`
- `artifacts/preprocessing.json`
- `artifacts/X_test.npy` / `artifacts/y_test.npy`
- `accuracy_report.md`

Expected accuracy:

- Logistic Regression: **78.0%**
- FFN: **70.5%**

The fixed random seed should reproduce these results.

## 3. Run Phase 2 — export, compile, verify

### 3.1 Export to ONNX

From `model/`:

```bash
python3 export_onnx.py
```

This exports `logreg.pt` to `../circuit/model.onnx` and checks PyTorch vs ONNX on 20 test records.

Expected:

```text
PyTorch vs ONNX claim mismatches on 20 test records: 0
```

### 3.2 Compile the EZKL circuit

```bash
cd ../circuit
python3 compile.py
```

For the lite setup, this generates settings, calibration, the compiled circuit, the local SRS, and setup keys. It may take a couple of minutes.

If the keys already exist, the script skips them. Use `--force` only when intentionally regenerating them.

### 3.3 Verify circuit correctness

Still inside `circuit/`:

```bash
python3 verify_circuit.py
```

This generates **10 real zero-knowledge proofs** from **10 held-out records**, verifies them, and compares the circuit claim with PyTorch.

Expected final output:

```text
10/10 records: proof verified AND circuit claim matches PyTorch.
```

Expected proving time in the project notes is about **7.5–9 seconds per record**, or roughly **1–2 minutes total**, depending on the machine.

## 4. Quick troubleshooting

### `ModuleNotFoundError: ezkl`

Activate the venv:

```bash
cd ~/zkml-credit-risk
source .venv/bin/activate
```

If needed:

```bash
pip install -r model/requirements.txt
```

### `compile.py` hangs or fails while generating the SRS

Confirm you are using this project's `circuit/compile.py`. It uses local `ezkl.gen_srs()` rather than `ezkl.get_srs()`, which may require a remote EZKL server connection.

### Accuracy numbers do not match 78.0% / 70.5%

Check that the expected PyTorch version and fixed random seed are being used. Different library versions can cause small floating-point differences.

## 5. Phase status

These areas are planned for Phase 3 onward and do not have a runnable implementation yet:

- `client/`
- `server/`
- `frontend/`
- `benchmarks/`

Do not mark Phase 2 complete until the circuit correctness test passes.

## 6. Git — push code files and this MD file only

**Do not use `git add .`** because the project contains generated EZKL artifacts such as proving keys and SRS files.

Check the repository:

```bash
cd ~/zkml-credit-risk
git status
```

Stage only the Python code files you want to push:

```bash
git add model/train.py model/export_onnx.py circuit/compile.py circuit/verify_circuit.py
```

Then stage this documentation file:

```bash
git add steps.md
```

Check exactly what is staged:

```bash
git status
```

Make sure generated files such as these are **not** staged:

- `pk.key`
- `vk.key`
- `kzg.srs`
- `model.compiled`
- generated `.onnx` artifacts
- generated `.npy` artifacts

Then:

```bash
git commit -m "Add project setup and run steps"
```

Finally:

```bash
git push origin main
```

### Safe Git rule

Prefer explicitly naming files with `git add` instead of `git add .`. This keeps generated proving keys, SRS files, and other generated artifacts out of GitHub.
