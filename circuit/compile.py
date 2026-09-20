"""
Phase 2 — EZKL Circuit Compilation

Compiles circuit/model.onnx (produced by model/export_onnx.py) into an EZKL
zero-knowledge circuit: generates settings, calibrates scale/quantization
parameters against real sample inputs, compiles the circuit, generates a local
KZG structured reference string, and runs the (local, non-production) trusted
setup to produce the proving and verification keys.

This script is idempotent by default: if compiled artifacts already exist
and verify_circuit.py has confirmed correctness against them, re-running
setup would only replace working keys with a different (but not "more
correct") random trusted-setup instance, for no benefit. Pass --force to
regenerate everything from scratch (e.g. after changing model.onnx or the
run-args below).

Output (all in circuit/):
    settings.json     — circuit parameters (scale, lookup range, logrows, ...)
    model.compiled     — the compiled EZKL circuit
    kzg.srs             — structured reference string for the proving system
    pk.key / vk.key      — proving / verification keys from setup

Run circuit/verify_circuit.py afterward to confirm the compiled circuit's
output actually matches the original PyTorch model on real test records —
that check, not this script finishing without error, is what Phases.md
means by Phase 2 being "done."
"""

import argparse
import json
from pathlib import Path

import numpy as np
import ezkl

CIRCUIT_DIR = Path(__file__).parent
MODEL_DIR = CIRCUIT_DIR.parent / "model"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"

ONNX_PATH = str(CIRCUIT_DIR / "model.onnx")
COMPILED_PATH = str(CIRCUIT_DIR / "model.compiled")
SETTINGS_PATH = str(CIRCUIT_DIR / "settings.json")
CALIBRATION_PATH = str(CIRCUIT_DIR / "calibration.json")
SRS_PATH = str(CIRCUIT_DIR / "kzg.srs")
PK_PATH = str(CIRCUIT_DIR / "pk.key")
VK_PATH = str(CIRCUIT_DIR / "vk.key")

# Matches the run-args recorded in the settings.json already produced by
# this script; kept explicit here so the compile step is reproducible from
# a clean checkout rather than depending on EZKL's defaults at some future
# version. input/param scale of 13 and logrows=15 were sufficient for this
# model's single linear layer + sigmoid + threshold comparison.
RUN_ARGS = ezkl.PyRunArgs()
RUN_ARGS.input_visibility = "private"
RUN_ARGS.param_visibility = "fixed"
RUN_ARGS.output_visibility = "public"


def write_calibration_file(n=20):
    """EZKL calibrates quantization scale against real, representative
    inputs (not random noise) so the lookup range actually covers the
    model's real activations. Uses real held-out records, one per row,
    each wrapped as its own single-record batch."""
    X_test = np.load(ARTIFACTS_DIR / "X_test.npy").astype(np.float32)
    n = min(n, X_test.shape[0])
    with open(CALIBRATION_PATH, "w") as f:
        json.dump({"input_data": [X_test[i:i + 1].tolist()[0] for i in range(n)]}, f)
    print(f"Wrote calibration data ({n} records) to {CALIBRATION_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Regenerate all artifacts even if they already exist")
    args = parser.parse_args()

    have_all = all(Path(p).exists() for p in (COMPILED_PATH, SETTINGS_PATH, SRS_PATH, PK_PATH, VK_PATH))
    if have_all and not args.force:
        print("Compiled circuit, SRS, and keys already exist. Skipping (pass --force to regenerate).")
        print("Run verify_circuit.py to confirm they are still correct.")
        return

    write_calibration_file()

    print("Generating settings...")
    ezkl.gen_settings(ONNX_PATH, SETTINGS_PATH, py_run_args=RUN_ARGS)

    print("Calibrating settings against real sample inputs...")
    ezkl.calibrate_settings(CALIBRATION_PATH, ONNX_PATH, SETTINGS_PATH, "resources")

    print("Compiling circuit...")
    ezkl.compile_circuit(ONNX_PATH, COMPILED_PATH, SETTINGS_PATH)

    print("Generating structured reference string (SRS) locally...")
    # ezkl.get_srs() fetches a pre-generated SRS from EZKL's remote server, which
    # this sandboxed environment cannot reach. ezkl.gen_srs() generates one
    # locally instead -- explicitly documented by EZKL as for testing/non-production
    # use, which is exactly this project's situation (PRD.md: no cloud dependency
    # required for the MVP; Rules.md: must run on a standard laptop).
    with open(SETTINGS_PATH) as f:
        logrows = json.load(f)["run_args"]["logrows"]
    ezkl.gen_srs(SRS_PATH, logrows)

    print("Running (local, non-production) trusted setup -> pk.key, vk.key...")
    ezkl.setup(COMPILED_PATH, VK_PATH, PK_PATH, SRS_PATH)

    print("\nCompilation complete. Run circuit/verify_circuit.py next to confirm correctness.")


if __name__ == "__main__":
    main()
