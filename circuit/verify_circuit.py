"""
Phase 2 — Circuit Correctness Verification

Phase 2's "done" condition (Phases.md): "circuit compiles and its output is
proven to match the original model's inference on a sample of test records."

This script closes that out. model.onnx, model.compiled, pk.key, and vk.key
already exist (compiled/set up earlier); this script is what actually proves
correctness end-to-end for a batch of real held-out records:

    for each of N test records:
        1. Run the ORIGINAL PyTorch model (probability + threshold) to get
           the ground-truth claim.
        2. Generate a witness for that record against the compiled circuit.
        3. Generate a real ZK proof from that witness.
        4. Verify the proof against the verification key.
        5. Decode the circuit's public output and compare it to the
           PyTorch claim.

Per Rules.md: never hardcode `valid: true`. Every proof here is a real EZKL
proof, and every verification is a real ezkl.verify() call; if any proof
fails to verify, or any decoded claim disagrees with PyTorch, this script
reports that plainly and does not paper over it.

Boundary-value testing under quantization is explicitly Phase 7 scope
(Phases.md) and is NOT attempted here — this script only checks agreement
on ordinary held-out records, which is what Phase 2 itself requires.
"""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import ezkl

CIRCUIT_DIR = Path(__file__).parent
MODEL_DIR = CIRCUIT_DIR.parent / "model"
ARTIFACTS_DIR = MODEL_DIR / "artifacts"

MODEL_COMPILED = str(CIRCUIT_DIR / "model.compiled")
PK_PATH = str(CIRCUIT_DIR / "pk.key")
VK_PATH = str(CIRCUIT_DIR / "vk.key")
SRS_PATH = str(CIRCUIT_DIR / "kzg.srs")
SETTINGS_PATH = str(CIRCUIT_DIR / "settings.json")

RISK_THRESHOLD = 0.5
N_SAMPLES = 10  # keep small: each proof takes real wall-clock time


class RawLogReg(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.linear(x)


def decode_claim(proof_public_output):
    """EZKL's public output is a felt-encoded fixed-point number. A value's
    'true-ness' here is whatever the circuit encoded for 1.0 vs 0.0 under
    its declared output scale (model_output_scales=[0] means an integer
    representation) -- so decode via ezkl's own felt_to_float using that
    scale, then round, rather than assuming raw felt magnitude."""
    with open(SETTINGS_PATH) as f:
        settings = json.load(f)
    scale = settings["model_output_scales"][0]
    val = ezkl.felt_to_float(proof_public_output, scale)
    return round(val)


def main():
    with open(ARTIFACTS_DIR / "preprocessing.json") as f:
        input_dim = json.load(f)["input_dim"]

    state_dict = torch.load(ARTIFACTS_DIR / "logreg.pt", weights_only=True)
    torch_model = RawLogReg(input_dim)
    torch_model.load_state_dict(state_dict)
    torch_model.eval()

    X_test = np.load(ARTIFACTS_DIR / "X_test.npy").astype(np.float32)
    n = min(N_SAMPLES, X_test.shape[0])
    print(f"Verifying circuit correctness on {n} held-out test records...\n")

    results = []
    for i in range(n):
        record = X_test[i:i + 1]

        with torch.no_grad():
            prob = torch.sigmoid(torch_model(torch.tensor(record))).item()
        pytorch_claim = 1 if prob < RISK_THRESHOLD else 0

        input_path = CIRCUIT_DIR / f"_verify_input_{i}.json"
        witness_path = CIRCUIT_DIR / f"_verify_witness_{i}.json"
        proof_path = CIRCUIT_DIR / f"_verify_proof_{i}.json"
        with open(input_path, "w") as f:
            json.dump({"input_data": record.tolist()}, f)

        t0 = time.time()
        ezkl.gen_witness(
            data=str(input_path),
            model=MODEL_COMPILED,
            output=str(witness_path),
            vk_path=VK_PATH,
            srs_path=SRS_PATH,
        )
        t_witness = time.time() - t0

        t0 = time.time()
        ezkl.prove(
            witness=str(witness_path),
            model=MODEL_COMPILED,
            pk_path=PK_PATH,
            proof_path=str(proof_path),
            srs_path=SRS_PATH,
        )
        t_prove = time.time() - t0

        t0 = time.time()
        is_valid = ezkl.verify(
            proof_path=str(proof_path),
            settings_path=SETTINGS_PATH,
            vk_path=VK_PATH,
            srs_path=SRS_PATH,
        )
        t_verify = time.time() - t0

        with open(proof_path) as f:
            proof_data = json.load(f)
        public_output_felt = proof_data["instances"][0][0]
        circuit_claim = decode_claim(public_output_felt)

        agree = (circuit_claim == pytorch_claim)
        results.append({
            "index": i,
            "pytorch_prob": prob,
            "pytorch_claim": pytorch_claim,
            "circuit_claim": circuit_claim,
            "proof_verified": bool(is_valid),
            "agree": agree,
            "t_witness": t_witness,
            "t_prove": t_prove,
            "t_verify": t_verify,
        })

        status = "OK" if (is_valid and agree) else "MISMATCH"
        print(f"[{i}] prob={prob:.4f} pytorch_claim={pytorch_claim} "
              f"circuit_claim={circuit_claim} proof_valid={is_valid} "
              f"witness={t_witness:.2f}s prove={t_prove:.2f}s verify={t_verify:.3f}s -> {status}")

        for p in (input_path, witness_path, proof_path):
            p.unlink(missing_ok=True)

    n_ok = sum(1 for r in results if r["proof_verified"] and r["agree"])
    n_bad = len(results) - n_ok
    print(f"\n{n_ok}/{len(results)} records: proof verified AND circuit claim matches PyTorch.")

    if n_bad > 0:
        print(f"{n_bad} record(s) FAILED verification or disagreed with PyTorch. Phase 2 is NOT done.")
    else:
        print("All sampled records agree. Circuit correctness confirmed for Phase 2.")

    report_path = CIRCUIT_DIR / "correctness_report.json"
    with open(report_path, "w") as f:
        json.dump({
            "n_samples": len(results),
            "n_agree_and_verified": n_ok,
            "n_failed": n_bad,
            "risk_threshold": RISK_THRESHOLD,
            "results": results,
        }, f, indent=2)
    print(f"Wrote {report_path}")

    assert n_bad == 0, f"{n_bad} record(s) failed proof verification or disagreed with PyTorch"


if __name__ == "__main__":
    main()
