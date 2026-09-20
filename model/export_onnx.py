"""
Phase 2 — ONNX Export

Loads the trained logistic regression model (selected in Phase 1) and exports
it to ONNX, the intermediate format EZKL requires to compile a zero-knowledge
circuit.

Design decision (see chat): the exported graph bakes the threshold comparison
in, rather than exporting the raw probability. Architecture.md specifies the
public output should be a threshold CLAIM (e.g. "risk_below_threshold: true"),
not the risk score itself — if we exported the raw probability as EZKL's
public output, a verifier would learn the applicant's exact risk score, which
is more than the claim is supposed to reveal. Instead the graph computes the
probability, compares it to a fixed threshold, and outputs only a 0/1 boolean.
The probability itself stays inside the proof as a private intermediate value.
This is also why Phases.md's Phase 7 calls out testing the threshold
comparison at boundary values under quantization — that concern only exists
if the comparison happens inside the circuit, confirming this is the intended
design, not a new one introduced here.

Input shape: a single record, already preprocessed into the 61-dimensional
feature vector described by model/artifacts/preprocessing.json. This script
does NOT do raw-record preprocessing — that stays the client's job in Phase 3,
so the ONNX graph itself is exactly the model plus the threshold check.

Output: circuit/model.onnx
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CIRCUIT_DIR = Path(__file__).parent.parent / "circuit"
CIRCUIT_DIR.mkdir(exist_ok=True)

# The risk-probability cutoff below which an applicant is claimed "safe".
# Matches the 0.5 decision boundary used for classification in Phase 1, so
# the claim is consistent with how the model's accuracy was evaluated.
RISK_THRESHOLD = 0.5


class LogisticRegressionWithClaim(nn.Module):
    """Wraps the trained logistic regression model with the public claim
    logic: outputs 1.0 if risk probability < RISK_THRESHOLD (safe), else 0.0.
    The intermediate probability is never exposed as a graph output."""

    def __init__(self, in_dim, threshold):
        super().__init__()
        self.linear = nn.Linear(in_dim, 1)
        self.register_buffer("threshold", torch.tensor(threshold, dtype=torch.float32))

    def forward(self, x):
        prob = torch.sigmoid(self.linear(x))
        claim = (prob < self.threshold).float()
        return claim


def main():
    with open(ARTIFACTS_DIR / "preprocessing.json") as f:
        preprocessing = json.load(f)
    input_dim = preprocessing["input_dim"]
    print(f"Loading logistic regression model, input_dim={input_dim}, threshold={RISK_THRESHOLD}")

    model = LogisticRegressionWithClaim(input_dim, RISK_THRESHOLD)
    state_dict = torch.load(ARTIFACTS_DIR / "logreg.pt", weights_only=True)
    model.load_state_dict(state_dict, strict=False)  # threshold buffer isn't in the checkpoint, that's fine
    model.eval()

    dummy_input = torch.randn(1, input_dim, dtype=torch.float32)

    onnx_path = CIRCUIT_DIR / "model.onnx"
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["claim"],
        dynamic_axes=None,  # fixed batch size of 1 — EZKL expects static shapes
        opset_version=13,
    )
    print(f"Wrote {onnx_path}")

    # Sanity check: does the ONNX graph's CLAIM match what the original
    # PyTorch model (probability + threshold, computed separately) says, on
    # real held-out records? This catches an export bug before compiling a
    # circuit from it. verify_circuit.py checks this more thoroughly later,
    # including at threshold boundary values.
    import onnxruntime as ort

    raw_model = nn.Sequential(nn.Linear(input_dim, 1))
    raw_model.load_state_dict({"0.weight": state_dict["linear.weight"], "0.bias": state_dict["linear.bias"]})
    raw_model.eval()

    X_test = np.load(ARTIFACTS_DIR / "X_test.npy")
    sample = X_test[:20].astype(np.float32)

    with torch.no_grad():
        torch_prob = torch.sigmoid(raw_model(torch.tensor(sample))).numpy().flatten()
        torch_claim = (torch_prob < RISK_THRESHOLD).astype(np.float32)

    sess = ort.InferenceSession(str(onnx_path))
    # The exported graph has a fixed batch size of 1 (EZKL expects static
    # shapes), so verify one record at a time rather than as one batch.
    onnx_claim = np.array([
        sess.run(None, {"input": sample[i:i + 1]})[0].flatten()[0]
        for i in range(sample.shape[0])
    ])

    mismatches = (torch_claim != onnx_claim).sum()
    print(f"PyTorch vs ONNX claim mismatches on 20 test records: {mismatches}")
    assert mismatches == 0, "ONNX export's claim does not match the PyTorch model's claim!"
    print("ONNX export verified: claim output matches PyTorch model exactly.")


if __name__ == "__main__":
    main()

