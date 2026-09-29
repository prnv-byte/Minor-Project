"""Phase 3 — Local Inference + Proof Generation Client.

Takes one raw German Credit applicant record, preprocesses it locally using
model/artifacts/preprocessing.json, and generates a real EZKL proof against
the already-compiled Phase 2 circuit.

Privacy boundary:
    Raw record -> LOCAL preprocessing -> 61-dim vector -> EZKL witness/proof
                                                        |
                                                        +--> proof + public claim

Only proof and public_output are returned by infer_and_prove.
The raw record is never sent to a network endpoint and is never logged.
The temporary EZKL input contains only the transformed numeric vector and is
removed immediately after witness generation.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

CLIENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CLIENT_DIR.parent
MODEL_ARTIFACTS = ROOT_DIR / "model" / "artifacts"
CIRCUIT_DIR = ROOT_DIR / "circuit"

PREPROCESSING_PATH = MODEL_ARTIFACTS / "preprocessing.json"
MODEL_PATH = MODEL_ARTIFACTS / "logreg.pt"
COMPILED_PATH = CIRCUIT_DIR / "model.compiled"
PK_PATH = CIRCUIT_DIR / "pk.key"
SRS_PATH = CIRCUIT_DIR / "kzg.srs"
VK_PATH = CIRCUIT_DIR / "vk.key"
SETTINGS_PATH = CIRCUIT_DIR / "settings.json"

RISK_THRESHOLD = 0.5


class RawLogReg(nn.Module):
    """The Phase 1 logistic-regression model, used only for local sanity checks."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _load_preprocessing() -> dict[str, Any]:
    if not PREPROCESSING_PATH.exists():
        raise FileNotFoundError(f"Missing preprocessing recipe: {PREPROCESSING_PATH}")
    with PREPROCESSING_PATH.open() as f:
        return json.load(f)


def _require_fields(record: Mapping[str, Any], preprocessing: Mapping[str, Any]) -> None:
    required = list(preprocessing["numeric_cols"]) + list(preprocessing["categorical_cols"])
    missing = [name for name in required if name not in record]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")


def preprocess_record(record: Mapping[str, Any]) -> np.ndarray:
    """Convert one raw applicant record into the exact 61-feature model vector.

    Categorical values are encoded as strict one-hot blocks. Exactly one entry
    in every categorical block must be 1. Unknown categories are rejected
    rather than silently producing an all-zero block.
    """
    preprocessing = _load_preprocessing()
    _require_fields(record, preprocessing)

    numeric_cols = preprocessing["numeric_cols"]
    categorical_cols = preprocessing["categorical_cols"]
    means = np.asarray(preprocessing["numeric_mean"], dtype=np.float32)
    stds = np.asarray(preprocessing["numeric_std"], dtype=np.float32)

    numeric_values: list[float] = []
    for col in numeric_cols:
        value = record[col]
        if isinstance(value, bool):
            raise ValueError(f"Numeric field '{col}' cannot be boolean")
        try:
            value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Numeric field '{col}' must be a finite number") from exc
        if not math.isfinite(value):
            raise ValueError(f"Numeric field '{col}' must be finite")
        numeric_values.append(value)

    numeric = (np.asarray(numeric_values, dtype=np.float32) - means) / stds
    blocks: list[np.ndarray] = [numeric]

    for col in categorical_cols:
        value = str(record[col])
        categories = preprocessing["categorical_categories"][col]
        if value not in categories:
            raise ValueError(
                f"Unknown value for '{col}': {value!r}. "
                f"Expected one of: {', '.join(categories)}"
            )
        block = np.zeros(len(categories), dtype=np.float32)
        block[categories.index(value)] = 1.0
        if not (np.count_nonzero(block) == 1 and np.sum(block) == 1.0):
            raise ValueError(f"Invalid one-hot encoding generated for '{col}'")
        blocks.append(block)

    vector = np.concatenate(blocks).astype(np.float32)
    expected_dim = int(preprocessing["input_dim"])
    if vector.shape != (expected_dim,):
        raise ValueError(
            f"Preprocessing produced shape {vector.shape}; expected ({expected_dim},)"
        )
    if not np.isfinite(vector).all():
        raise ValueError("Preprocessing produced a non-finite feature value")

    return vector


def local_claim(feature_vector: np.ndarray) -> tuple[float, int]:
    """Run the original PyTorch model locally as a cheap pre-proof sanity check."""
    preprocessing = _load_preprocessing()
    model = RawLogReg(int(preprocessing["input_dim"]))
    state_dict = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        probability = torch.sigmoid(
            model(torch.from_numpy(feature_vector[None, :]))
        ).item()
    claim = int(probability < RISK_THRESHOLD)
    return probability, claim


def _require_circuit_artifacts() -> None:
    required = [COMPILED_PATH, PK_PATH, VK_PATH, SRS_PATH, SETTINGS_PATH]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Phase 2 proving artifacts: " + ", ".join(missing)
        )


def generate_proof(feature_vector: np.ndarray) -> dict[str, Any]:
    """Generate a real EZKL proof from a preprocessed local feature vector."""
    import ezkl

    _require_circuit_artifacts()

    with tempfile.TemporaryDirectory(prefix="zkml_phase3_") as temp_dir:
        temp = Path(temp_dir)
        input_path = temp / "input.json"
        witness_path = temp / "witness.json"
        proof_path = temp / "proof.json"

        # EZKL consumes JSON input. This contains only the transformed feature
        # vector, never the raw applicant record.
        with input_path.open("w") as f:
            json.dump({"input_data": [feature_vector.tolist()]}, f)

        try:
            ezkl.gen_witness(
                data=str(input_path),
                model=str(COMPILED_PATH),
                output=str(witness_path),
                vk_path=str(VK_PATH),
                srs_path=str(SRS_PATH),
            )
            ezkl.prove(
                witness=str(witness_path),
                model=str(COMPILED_PATH),
                pk_path=str(PK_PATH),
                proof_path=str(proof_path),
                srs_path=str(SRS_PATH),
            )
        except Exception as exc:
            raise RuntimeError(f"EZKL proof generation failed: {exc}") from exc

        with proof_path.open() as f:
            proof_data = json.load(f)

    try:
        public_output = proof_data["instances"][0][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("EZKL proof did not contain the expected public output") from exc

    return {"proof": proof_data, "public_output": public_output}


def decode_public_claim(public_output: str) -> int:
    """Decode EZKL's public 0/1 claim using the circuit's declared output scale."""
    import ezkl

    with SETTINGS_PATH.open() as f:
        settings = json.load(f)
    scale = settings["model_output_scales"][0]
    value = ezkl.felt_to_float(public_output, scale)
    rounded = round(value)
    if rounded not in (0, 1):
        raise RuntimeError(f"Circuit public output decoded to unexpected value: {value}")
    return rounded


def infer_and_prove(record: Mapping[str, Any]) -> dict[str, Any]:
    """Public Phase 3 API: raw record in, proof + public claim out."""
    feature_vector = preprocess_record(record)

    # The exact probability is local only and is never returned.
    _, expected_claim = local_claim(feature_vector)

    result = generate_proof(feature_vector)
    actual_claim = decode_public_claim(result["public_output"])

    if actual_claim != expected_claim:
        raise RuntimeError(
            f"Circuit claim ({actual_claim}) disagrees with local model claim "
            f"({expected_claim})"
        )

    return result


def load_record(path: Path) -> dict[str, Any]:
    with path.open() as f:
        record = json.load(f)
    if not isinstance(record, dict):
        raise ValueError("Sample record must be a JSON object")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a local ZK proof for one raw record"
    )
    parser.add_argument("record", type=Path, help="Path to a raw applicant JSON record")
    args = parser.parse_args()

    record = load_record(args.record)
    result = infer_and_prove(record)

    # CLI output deliberately excludes the raw record and exact local risk score.
    print(json.dumps(result))


if __name__ == "__main__":
    main()
