"""Phase 3 proof-generation smoke runner with a hard per-record timeout.

This runner deliberately executes the client as a separate process. A stuck
EZKL/Halo2 proving operation therefore cannot hang the test harness forever.
It reports only timing/status/public output; it never prints raw records.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

CLIENT_DIR = Path(__file__).resolve().parent
SAMPLES_DIR = CLIENT_DIR / "sample_records"
CLIENT_SCRIPT = CLIENT_DIR / "infer_and_prove.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    samples = sorted(SAMPLES_DIR.glob("*.json"))
    if not samples:
        print("No sample records found.")
        return 1

    failures = 0
    print(f"Running {len(samples)} Phase 3 sample(s), timeout={args.timeout:.0f}s each")

    for sample in samples:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [sys.executable, str(CLIENT_SCRIPT), str(sample)],
                cwd=CLIENT_DIR.parent,
                capture_output=True,
                text=True,
                timeout=args.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            elapsed = time.monotonic() - started
            print(f"[{sample.stem}] TIMEOUT after {elapsed:.2f}s")
            failures += 1
            continue

        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            print(f"[{sample.stem}] FAILED after {elapsed:.2f}s")
            if completed.stderr:
                print(completed.stderr.strip())
            failures += 1
            continue

        try:
            result = json.loads(completed.stdout)
            if set(result) != {"proof", "public_output"}:
                raise ValueError("unexpected client output keys")
            print(
                f"[{sample.stem}] OK in {elapsed:.2f}s "
                f"public_output={result['public_output']}"
            )
        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[{sample.stem}] FAILED after {elapsed:.2f}s: {exc}")
            failures += 1

    print(
        f"\nResult: {len(samples) - failures}/{len(samples)} sample(s) "
        "completed successfully."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
