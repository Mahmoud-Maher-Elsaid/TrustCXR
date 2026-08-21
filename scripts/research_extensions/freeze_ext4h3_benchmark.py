"""Materialize and validate the frozen EXT-4H.3 benchmark locally.

This command performs no model loading or inference. It writes only the
deterministic evidence, semantic plans, requests, manifests, and hashes.
"""

from __future__ import annotations

import json
from pathlib import Path

from trustcxr.grounded_llm.ext4h3_benchmark import (
    BENCHMARK_ID,
    benchmark_sha256,
    build_ext4h3_cases,
    canonical_benchmark_payload,
    validate_ext4h3_design,
)

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = (
    ROOT
    / "configs/research_extensions/ext4h/ext4h_fresh_development_benchmark_materialized_v1.json"
)


def main() -> int:
    cases = build_ext4h3_cases()
    validate_ext4h3_design(cases)
    payload = canonical_benchmark_payload(cases)
    payload["benchmark_id"] = BENCHMARK_ID
    payload["benchmark_sha256"] = benchmark_sha256(cases)
    payload["model_load_calls"] = 0
    payload["model_forward_calls"] = 0
    payload["model_generate_calls"] = 0
    payload["historical_benchmark_accessed"] = 0
    payload["frozen_final_cases_accessed"] = 0
    payload["locked_test_accessed"] = False
    FIXTURE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "benchmark_id": BENCHMARK_ID,
                "case_count": len(cases),
                "benchmark_sha256": payload["benchmark_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
