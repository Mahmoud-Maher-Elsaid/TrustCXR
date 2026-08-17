"""Offline gate for the prepared six-case Candidate #1 evaluation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trustcxr.grounded_llm.development_evaluation import (  # noqa: E402
    EXPECTED_CONFIG_SHA256,
    build_evaluation_plan,
    sha256_file,
)


def main() -> int:
    config_path = ROOT / "configs/research_extensions/ext4e_candidate1_development_evaluation.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if (
        sha256_file(ROOT / "configs/research_extensions/ext4d_benchmark.json")
        != EXPECTED_CONFIG_SHA256
    ):
        raise RuntimeError("EXT-4D config hash mismatch.")
    if config["final_cases_accessed"] != 0 or config["locked_test_accessed"]:
        raise RuntimeError("Final or locked-test access is prohibited.")
    plan = build_evaluation_plan(
        ROOT / "tests/fixtures/ext4d_benchmark_cases.json",
        ROOT / "configs/research_extensions/ext4e2d_candidate1_dev_smoke.json",
        ROOT / "artifacts/research_extensions/ext4e2_candidate1/dev_case_smoke",
    )
    if plan["development_case_count"] != 6 or len(plan["remaining_case_ids"]) != 5:
        raise RuntimeError("The six-case development plan is incomplete.")
    print("EXT-4E Candidate #1 six-development-case pre-inference gate: PASS")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
