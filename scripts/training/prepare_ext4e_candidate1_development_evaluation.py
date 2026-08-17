"""Prepare and aggregate the EXT-4E1 six-case development evaluation offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trustcxr.grounded_llm.development_evaluation import (  # noqa: E402
    aggregate_evidence,
    build_evaluation_plan,
)

CASES = ROOT / "tests/fixtures/ext4d_benchmark_cases.json"
CONFIG = ROOT / "configs/research_extensions/ext4e2d_candidate1_dev_smoke.json"
HISTORICAL_ROOT = ROOT / "artifacts/research_extensions/ext4e2_candidate1/dev_case_smoke"


def main() -> int:
    plan = build_evaluation_plan(CASES, CONFIG, HISTORICAL_ROOT)
    if len(sys.argv) == 1:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if sys.argv[1] != "--aggregate":
        raise SystemExit("Only offline plan preparation or --aggregate is supported.")
    paths = {
        case_id: ROOT
        / "artifacts/research_extensions/ext4e2_candidate1/development_evaluation"
        / case_id
        for case_id in plan["development_case_ids"]
    }
    paths["dev_supported"] = HISTORICAL_ROOT / "20260817T091916Z"
    aggregate = aggregate_evidence(CASES, paths)
    print(json.dumps(aggregate, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
