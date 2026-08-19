"""Import frozen EXT-4F.6 blinded rubric results; never runs a model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from trustcxr.grounded_llm.ext4f5_benchmark import (  # noqa: E402
    build_development_cases,
    final_development_decision,
    import_review_results,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review-json", type=Path, required=True)
    parser.add_argument("--automatic-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    review = json.loads(args.review_json.read_text(encoding="utf-8"))
    automatic = json.loads(args.automatic_report.read_text(encoding="utf-8"))
    cases = build_development_cases()
    imported = import_review_results(automatic["benchmark_sha256"], cases, review)
    result = final_development_decision(automatic["aggregate"], imported)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
