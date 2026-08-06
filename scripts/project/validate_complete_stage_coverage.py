from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ALLOWED_STATUSES = {
    "COMPLETED_AND_VERIFIED",
    "PARTIALLY_COMPLETED",
    "CURRENTLY_IN_PROGRESS",
    "READY_TO_PREPARE",
    "READY_TO_RUN",
    "WAITING_FOR_UPSTREAM_GATE",
    "WAITING_FOR_DATA",
    "WAITING_FOR_DATA_ACCESS",
    "WAITING_FOR_LICENSE",
    "WAITING_FOR_ANNOTATIONS",
    "WAITING_FOR_RESULTS",
    "WAITING_FOR_HARDWARE",
    "NOT_YET_IMPLEMENTED",
    "SCIENTIFICALLY_UNDEFINED",
    "BLOCKED",
}
REQUIRED_ACTUAL = {
    "0",
    "1",
    "2",
    "3",
    "4",
    "4.1",
    "4.2",
    "4.3",
    "4.4",
    "5",
    "6",
    "7B",
    "7C",
    "7D",
    "7E",
    "8A",
    "8B",
    "8C",
    "8D",
    "8E",
    "9A",
    "9B",
    "9C",
}
REQUIRED_SPEC_TERMS = {
    "Current status",
    "Upstream gate",
    "Downstream gate",
    "Success criteria",
    "Failure criteria",
    "Test policy",
    "Identity and leakage checks",
    "Long-running command",
}


def validate(root: Path) -> dict[str, object]:
    registry_path = root / "docs/execution/complete_stage_registry.json"
    payload = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = payload["original_roadmap_stages"]
    numbers = [row["original_roadmap_stage"] for row in rows]
    errors: list[str] = []
    if numbers != list(range(25)):
        errors.append(f"Original stages must be exactly 0..24 once; observed {numbers}")
    duplicates = [key for key, count in Counter(numbers).items() if count != 1]
    if duplicates:
        errors.append(f"Duplicate or missing original-stage mappings: {duplicates}")
    actual = set(payload["actual_repository_stages"])
    if actual != REQUIRED_ACTUAL:
        missing_actual = sorted(REQUIRED_ACTUAL - actual)
        extra_actual = sorted(actual - REQUIRED_ACTUAL)
        errors.append(
            f"Actual-stage registry mismatch: missing={missing_actual}, extra={extra_actual}"
        )
    for row in rows:
        stage = row["original_roadmap_stage"]
        if row.get("current_status") not in ALLOWED_STATUSES:
            errors.append(f"Stage {stage} has invalid status")
        for field in ("upstream_gate", "downstream_gate", "test_set_policy", "leakage_policy"):
            if not str(row.get(field, "")).strip():
                errors.append(f"Stage {stage} lacks {field}")
        spec = root / row["specification"]
        if not spec.is_file():
            errors.append(f"Stage {stage} specification missing: {spec}")
            continue
        text = spec.read_text(encoding="utf-8")
        for term in REQUIRED_SPEC_TERMS:
            if term not in text:
                errors.append(f"Stage {stage} specification lacks {term}")
        if "TODO" in text or "TBD" in text:
            errors.append(f"Stage {stage} contains prohibited placeholder wording")
        if row["current_status"] == "COMPLETED_AND_VERIFIED" and "Evidence commit" not in text:
            errors.append(f"Stage {stage} is complete without evidence")
        if stage > 9 and row["current_status"] == "READY_TO_RUN":
            errors.append(
                f"Future Stage {stage} is marked runnable without a registered real launcher"
            )
    if errors:
        raise RuntimeError("Complete-stage coverage validation failed:\n- " + "\n- ".join(errors))
    counts = Counter(row["current_status"] for row in rows)
    return {
        "status": "PASSED",
        "original_stages_expected": 25,
        "original_stages_documented": len(rows),
        "actual_repository_stages_discovered": len(REQUIRED_ACTUAL),
        "actual_repository_stages_mapped": len(actual),
        "missing_stages": 0,
        "duplicate_unresolved_mappings": 0,
        "status_counts": dict(sorted(counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    print(json.dumps(validate(args.project_root.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
