from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.quality.run_stage12d_remaining_candidate_discovery import (  # noqa: E402
    FIELDS,
    contact_sheet,
    scan_chexpert,
    scan_nih,
    scan_rsna,
)


def search(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["automatic_approval_permitted"],
        config["synthetic_examples_permitted"],
        config["deliberate_corruption_permitted"],
        config["complete_model_training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited):
        raise RuntimeError("Stage 12G safety contract changed.")
    upstream = json.loads((root / config["stage12f_evidence"]).read_text(encoding="utf-8"))
    if upstream.get("status") != "PASSED_PARTIAL_SCOPE_EVIDENCE_FREEZE":
        raise RuntimeError("Stage 12G requires the completed Stage 12F gate.")
    if upstream.get("unsupported_slot_count") != 9 or upstream.get("missing_view_classes") != [
        "OTHER"
    ]:
        raise RuntimeError("Stage 12G unresolved evidence scope changed.")
    missing = {tuple(value.split("/", 1)) for value in upstream["unsupported_slots"]}
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str, str]] = set()
    coverage = {
        "chexpert_small": scan_chexpert(
            root,
            missing,
            rows,
            seen,
            pixel_skip_train=config["chexpert_train_pixel_skip"],
            pixel_budget_train=config["chexpert_train_pixel_budget"],
        ),
        "nih_chestxray14": scan_nih(root, missing),
        "rsna_pneumonia": scan_rsna(
            root,
            missing,
            rows,
            seen,
            mapping_skip=config["rsna_mapping_skip"],
            header_budget=config["rsna_header_budget"],
        ),
    }
    output_root = root / config["output_root"]
    output_root.mkdir(parents=True, exist_ok=True)
    review_path = output_root / "stage12g_candidate_review.csv"
    sheet_path = output_root / "stage12g_candidate_contact_sheet.png"
    summary_path = output_root / "stage12g_search_summary.json"
    if any(path.exists() for path in (review_path, sheet_path, summary_path)):
        raise RuntimeError("Stage 12G output exists; preserve it before rerun.")
    with review_path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    contact_sheet(rows, sheet_path)
    found_slots = sorted({f"{row['split']}/{row['proposed_rejection_class']}" for row in rows})
    summary = {
        "stage": "12G",
        "status": "COMPLETED_ADDITIONAL_DEVELOPMENT_EVIDENCE_SEARCH",
        "protocol_version": config["protocol_version"],
        "candidate_count": len(rows),
        "candidate_slots": found_slots,
        "other_view_candidates_require_separate_positive_metadata_adjudication": True,
        "coverage": coverage,
        "labels_approved": 0,
        "unsupported_slots_promoted": 0,
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "frozen_results_modified": False,
        "review_csv": str(review_path),
        "contact_sheet": str(sheet_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 12G development evidence search.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = search(config, root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
