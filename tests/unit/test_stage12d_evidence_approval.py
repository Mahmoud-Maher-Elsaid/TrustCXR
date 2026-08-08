from __future__ import annotations

import json
from pathlib import Path


def test_readiness_does_not_require_invented_other_label() -> None:
    root = Path(__file__).resolve().parents[2]
    config = json.loads(
        (root / "configs/quality/stage12d_annotation_cohort_readiness.json").read_text()
    )
    minimums = config["manifests"]["expanded_view"]["minimum_label_counts"]
    assert minimums == {"UNKNOWN": 1}


def test_input_rejection_application_preserves_six_class_contract() -> None:
    from scripts.quality.validate_stage12d_manual_annotations import REJECTION_CLASSES

    assert REJECTION_CLASSES == [
        "CORRUPT_INPUT",
        "UNSUPPORTED_FORMAT",
        "NON_CHEST_INPUT",
        "INCOMPLETE_ANATOMY",
        "INADEQUATE_QUALITY",
        "UNSUPPORTED_VIEW",
    ]
