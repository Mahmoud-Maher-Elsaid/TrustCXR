from __future__ import annotations

import json
from pathlib import Path

from scripts.verification.run_stage19a_textual_anatomical_verifier_data_readiness import audit

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/verification/stage19a_textual_anatomical_verifier_data_readiness.json"


def test_stage19a_textual_verification_is_exact_and_nonsemantic() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["textual_readiness"] == "READY_EXACT_DETERMINISTIC_TEMPLATES_AND_PROVENANCE"
    assert result["verifier_statuses"] == [
        "VERIFIED",
        "PARTIALLY_VERIFIED",
        "UNVERIFIED",
        "CONTRADICTED",
        "NOT_APPLICABLE",
        "WITHHELD_INSUFFICIENT_EVIDENCE",
    ]


def test_stage19a_preserves_anatomical_and_identity_limits() -> None:
    result = audit(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["anatomical_readiness"]["positive_localization_support"] == (
        "WITHHELD_INSUFFICIENT_EVIDENCE"
    )
    assert (
        "STAGE10_IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY"
        in result["anatomical_readiness"]["proxy"]
    )
    assert not result["localization_absence_may_contradict_classifier"]
    assert result["identity_policy"]["exact_governed_record_or_study_join_required"]
    assert result["locked_test_records_accessed"] == 0
