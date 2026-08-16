from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from trustcxr.grounded_llm.contracts import (
    EvidenceEnvelope,
    EvidenceItem,
    EvidenceStatus,
    EvidenceValue,
    SourceStage,
    ValueKind,
    build_synthetic_case,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/research_extensions/ext4a_grounded_llm_governance.json"


def test_schema_version_and_supported_synthetic_case() -> None:
    case = build_synthetic_case("supported")
    assert case.schema_id == "EXT4_GROUNDING_SCHEMA"
    assert case.schema_version == "1"
    assert case.evidence_items[0].status is EvidenceStatus.SUPPORTED


def test_unknown_status_stage_and_patient_or_raw_fields_fail_closed() -> None:
    with pytest.raises(ValidationError):
        EvidenceEnvelope.model_validate(
            {**build_synthetic_case("supported").model_dump(), "schema_version": "99"}
        )
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="x",
            source_stage="STAGE_99",
            evidence_type="CLASSIFIER_SIGNAL",
            status="UNKNOWN",
        )
    with pytest.raises(ValidationError):
        EvidenceEnvelope.model_validate(
            {**build_synthetic_case("supported").model_dump(), "patient_id": "not_allowed"}
        )
    with pytest.raises(ValidationError):
        EvidenceEnvelope.model_validate(
            {**build_synthetic_case("supported").model_dump(), "raw_image_path": "x.png"}
        )


def test_supported_requires_provenance_and_withheld_is_not_support() -> None:
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="x",
            source_stage=SourceStage.STAGE_9,
            evidence_type="CLASSIFIER_SIGNAL",
            status=EvidenceStatus.SUPPORTED,
            value=EvidenceValue(kind=ValueKind.NUMBER, number=0.5),
        )
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="x",
            source_stage=SourceStage.STAGE_9,
            evidence_type="LOCALIZATION",
            status=EvidenceStatus.WITHHELD,
            value=EvidenceValue(kind=ValueKind.TEXT, text="no lesion"),
            withheld_reason="LOCALIZATION_INTEGRATION_WITHHELD",
        )
    with pytest.raises(ValidationError):
        EvidenceItem(
            evidence_id="x",
            source_stage=SourceStage.STAGE_9,
            evidence_type="CLASSIFIER_SIGNAL",
            status=EvidenceStatus.SUPPORTED,
            value=EvidenceValue(kind=ValueKind.NUMBER, number=0.5),
            provenance=None,
            claim_permissions=("definitive_diagnosis",),
        )
    case = build_synthetic_case("withheld")
    assert case.withheld_evidence[0].status is EvidenceStatus.WITHHELD
    assert case.withheld_evidence[0].category == "LOCALIZATION"


def test_defer_uncertainty_conflict_and_missing_cases_preserve_state() -> None:
    defer_case = build_synthetic_case("defer")
    assert defer_case.decision_state.defer_active is True
    assert defer_case.decision_state.decision == "DEFER"
    uncertain_case = build_synthetic_case("uncertainty")
    assert uncertain_case.uncertainty.status == "AVAILABLE"
    assert uncertain_case.uncertainty.value == 0.91
    conflict_case = build_synthetic_case("conflict")
    assert conflict_case.evidence_items[0].status is EvidenceStatus.CONTRADICTED
    missing_case = build_synthetic_case("missing")
    assert missing_case.evidence_items[0].status is EvidenceStatus.NOT_AVAILABLE


def test_non_finite_values_and_incompatible_defer_fail_closed() -> None:
    with pytest.raises(ValidationError):
        EvidenceValue(kind=ValueKind.NUMBER, number=float("nan"))
    with pytest.raises(ValidationError):
        EvidenceEnvelope.model_validate(
            {
                **build_synthetic_case("supported").model_dump(),
                "decision_state": {
                    "decision": "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
                    "defer_active": True,
                },
            }
        )


def test_config_prohibits_provider_and_implementation() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["scope"]["implementation_started"] is False
    assert config["provider_governance"]["provider_selection"] == "NOT_AUTHORIZED"
    assert config["scope"]["external_api_called"] is False
    assert config["localization_boundary"]["integration_status"] == (
        "WITHHELD_NOT_ACCEPTED_FOR_TRUSTED_INTEGRATION"
    )
