from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/research_extensions/ext4a_grounded_llm_governance.json"
DOC_PATH = ROOT / "docs/research_extensions/grounded_llm/EXT4A_GROUNDED_LLM_GOVERNANCE.md"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_ext4a_is_governance_only_and_selects_no_provider() -> None:
    config = load_config()
    assert config["status"] == "GOVERNANCE_COMPLETED_IMPLEMENTATION_NOT_STARTED"
    assert config["scope"]["implementation_started"] is False
    assert config["scope"]["provider_selected"] is False
    assert config["scope"]["external_api_called"] is False
    assert config["next_stage"] == "EXT-4B EVIDENCE GROUNDING SCHEMA"


def test_ext4a_forbids_raw_images_identity_and_locked_test() -> None:
    config = load_config()
    forbidden = config["input_field_governance"]["forbidden"]
    for value in (
        "raw CXR image",
        "DICOM file or metadata",
        "patient ID",
        "locked-test membership",
    ):
        assert value in forbidden
    assert config["scope"]["raw_image_input"] == "FORBIDDEN"
    assert config["scope"]["locked_test_accessed"] is False


def test_ext4a_preserves_frozen_core_and_localization_boundaries() -> None:
    config = load_config()
    assert "stage18_deterministic_report" in config["frozen_core_authority"]
    assert "stage19_deterministic_verifier" in config["frozen_core_authority"]
    assert config["localization_boundary"]["integration_status"] == (
        "WITHHELD_NOT_ACCEPTED_FOR_TRUSTED_INTEGRATION"
    )
    assert config["localization_boundary"]["gradcam_status"] == (
        "ATTRIBUTION_ONLY_NOT_LESION_LOCALIZATION"
    )
    assert "DEFER remains DEFER" in config["evidence_state_rules"]


def test_ext4a_claim_taxonomy_abstention_and_baseline_are_governed() -> None:
    config = load_config()
    forbidden = config["forbidden_claim_categories"]
    for value in (
        "definitive_diagnosis",
        "anatomical_or_lesion_localization",
        "DEFER_override",
        "WITHHELD_as_negative",
        "fabricated_provenance",
    ):
        assert value in forbidden
    assert "malformed structured input fails closed" in config["abstention_contract"]
    assert config["deterministic_baseline"]["replacement_allowed"] is False


def test_ext4a_document_matches_non_implementation_boundary() -> None:
    document = DOC_PATH.read_text(encoding="utf-8")
    for phrase in (
        "COMPLETED — GOVERNANCE ONLY",
        "Implementation: `NOT STARTED`",
        "Raw CXR images",
        "EXT-3 is a controlled negative",
        "Stage 18 deterministic reporting remains the frozen baseline",
        "No LLM client, provider, external API",
    ):
        assert phrase in document
    assert "diagnostic evidence" not in document.lower()
