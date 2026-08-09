from __future__ import annotations

import json
from pathlib import Path

from scripts.verification.run_stage19b_textual_anatomical_verifier_contract import (
    freeze_contract,
)

from trustcxr.verification.deterministic_verifier import verify_anatomical, verify_textual

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/verification/stage19b_textual_anatomical_verifier_contract.json"
REPORT_CONTRACT = (
    ROOT / "reports/stage18/stage18b_deterministic_grounded_report_contract_summary.json"
)


def statement() -> dict:
    return {
        "evidence_type": "model_identified_view_ap_pa_or_lateral",
        "grounding_status": "DIRECT_STRUCTURED",
        "template_id": "VIEW_IDENTIFIED",
        "template_parameters": {"view": "PA"},
        "source_stage": "5",
        "source_version": "synthetic-v1",
        "evidence_code": "VIEW_MODEL_OUTPUT",
        "structured_source_field": "view/prediction",
    }


def test_textual_status_rules_are_deterministic() -> None:
    contract = json.loads(REPORT_CONTRACT.read_text(encoding="utf-8"))
    assert (
        verify_textual(
            statement(),
            "Model-identified view: PA.",
            contract,
            evidence_available=True,
            exact_identity=True,
        )
        == "VERIFIED"
    )
    assert (
        verify_textual(
            statement(),
            "Altered view wording.",
            contract,
            evidence_available=True,
            exact_identity=True,
        )
        == "UNVERIFIED"
    )
    assert (
        verify_textual(
            statement(),
            "Model-identified view: PA.",
            contract,
            evidence_available=True,
            exact_identity=True,
            explicit_accepted_conflict=True,
        )
        == "CONTRADICTED"
    )


def test_textual_missing_or_malformed_grounding_is_unverified() -> None:
    contract = json.loads(REPORT_CONTRACT.read_text(encoding="utf-8"))
    missing = statement()
    del missing["source_version"]
    malformed = statement()
    malformed["evidence_code"] = "NOT_A_GOVERNED_CODE"
    for candidate in (missing, malformed):
        assert (
            verify_textual(
                candidate,
                "Model-identified view: PA.",
                contract,
                evidence_available=True,
                exact_identity=True,
            )
            == "UNVERIFIED"
        )


def test_textual_missing_evidence_identity_mismatch_and_forbidden_claim_fail_closed() -> None:
    contract = json.loads(REPORT_CONTRACT.read_text(encoding="utf-8"))
    assert (
        verify_textual(
            statement(),
            "Model-identified view: PA.",
            contract,
            evidence_available=False,
            exact_identity=True,
        )
        == "UNVERIFIED"
    )
    assert (
        verify_textual(
            statement(),
            "Model-identified view: PA.",
            contract,
            evidence_available=True,
            exact_identity=False,
        )
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )
    forbidden = statement()
    forbidden["evidence_type"] = "reliable_positive_localization_claim"
    assert (
        verify_textual(
            forbidden,
            "A lesion is localized.",
            contract,
            evidence_available=True,
            exact_identity=True,
        )
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )


def test_anatomical_proxy_absence_and_identity_fail_closed() -> None:
    assert (
        verify_anatomical(
            "STAGE10_IMAGE_GEOMETRY_AND_THORACIC_LOCATION_PROXY_ONLY",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        )
        == "PARTIALLY_VERIFIED"
    )
    assert (
        verify_anatomical(
            "RELIABLE_POSITIVE_LESION_LOCALIZATION",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        )
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )
    assert (
        verify_anatomical(
            "FINDING_LATERALITY_FROM_LOCALIZATION",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        )
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )
    assert (
        verify_anatomical(
            "IMAGE_BOUNDS_AND_EXACT_RECORD_IDENTITY",
            exact_identity=True,
            within_governed_scope=False,
            evidence_available=True,
        )
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )
    assert (
        verify_anatomical(
            "severity",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
        )
        == "NOT_APPLICABLE"
    )


def test_verification_is_repeatable() -> None:
    contract = json.loads(REPORT_CONTRACT.read_text(encoding="utf-8"))
    outcomes = {
        verify_textual(
            statement(),
            "Model-identified view: PA.",
            contract,
            evidence_available=True,
            exact_identity=True,
        )
        for _ in range(10)
    }
    assert outcomes == {"VERIFIED"}
    assert (
        verify_anatomical(
            "IMAGE_BOUNDS_AND_EXACT_RECORD_IDENTITY",
            exact_identity=False,
            within_governed_scope=True,
            evidence_available=True,
        )
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )
    assert (
        verify_anatomical(
            "IMAGE_BOUNDS_AND_EXACT_RECORD_IDENTITY",
            exact_identity=True,
            within_governed_scope=True,
            evidence_available=True,
            absence_only=True,
        )
        == "WITHHELD_INSUFFICIENT_EVIDENCE"
    )


def test_stage19b_contract_preserves_statuses_and_safety() -> None:
    result = freeze_contract(json.loads(CONFIG.read_text(encoding="utf-8")), ROOT)
    assert result["anatomical_policy"]["stage11_maximum_support"] == "PARTIALLY_SUPPORTED"
    assert not result["identity_policy"]["heuristic_identity_matching_permitted"]
    assert result["locked_test_records_accessed"] == 0
    assert result["patient_identifiers_used"] == 0
