from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from scripts.ui.run_stage22e_synthetic_ui_acceptance_decision import decide_acceptance

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/ui/stage22e_synthetic_ui_acceptance_decision.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_stage22e_accepts_exact_frozen_synthetic_runtime_evidence() -> None:
    result = decide_acceptance(CONFIG, ROOT)
    assert result["status"] == ("SYNTHETICALLY_VALIDATED_LOCAL_RESEARCH_UI_EXPERT_REVIEW_REQUIRED")
    assert result["stage22b_contract_fingerprint"] == (
        "f413fbb969261cf9b9b51eb0870aba4de9555483acff42212c04639f293b3a2f"
    )
    assert result["stage22c_summary_sha256"] == (
        "043c021240d097e6c3547191c24800c35646762871409cb86009ad3e953182f9"
    )
    assert result["accepted_evidence_scope"] == "SYNTHETIC_NON_PATIENT_ONLY"


def test_stage22e_preserves_research_banner_and_minimal_technology() -> None:
    assert CONFIG["research_banner"] == [
        "RESEARCH USE ONLY",
        "NOT A MEDICAL DIAGNOSIS",
        "EXPERT REVIEW REQUIRED",
    ]
    assert set(CONFIG["technology"]) == {
        "STATIC_HTML",
        "CSS",
        "VANILLA_JAVASCRIPT",
        "EXISTING_FASTAPI_STARLETTE",
        "EXISTING_V1_API",
        "NO_FRONTEND_FRAMEWORK",
        "NO_NPM_NODE",
        "NO_CDN",
        "NO_CLOUD_ASSETS",
        "NO_EXTERNAL_ANALYTICS",
    }


def test_stage22e_freezes_image_formats_and_overlay_holds() -> None:
    assert CONFIG["image_formats"] == {
        "PNG": "ACCEPTED_GOVERNED_LOCAL_RESEARCH_VIEWER_CONTRACT",
        "JPEG": "ACCEPTED_GOVERNED_LOCAL_RESEARCH_VIEWER_CONTRACT",
        "DICOM": "WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT",
        "TENSOR_NPZ": "INTERNAL_ONLY_NOT_PUBLIC_BROWSER_DISPLAY",
    }
    assert set(CONFIG["overlays"].values()) == {
        "WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION"
    }
    assert "RELIABLE_POSITIVE_LESION_LOCALIZATION" in CONFIG["prohibited_overlay_inferences"]
    assert "NEGATION_FROM_LOCALIZATION_ABSENCE" in CONFIG["prohibited_overlay_inferences"]


def test_stage22e_preserves_reliability_verifier_and_decision_semantics() -> None:
    assert CONFIG["stage16_contract"]["allowed_term"] == "PREDICTIVE UNCERTAINTY"
    assert CONFIG["stage16_contract"]["prohibited_term"] == "EPISTEMIC UNCERTAINTY"
    assert CONFIG["stage16_contract"]["ood"] == "WITHHELD"
    assert CONFIG["stage19_statuses"] == [
        "VERIFIED",
        "PARTIALLY_VERIFIED",
        "UNVERIFIED",
        "CONTRADICTED",
        "NOT_APPLICABLE",
        "WITHHELD_INSUFFICIENT_EVIDENCE",
    ]
    assert CONFIG["stage20_precedence"] == [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]
    assert CONFIG["stage20_accept_meaning"] == "NOT_CLINICAL_APPROVAL"


def test_stage22e_preserves_privacy_security_and_failure_semantics() -> None:
    assert CONFIG["failure_semantics"] == {
        "DEFER": "SAFETY_OR_EVIDENCE_LIMITATION",
        "FAILED_SANITIZED": "TECHNICAL_INFRASTRUCTURE_FAILURE",
    }
    assert {
        "NO_EXTERNAL_REQUESTS",
        "NO_BROWSER_PERSISTENCE",
        "NO_PATIENT_IDENTIFIERS",
        "NO_PHI",
        "NO_INTERNAL_PATHS",
        "NO_STACK_TRACES",
        "DETERMINISTIC_CLEANUP",
        "BOUNDED_SERVER_TERMINATED",
    } <= set(CONFIG["browser_privacy_guarantees"])
    assert {
        "INJECTION_SAFETY_PASSED",
        "ACCESSIBILITY_CONTRACT_PASSED",
        "NON_COLOR_DISTINCTIONS",
        "MODEL_SIGNAL_DISTINCT_FROM_VERIFIED_EVIDENCE",
    } <= set(CONFIG["security_accessibility_evidence"])


@pytest.mark.parametrize(
    "key",
    [
        "real_image_display_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "patient_processing_permitted",
        "locked_test_access_permitted",
        "stage8_overlay_activation_permitted",
        "stage10_overlay_activation_permitted",
        "dicom_support_activation_permitted",
        "persistent_service_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage22e_rejects_scope_expansion(key: str) -> None:
    changed = copy.deepcopy(CONFIG)
    changed[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        decide_acceptance(changed, ROOT)


def test_stage22e_acceptance_validation_has_no_output_side_effect() -> None:
    output = ROOT / "reports/stage22/stage22e_synthetic_ui_acceptance_decision_summary.json"
    before = output.read_bytes()
    decide_acceptance(CONFIG, ROOT)
    assert output.read_bytes() == before


def test_stage22e_next_stage_has_no_runtime_or_llm_authorization() -> None:
    assert CONFIG["next_canonical_stage"] == "23A_DICOM_INTEROPERABILITY_DATA_READINESS"
    for key in (
        "next_stage_authorizes_real_image_display",
        "next_stage_authorizes_real_model_loading",
        "next_stage_authorizes_bounded_real_inference",
        "next_stage_authorizes_gpu_residency_profiling",
        "next_stage_authorizes_patient_processing",
        "next_stage_authorizes_language_model_work",
    ):
        assert not CONFIG[key]
    assert CONFIG["currently_planned_llm_authorized_gate"] is None
