from pathlib import Path

from scripts.release.run_stage27a_paper_final_release_audit import audit

ROOT = Path(__file__).resolve().parents[2]


def test_stage27a_release_audit_contract() -> None:
    result = audit(ROOT)
    assert result["status"] == "PASSED_PAPER_FINAL_RELEASE_AUDIT"
    assert result["accepted_checkpoint_count"] == 7
    assert result["blocking_final_release_issues"] == []
    assert result["mandatory_remaining"] == ["STAGE_27B_FINAL_RELEASE_CLOSURE"]
    assert result["scientific_execution_performed"] is False
    assert result["language_model_used"] is False


def test_final_claims_keep_extensions_and_clinical_claims_prohibited() -> None:
    claims = (ROOT / "docs/release/FINAL_CLAIMS_MATRIX.md").read_text(encoding="utf-8")
    for text in (
        "NO CLINICAL DIAGNOSIS",
        "NO RELIABLE POSITIVE LESION LOCALIZATION",
        "EXTERNAL_VALIDATION_NOT_PERFORMED",
        "Grad-CAM",
        "grounded LLM",
    ):
        assert text in claims
