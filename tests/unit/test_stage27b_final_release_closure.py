from pathlib import Path

from scripts.release.run_stage27b_final_release_closure import close

ROOT = Path(__file__).resolve().parents[2]


def test_stage27b_closes_core_without_extensions() -> None:
    result = close(ROOT)
    assert result["status"] == "TRUSTCXR_FROZEN_RESEARCH_RELEASE"
    assert result["mandatory_core_stages_remaining"] == 0
    assert result["grad_cam_implemented"] is False
    assert result["true_localization_implemented"] is False
    assert result["language_model_implemented"] is False
    assert result["multimodal_vlm_implemented"] is False
