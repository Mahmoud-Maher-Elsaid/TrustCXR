from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_gradcam_config_tracks_accepted_technical_baseline() -> None:
    config = (ROOT / "configs/explainability/gradcam_stage9.json").read_text(encoding="utf-8")
    assert '"status": "TECHNICAL_GRADCAM_BASELINE_ACCEPTED"' in config
    assert '"method": "gradcam"' in config
    assert '"true_lesion_localization"' in config
    assert '"clinical_diagnosis"' in config
    assert '"ui_gate_required": true' in config


def test_roadmap_keeps_follow_on_stages_unimplemented() -> None:
    roadmap = (
        ROOT / "docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md"
    ).read_text(encoding="utf-8")
    assert "EXT-1A Explainability Foundation:** `COMPLETED`" in roadmap
    assert "EXT-1B Grad-CAM Implementation and Technical Validation:** `COMPLETED`" in roadmap
    assert "Grad-CAM UI integration:** `NOT STARTED`" in roadmap
    assert "Grounded LLM reporting:** `NOT STARTED`" in roadmap
    assert "Multimodal VLM:** `NOT STARTED`" in roadmap


def test_explainability_policy_distinguishes_attribution_from_localization() -> None:
    policy = (
        ROOT / "docs/research_extensions/explainability/EXPLAINABILITY_CLAIMS_POLICY.md"
    ).read_text(encoding="utf-8")
    assert "Class-specific model attribution" in policy
    assert "True lesion localization" in policy
    assert "not ground-truth localization" in policy
