from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
STATIC_ROOT = ROOT / "src/trustcxr/serving/static"


def test_research_ui_redesign_remains_local_static_and_framework_free() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    css = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
    combined = f"{html}\n{css}".lower()

    assert "trustcxr research review" in combined
    assert "dashboard-grid" in combined
    assert "--background:" in css
    assert "--surface:" in css
    assert "@media (max-width: 860px)" in css
    assert "cdn" not in combined
    assert "react" not in combined
    assert "tailwind" not in combined
    assert "http://" not in combined and "https://" not in combined


def test_local_image_selection_is_preview_only_and_releases_object_urls() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'accept=".png,.jpg,.jpeg,image/png,image/jpeg"' in html
    assert "browser-memory preview only" in html
    assert "URL.createObjectURL(file)" in javascript
    assert "URL.revokeObjectURL(localPreviewUrl)" in javascript
    assert 'fetch("/ui/fixtures.json"' in javascript
    assert 'fetch("/ui/research-review"' in javascript
    assert "FormData" not in javascript
    assert "XMLHttpRequest" not in javascript


def test_redesign_preserves_visible_research_boundaries() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    for text in (
        "RESEARCH USE ONLY",
        "NOT A MEDICAL DIAGNOSIS",
        "EXPERT REVIEW REQUIRED",
        "EXTERNAL_VALIDATION_NOT_PERFORMED",
        "Stage 23 synthetic-only DICOM",
        "Stage 24 active learning WITHHELD",
    ):
        assert text in html


def test_redesign_does_not_use_browser_persistence_or_unsafe_fixture_html() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    for prohibited in (
        "innerHTML",
        "outerHTML",
        "localStorage",
        "sessionStorage",
        "indexedDB",
        "document.cookie",
        "insertAdjacentHTML",
    ):
        assert prohibited not in javascript


def test_local_review_mode_clears_fixture_results_on_ready_analyzing_and_failure() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'id="analysis-mode-label">SYNTHETIC DEMO OUTPUT' in html
    assert '"LOCAL IMAGE REVIEW · READY"' in javascript
    assert '"LOCAL IMAGE REVIEW · ANALYZING"' in javascript
    assert '"LOCAL IMAGE REVIEW · FAILED_SANITIZED"' in javascript
    assert javascript.count("clearAnalysisForLocalReview(") >= 4
    assert "No current-image result is available because the review failed." in javascript
    assert "LOCAL_REVIEW_ENDPOINT_UNAVAILABLE_RESTART_SERVER" in javascript
    assert 'localReview ? "LOCAL_IMAGE_REVIEW" : "SYNTHETIC_DEMO"' in javascript


def test_presentation_layer_keeps_raw_governance_and_human_readable_mapping() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'DEFER: "Expert Review Required"' in javascript
    assert 'REVISE_DETERMINISTICALLY: "Draft Requires Revision"' in javascript
    assert 'ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW: "Ready for Expert Review"' in javascript
    assert 'addDataRow(decision, "Raw Stage 20 decision"' in javascript
    assert '<details class="technical-details" id="technical-details">' in html
    assert '<details class="technical-details" id="technical-details" open' not in html


def test_findings_are_sorted_without_mutating_values_and_top_three_are_derived() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert ".toSorted(" in javascript
    assert "right.score - left.score" in javascript
    assert "topSignals: sortedScores.slice(0, 3)" in javascript
    assert "item.score.toFixed(4)" in javascript
    assert "Pleural Thickening" not in javascript  # Display text is derived from the frozen key.


def test_main_presentation_is_concise_and_verifier_is_aggregated() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "Highest model signals" in javascript
    assert "Lesion localization is not available for this review." in javascript
    assert "verificationCounts" in javascript
    assert "Evidence verification is limited for this local image review." in javascript
    assert "verifier-card" not in html
    assert "data.report.statements.forEach" not in javascript


def test_failure_card_clears_current_results_without_fixture_fallback() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "showAnalysisFailure(error.message)" in javascript
    assert 'byId("analysis-error").classList.remove("is-hidden")' in javascript
    assert "renderResult(result, true)" in javascript
    assert "renderResult(fixtureData" not in javascript
