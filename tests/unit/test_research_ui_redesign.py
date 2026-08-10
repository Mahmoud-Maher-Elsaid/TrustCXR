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
    assert 'preservePreview ? "Local frozen-model research inference"' in javascript
