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

    assert "Top model signals" in html
    assert "produced the highest signal" in javascript
    assert "Reliable lesion-localization evidence is not available for this image" in javascript
    assert "verificationCounts" in javascript
    assert "Evidence verification is limited for this local image review." in javascript
    assert "verifier-card" not in html
    assert "data.report.statements.forEach" not in javascript


def test_case_interpretation_derives_primary_and_secondary_signals() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "Research Interpretation" in html
    assert "const primary = presentation.topSignals[0]" in javascript
    assert "presentation.topSignals.slice(1)" in javascript
    assert "presentation.topSignals[1]" in javascript
    assert "presentation.topSignals[2]" in javascript
    assert (
        'byId("interpretation-primary-score").textContent = primary.score.toFixed(3)' in javascript
    )


def test_case_explanation_is_deterministic_and_uses_current_values() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "The uploaded chest X-ray was identified as a ${presentation.view} view" in javascript
    assert "${primary.score.toFixed(3)}" in javascript
    assert "${second.score.toFixed(3)}" in javascript
    assert "${third.score.toFixed(3)}" in javascript
    assert "uncertaintyExplanation(presentation.uncertainty.label)" in javascript
    assert (
        "These outputs are research model signals and do not constitute a medical diagnosis."
        in javascript
    )
    for prohibited in (
        "probability of disease",
        "diagnosis confidence",
        "confirmed finding",
        "positive disease",
    ):
        assert prohibited not in javascript.lower()


def test_uncertainty_explanations_follow_the_presentation_category() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert (
        "Predictive uncertainty is high. Model outputs should be interpreted cautiously"
        in javascript
    )
    assert (
        "Predictive uncertainty is moderate. Model outputs contain meaningful uncertainty"
        in javascript
    )
    assert "Predictive uncertainty is relatively low for this model" in javascript
    assert 'if (fraction < 1 / 3) return { label: "Low", value }' in javascript
    assert 'if (fraction < 2 / 3) return { label: "Moderate", value }' in javascript


def test_advanced_research_details_are_collapsed_and_contain_raw_governance() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")

    marker = '<details class="technical-details" id="technical-details">'
    assert marker in html
    advanced = html.split(marker, maxsplit=1)[1].split("</details>", maxsplit=1)[0]
    assert "Advanced Research Details" in advanced
    assert "Raw decision state" in advanced
    assert "Raw verifier evidence" in advanced
    assert "Safe provenance" in advanced
    assert "Stage 11" in advanced


def test_failure_card_clears_current_results_without_fixture_fallback() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert "showAnalysisFailure(error.message)" in javascript
    assert 'byId("analysis-error").classList.remove("is-hidden")' in javascript
    assert "renderResult(result, true)" in javascript
    assert "renderResult(fixtureData" not in javascript


def test_frozen_core_ui_final_polish_contract() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")

    assert 'content="FROZEN_CORE_RESEARCH_REVIEW_UI"' in html
    assert "Analysis completed. Expert review required." in javascript
    assert "Highest model response — not a diagnosis." in html
    assert "Evidence verification is limited for this local image review." in javascript
    assert (
        "Reliable lesion localization and finding laterality are unavailable for this review."
        in html
    )
    marker = '<details class="technical-details" id="technical-details">'
    advanced = html.split(marker, maxsplit=1)[1].split("</details>", maxsplit=1)[0]
    assert "Raw verifier evidence" in advanced
    assert "WITHHELD_INSUFFICIENT_EVIDENCE" not in html.split(marker, maxsplit=1)[0]


def test_post_release_extension_roadmap_preserves_current_lifecycle_and_core_boundaries() -> None:
    roadmap = (
        ROOT / "docs/research_extensions/POST_RELEASE_RESEARCH_EXTENSION_ROADMAP.md"
    ).read_text(encoding="utf-8")
    governance = (
        ROOT / "docs/research_extensions/grounded_llm/EXT4A_GROUNDED_LLM_GOVERNANCE.md"
    ).read_text(encoding="utf-8")
    governance_normalized = " ".join(governance.split())

    for required in (
        "OPTIONAL_POST_RELEASE_RESEARCH_EXTENSION",
        "Implementation status:** `EXT-4A GOVERNANCE COMPLETED — IMPLEMENTATION NOT STARTED`",
        "Authorization:** `POST-CORE-RELEASE ONLY`",
        "EXT-1A Explainability Foundation:** `COMPLETED`",
        "EXT-1B Grad-CAM Implementation and Technical Validation:** `COMPLETED`",
        "Grad-CAM UI Integration / EXT-1C:** `COMPLETED`",
        "EXT-3 Final RSNA Lung Opacity localization:** `CLOSED_CONTROLLED_NEGATIVE_RESULT`",
        "EXT-4A Grounded LLM Governance:** `COMPLETED — GOVERNANCE ONLY`",
        "EXT-4B Evidence Grounding Schema:** `COMPLETED — SCHEMA ONLY`",
        "EXT-4C Output Contract:** `COMPLETED — OUTPUT CONTRACT ONLY`",
        "EXT-4D Hallucination / Faithfulness Benchmark:** `COMPLETED — "
        "FROZEN BENCHMARK DEFINITION ONLY`",
        "EXT-4E2B Qwen Runtime/Model Identity:** `RUNTIME_AND_MODEL_IDENTITY_FROZEN`",
        "EXT-4E2C Qwen Load-Only GPU Smoke:** `LOAD_ONLY_PASS — TECHNICALLY "
        "LOADABLE; NOT SCIENTIFICALLY EVALUATED`",
        "EXT-4E2D Candidate #1 Single Development-Case Smoke:** `CLOSED / PASS",
        "EXT-4E2D0 Structured Output Compatibility Smoke:** `CLOSED / PASS — "
        "REQUEST_RESPONSE_FORMAT_JSON_OBJECT_WITH_SCHEMA_VERIFIED`",
        "Structured-output compatibility:** `VERIFIED`",
        "Next authorized stage:** `EXT4F.4_SYNTHETIC_LLM_TECHNICAL_SMOKE — NOT STARTED`",
        "Grounded LLM reporting:** `EXT-4F REALIZATION ADAPTER IMPLEMENTED — ZERO GENERATION`",
        "attention or attribution is not validated lesion localization",
        "NO_RELIABLE_POSITIVE_LESION_LOCALIZATION",
        "LOCALIZATION_ABSENCE_MUST_NOT_CONTRADICT_CLASSIFIER_EVIDENCE",
        "DETERMINISTIC_REPORTER",
        "GROUNDED_LLM_REPORTER",
        "Frozen vision models",
        "Deterministic verifier",
        "Stage 18 deterministic renderer remains the safe fallback",
    ):
        assert required in roadmap

    for required in (
        "No provider or model is selected by EXT-4A.",
        "No LLM client, provider, external API, UI, training, or fine-tuning is part of EXT-4A.",
    ):
        assert required in governance_normalized

    assert "research-extension/explainability-grounded-llm" in roadmap
