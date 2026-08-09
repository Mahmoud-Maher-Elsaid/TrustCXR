from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from scripts.ui.run_stage22d_bounded_synthetic_ui_runtime_browser_validation import (
    PROHIBITED_RENDERED_TEXT,
    REQUIRED_DOM_TEXT,
    asgi_asset,
    browser_arguments,
    select_browser,
)

from trustcxr.serving.api import create_app

ROOT = Path(__file__).resolve().parents[2]
CONFIG = json.loads(
    (ROOT / "configs/ui/stage22d_bounded_synthetic_ui_runtime_browser_validation.json").read_text(
        encoding="utf-8"
    )
)
STATIC_ROOT = ROOT / "src/trustcxr/serving/static"


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/ui", "text/html"),
        ("/ui/app.css", "text/css"),
        ("/ui/app.js", "text/javascript"),
        ("/ui/fixtures.json", "application/json"),
    ],
)
def test_stage22d_actual_asgi_ui_routes(path: str, content_type: str) -> None:
    status, headers, body = asyncio.run(asgi_asset(create_app(), path))
    assert status == 200 and body
    assert headers["content-type"].startswith(content_type)
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"


def test_stage22d_assets_contain_no_external_resources_or_analytics() -> None:
    combined = "\n".join(
        (STATIC_ROOT / name).read_text(encoding="utf-8")
        for name in ("index.html", "app.css", "app.js", "fixtures.json")
    ).lower()
    for prohibited in (
        "cdn.",
        "google-analytics",
        "segment.com",
        "telemetry",
        "http://",
    ):
        assert prohibited not in combined
    assert "https://example.invalid" in combined
    assert combined.count("https://example.invalid") == 1
    assert "security_cases" in combined


def test_stage22d_malicious_external_markup_is_never_rendered() -> None:
    fixtures = json.loads((STATIC_ROOT / "fixtures.json").read_text(encoding="utf-8"))
    malicious = fixtures["security_cases"]["malicious_text"]
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "<script" in malicious
    assert "security_cases" not in javascript
    assert "innerHTML" not in javascript
    assert "textContent" in javascript


def test_stage22d_required_rendered_text_contract_is_complete() -> None:
    assert {
        "RESEARCH USE ONLY",
        "NOT A MEDICAL DIAGNOSIS",
        "EXPERT REVIEW REQUIRED",
        "PREDICTIVE UNCERTAINTY",
        "PARTIALLY_SUPPORTED",
        "NOT CLINICAL APPROVAL",
        "DETERMINISTIC CANONICAL TEMPLATE REPAIR ONLY",
    } <= set(REQUIRED_DOM_TEXT)


def test_stage22d_prohibited_rendered_text_contract_is_complete() -> None:
    assert {
        "EPISTEMIC UNCERTAINTY",
        "ROUTINE",
        "PRIORITY",
        "URGENT",
        "CRITICAL",
        "checkpoint_path",
        "patient_id",
    } <= set(PROHIBITED_RENDERED_TEXT)


def test_stage22d_fixture_has_exact_scores_verifiers_decisions_and_provenance() -> None:
    fixture = json.loads((STATIC_ROOT / "fixtures.json").read_text(encoding="utf-8"))
    assert len(fixture["classifier_scores"]) == 14
    assert [item["label"] for item in fixture["classifier_scores"]] == [
        "Atelectasis",
        "Cardiomegaly",
        "Effusion",
        "Infiltration",
        "Mass",
        "Nodule",
        "Pneumonia",
        "Pneumothorax",
        "Consolidation",
        "Edema",
        "Emphysema",
        "Fibrosis",
        "Pleural_Thickening",
        "Hernia",
    ]
    assert len(fixture["verifier_statuses"]) == 6
    assert fixture["decisions"]["precedence"] == [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]
    assert list(fixture["provenance"]) == [
        "SOURCE_STAGE",
        "SOURCE_VERSION",
        "EVIDENCE_CODE",
        "COMPONENT_IDENTIFIER",
        "CONFIG_FINGERPRINT",
        "CHECKPOINT_FINGERPRINT",
        "VERIFIER_EVIDENCE_REFERENCE",
    ]


def test_stage22d_fixture_is_nonpatient_and_overlays_dicom_are_withheld() -> None:
    fixture = json.loads((STATIC_ROOT / "fixtures.json").read_text(encoding="utf-8"))
    assert fixture["non_patient"]
    assert fixture["job"]["job_id"].startswith("job_")
    assert fixture["security_cases"]["patient_identifier_field"] == "REJECTED"
    assert fixture["security_cases"]["internal_path_field"] == "REJECTED"
    assert fixture["security_cases"]["dicom"] == "WITHHELD"
    assert fixture["security_cases"]["stage8_overlay"] == "WITHHELD"
    assert fixture["security_cases"]["stage10_overlay"] == "WITHHELD"


def test_stage22d_javascript_has_no_browser_persistence_or_cookie_access() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    for prohibited in ("localStorage", "sessionStorage", "document.cookie", "indexedDB"):
        assert prohibited not in javascript
    assert 'credentials: "omit"' in javascript
    assert 'cache: "no-store"' in javascript


def test_stage22d_accessibility_is_textual_and_not_color_only() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    css = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
    assert html.count("<h2>") == 11
    assert 'role="banner"' in html
    assert 'alt="Synthetic non-patient grayscale validation fixture"' in html
    assert "button" in css and "select" in css
    assert "status-partial" in css and "status-withheld" in css


def test_stage22d_already_installed_browser_candidates_are_bounded() -> None:
    browser = select_browser(CONFIG["browser_candidates"])
    assert browser is not None
    assert browser.name in {"msedge.exe", "chrome.exe"}
    arguments = browser_arguments(Path("synthetic-profile"))
    assert "--headless=new" in arguments
    assert "--disable-background-networking" in arguments
    assert "--no-pings" in arguments
    assert any(item.startswith("--host-resolver-rules=") for item in arguments)


def test_stage22d_no_browser_automation_package_is_required() -> None:
    lock = (ROOT / "requirements/lock-serving-stage21.txt").read_text(encoding="utf-8").lower()
    assert "playwright" not in lock
    assert "selenium" not in lock
    assert "pyppeteer" not in lock


def test_stage22d_runtime_configuration_is_loopback_bounded_and_cleanup_scoped() -> None:
    assert CONFIG["loopback_host"] == "127.0.0.1"
    assert CONFIG["bounded_port"] == 42122
    assert CONFIG["browser_timeout_seconds"] == 30
    assert CONFIG["runtime_root"].startswith("cache/")
    assert not CONFIG["persistent_server_permitted"]
    assert not CONFIG["persistent_browser_permitted"]


@pytest.mark.parametrize(
    "key",
    [
        "real_image_display_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "real_patient_processing_permitted",
        "locked_test_access_permitted",
        "stage8_overlay_activation_permitted",
        "stage10_overlay_activation_permitted",
        "dicom_support_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage22d_prohibited_capabilities_remain_false(key: str) -> None:
    assert not CONFIG[key]


def test_stage22d_next_stage_is_acceptance_only_without_runtime_expansion() -> None:
    assert CONFIG["next_canonical_stage"] == "22E_SYNTHETIC_UI_ACCEPTANCE_DECISION"
    assert not CONFIG["next_stage_authorizes_real_image_display"]
    assert not CONFIG["next_stage_authorizes_real_model_loading_inference"]
    assert not CONFIG["next_stage_authorizes_gpu_residency_profiling"]
    assert not CONFIG["next_stage_authorizes_language_model_work"]
    assert CONFIG["currently_planned_llm_authorized_gate"] is None
