from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from scripts.ui.run_stage22c_synthetic_research_ui_implementation_validation import (
    EXPECTED_LABEL_ORDER,
    REQUIRED_REGIONS,
    decode_synthetic_image,
    validate_implementation,
)

from trustcxr.serving.api import create_app

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/ui/stage22c_synthetic_research_ui_implementation_validation.json"
STATIC_ROOT = ROOT / "src/trustcxr/serving/static"


def config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def fixtures() -> dict:
    return json.loads((STATIC_ROOT / "fixtures.json").read_text(encoding="utf-8"))


async def asgi_bytes(path: str) -> tuple[int, dict[str, str], bytes]:
    app = create_app()
    requests = [
        {"type": "http.request", "body": b"", "more_body": False},
        {"type": "http.disconnect"},
    ]
    messages: list[dict] = []

    async def receive() -> dict:
        return requests.pop(0) if requests else {"type": "http.disconnect"}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "GET",
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "root_path": "",
        "headers": [(b"host", b"127.0.0.1")],
        "client": ("127.0.0.1", 41000),
        "server": ("127.0.0.1", 42121),
    }
    await app(scope, receive, send)
    start = next(item for item in messages if item["type"] == "http.response.start")
    headers = {key.decode().lower(): value.decode() for key, value in start["headers"]}
    body = b"".join(
        item.get("body", b"") for item in messages if item["type"] == "http.response.body"
    )
    return start["status"], headers, body


def test_stage22c_static_asset_hashes_and_contract_validate() -> None:
    result = validate_implementation(config(), ROOT)
    assert result["status"] == "PASSED_SYNTHETIC_RESEARCH_UI_IMPLEMENTATION_VALIDATION"
    assert result["stage22b_contract_fingerprint"] == (
        "f413fbb969261cf9b9b51eb0870aba4de9555483acff42212c04639f293b3a2f"
    )
    assert result["static_asset_integrity"] == "PASSED"
    assert result["html_javascript_safety"] == "PASSED"


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/ui", "text/html"),
        ("/ui/app.css", "text/css"),
        ("/ui/app.js", "text/javascript"),
        ("/ui/fixtures.json", "application/json"),
    ],
)
def test_stage22c_fixed_ui_routes_serve_local_assets(path: str, content_type: str) -> None:
    status, headers, body = asyncio.run(asgi_bytes(path))
    assert status == 200 and body
    assert headers["content-type"].startswith(content_type)
    assert headers["cache-control"] == "no-store"
    assert headers["x-content-type-options"] == "nosniff"


def test_stage22c_frozen_v1_routes_are_preserved() -> None:
    routes = {
        (method, route.path)
        for route in create_app().routes
        for method in (getattr(route, "methods", None) or set())
    }
    assert {("POST", "/v1/jobs"), ("GET", "/v1/jobs/{job_id}"), ("GET", "/health")} <= routes


def test_stage22c_html_contains_banner_regions_csp_and_alt_text() -> None:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    assert all(
        phrase in html
        for phrase in (
            "RESEARCH USE ONLY",
            "NOT A MEDICAL DIAGNOSIS",
            "EXPERT REVIEW REQUIRED",
        )
    )
    assert all(f'data-region="{region}"' in html for region in REQUIRED_REGIONS)
    assert "Synthetic non-patient grayscale validation fixture" in html
    assert "object-src 'none'" in html
    assert "base-uri 'none'" in html
    assert "form-action 'none'" in html
    assert "https://" not in html and "http://" not in html


def test_stage22c_javascript_uses_safe_text_rendering_and_no_persistence() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "textContent" in javascript and "createElement" in javascript
    for prohibited in (
        "innerHTML",
        "outerHTML",
        "localStorage",
        "sessionStorage",
        "document.cookie",
        "eval(",
        "https://",
        "http://",
    ):
        assert prohibited not in javascript
    assert 'credentials: "omit"' in javascript
    assert 'cache: "no-store"' in javascript


def test_stage22c_has_two_valid_synthetic_nonpatient_viewer_fixtures() -> None:
    data = fixtures()
    assert data["non_patient"]
    assert set(data["synthetic_images"]) == {"PNG", "JPEG"}
    assert decode_synthetic_image(data["synthetic_images"]["PNG"], "PNG") == (32, 24)
    assert decode_synthetic_image(data["synthetic_images"]["JPEG"], "JPEG") == (32, 24)
    assert all("Synthetic non-patient" in item["alt"] for item in data["synthetic_images"].values())


def test_stage22c_view_panel_is_limited_to_ap_pa_lateral_and_proxy_wording() -> None:
    data = fixtures()
    assert data["view"]["states"] == ["AP", "PA", "LATERAL"]
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "Model-identified view" in javascript
    assert "not clinical certification" in javascript
    assert (
        "Research technical-quality proxy warning; not a clinical image-quality assessment."
        in javascript
    )


def test_stage22c_classifier_has_exact_14_labels_and_research_scores() -> None:
    data = fixtures()
    assert [item["label"] for item in data["classifier_scores"]] == EXPECTED_LABEL_ORDER
    assert len(data["classifier_scores"]) == 14
    assert all(isinstance(item["score"], float) for item in data["classifier_scores"])
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "Research model signal only" in javascript
    assert (
        "not diagnoses, thresholds, severity, temporal change, or clinical certainty" in javascript
    )


def test_stage22c_reliability_fusion_and_triage_wording_is_frozen() -> None:
    data = fixtures()
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "PREDICTIVE UNCERTAINTY" in javascript
    assert "EPISTEMIC UNCERTAINTY" not in javascript
    assert data["reliability"]["ood"] == "WITHHELD"
    assert data["reliability"]["stage13_selective_prediction"] == "NOT_ACCEPTED"
    assert data["fusion"]["status"] == "PARTIALLY_SUPPORTED"
    assert "not full verification" in data["fusion"]["label"]
    assert "DEFER" in data["decisions"]["precedence"]


def test_stage22c_report_verifier_and_decision_contracts_are_complete() -> None:
    data = fixtures()
    assert data["report"]["identity"] == "AI_GENERATED_RESEARCH_REPORT_DRAFT_FOR_EXPERT_REVIEW"
    assert {item["status"] for item in data["verifier_statuses"]} == {
        "VERIFIED",
        "PARTIALLY_VERIFIED",
        "UNVERIFIED",
        "CONTRADICTED",
        "NOT_APPLICABLE",
        "WITHHELD_INSUFFICIENT_EVIDENCE",
    }
    assert data["decisions"]["precedence"] == [
        "DEFER",
        "REVISE_DETERMINISTICALLY",
        "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
    ]
    assert "NOT CLINICAL APPROVAL" in data["decisions"]["accept"]
    assert "DETERMINISTIC CANONICAL TEMPLATE REPAIR ONLY" in data["decisions"]["revise"]


def test_stage22c_defer_and_failed_sanitized_are_visually_and_textually_distinct() -> None:
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    css = (STATIC_ROOT / "app.css").read_text(encoding="utf-8")
    assert "DEFER — safety/evidence limitation" in javascript
    assert "FAILED_SANITIZED — technical processing failure" in javascript
    assert ".status-defer" in css and ".status-failed" in css


def test_stage22c_provenance_contains_only_safe_synthetic_fields() -> None:
    provenance = fixtures()["provenance"]
    assert set(provenance) == {
        "SOURCE_STAGE",
        "SOURCE_VERSION",
        "EVIDENCE_CODE",
        "COMPONENT_IDENTIFIER",
        "CONFIG_FINGERPRINT",
        "CHECKPOINT_FINGERPRINT",
        "VERIFIER_EVIDENCE_REFERENCE",
    }
    serialized = json.dumps(provenance).lower()
    assert "patient" not in serialized and "local_path" not in serialized


def test_stage22c_dicom_and_both_overlays_remain_withheld() -> None:
    cases = fixtures()["security_cases"]
    assert cases["dicom"] == "WITHHELD"
    assert cases["stage8_overlay"] == "WITHHELD"
    assert cases["stage10_overlay"] == "WITHHELD"
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert javascript.count("WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION") == 2


def test_stage22c_malicious_fixture_cannot_be_inserted_as_html() -> None:
    malicious = fixtures()["security_cases"]["malicious_text"]
    assert "<script" in malicious
    javascript = (STATIC_ROOT / "app.js").read_text(encoding="utf-8")
    assert "innerHTML" not in javascript
    assert "textContent" in javascript


def test_stage22c_css_has_textual_noncolor_distinctions_and_no_diagnostic_names() -> None:
    css = (STATIC_ROOT / "app.css").read_text(encoding="utf-8").lower()
    assert "status-partial" in css and "status-withheld" in css
    assert "disease" not in css and "healthy" not in css
    assert "severity" not in css and "diagnos" not in css


@pytest.mark.parametrize(
    "key",
    [
        "real_image_display_permitted",
        "real_model_loading_permitted",
        "real_inference_permitted",
        "gpu_residency_profiling_permitted",
        "real_patient_processing_permitted",
        "locked_test_access_permitted",
        "persistent_server_permitted",
        "persistent_browser_permitted",
        "stage8_overlay_activation_permitted",
        "stage10_overlay_activation_permitted",
        "dicom_support_permitted",
        "language_model_used",
        "language_model_endpoint_prepared",
    ],
)
def test_stage22c_rejects_scope_expansion(key: str) -> None:
    cfg = config()
    cfg[key] = True
    with pytest.raises(RuntimeError, match="prohibits"):
        validate_implementation(cfg, ROOT)


def test_stage22c_next_stage_authorizes_only_bounded_synthetic_ui_runtime() -> None:
    result = validate_implementation(config(), ROOT)
    assert result["next_canonical_stage"] == ("22D_BOUNDED_SYNTHETIC_UI_RUNTIME_BROWSER_VALIDATION")
    assert result["next_stage_authorizes_bounded_ui_runtime_browser_validation"]
    assert not result["next_stage_authorizes_real_image_display"]
    assert not result["next_stage_authorizes_real_model_inference"]
    assert not result["next_stage_authorizes_language_model_work"]
    assert result["currently_planned_llm_authorized_gate"] is None


def test_stage22c_validation_does_not_write_summary() -> None:
    output = ROOT / "reports/stage22/stage22c_synthetic_research_ui_implementation_summary.json"
    before = output.read_bytes() if output.exists() else None
    validate_implementation(config(), ROOT)
    after = output.read_bytes() if output.exists() else None
    assert after == before
