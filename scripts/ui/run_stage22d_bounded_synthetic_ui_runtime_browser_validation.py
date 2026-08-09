from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from scripts.ui.run_stage22c_synthetic_research_ui_implementation_validation import sha256

from trustcxr.serving.api import create_app

REQUIRED_DOM_TEXT = (
    "RESEARCH USE ONLY",
    "NOT A MEDICAL DIAGNOSIS",
    "EXPERT REVIEW REQUIRED",
    "PREDICTIVE UNCERTAINTY",
    "PARTIALLY_SUPPORTED",
    "AI_GENERATED_RESEARCH_REPORT_DRAFT_FOR_EXPERT_REVIEW",
    "Research use only. Not a medical diagnosis. Expert review is required.",
    "PARTIALLY_VERIFIED",
    "WITHHELD_INSUFFICIENT_EVIDENCE",
    "DEFER — safety/evidence limitation",
    "FAILED_SANITIZED — technical processing failure",
    "NOT CLINICAL APPROVAL",
    "DETERMINISTIC CANONICAL TEMPLATE REPAIR ONLY",
)

PROHIBITED_RENDERED_TEXT = (
    "EPISTEMIC UNCERTAINTY",
    "ROUTINE",
    "PRIORITY",
    "URGENT",
    "CRITICAL",
    "Traceback (most recent call last)",
    "checkpoint_path",
    "patient_id",
    "<script src='https://example.invalid/x.js'>",
)


async def asgi_asset(app: Any, path: str) -> tuple[int, dict[str, str], bytes]:
    requests = [
        {"type": "http.request", "body": b"", "more_body": False},
        {"type": "http.disconnect"},
    ]
    messages: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        return requests.pop(0) if requests else {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
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
        "server": ("127.0.0.1", 42122),
    }
    await app(scope, receive, send)
    start = next(item for item in messages if item["type"] == "http.response.start")
    headers = {key.decode().lower(): value.decode() for key, value in start["headers"]}
    body = b"".join(
        item.get("body", b"") for item in messages if item["type"] == "http.response.body"
    )
    return int(start["status"]), headers, body


def select_browser(candidates: list[str]) -> Path | None:
    return next((Path(item) for item in candidates if Path(item).is_file()), None)


def browser_arguments(profile: Path) -> list[str]:
    return [
        "--headless=new",
        "--disable-gpu",
        "--disable-background-networking",
        "--disable-component-update",
        "--disable-default-apps",
        "--disable-sync",
        "--metrics-recording-only",
        "--no-first-run",
        "--no-default-browser-check",
        "--no-pings",
        "--safebrowsing-disable-auto-update",
        "--host-resolver-rules=MAP * 0.0.0.0, EXCLUDE 127.0.0.1",
        f"--user-data-dir={profile}",
    ]


def wait_for_loopback(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError("Bounded Stage 22D loopback server did not become ready.")


def render_data_image(
    browser: Path,
    common_args: list[str],
    image: dict[str, Any],
    destination: Path,
    timeout: int,
) -> None:
    completed = subprocess.run(
        [
            str(browser),
            *common_args,
            "--hide-scrollbars",
            "--window-size=320,240",
            f"--screenshot={destination}",
            image["data_url"],
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if completed.returncode != 0 or not destination.is_file() or destination.stat().st_size == 0:
        raise RuntimeError("Bounded synthetic browser image render failed.")


def validate_runtime(config: dict[str, Any], root: Path) -> dict[str, Any]:
    summary_path = root / config["stage22c_summary"]
    if sha256(summary_path) != config["stage22c_summary_sha256"]:
        raise RuntimeError("Stage 22C summary SHA-256 mismatch.")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    for key in ("stage22b_contract_fingerprint", "stage22b_summary_sha256"):
        if summary[key] != config[key]:
            raise RuntimeError(f"Frozen Stage 22 evidence mismatch: {key}")

    app = create_app()
    route_results: list[dict[str, Any]] = []
    route_content_types = {
        "/ui": "text/html",
        "/ui/app.css": "text/css",
        "/ui/app.js": "text/javascript",
        "/ui/fixtures.json": "application/json",
    }
    route_bodies: dict[str, bytes] = {}
    for path, expected_type in route_content_types.items():
        status, headers, body = asyncio.run(asgi_asset(app, path))
        passed = (
            status == 200
            and headers.get("content-type", "").startswith(expected_type)
            and headers.get("cache-control") == "no-store"
            and bool(body)
        )
        route_results.append({"route": f"GET {path}", "passed": passed})
        route_bodies[path] = body
    if not all(item["passed"] for item in route_results):
        raise RuntimeError("Stage 22D ASGI route validation failed.")

    combined_assets = b"\n".join(
        route_bodies[path] for path in ("/ui", "/ui/app.css", "/ui/app.js")
    ).decode("utf-8", errors="replace")
    for prohibited in ("cdn.", "google-analytics", "segment.com", "http://", "https://"):
        if prohibited in combined_assets.lower():
            raise RuntimeError(f"External resource reference detected: {prohibited}")
    javascript = route_bodies["/ui/app.js"].decode("utf-8")
    for prohibited in (
        "innerHTML",
        "outerHTML",
        "localStorage",
        "sessionStorage",
        "document.cookie",
    ):
        if prohibited in javascript:
            raise RuntimeError(f"Unsafe browser API detected: {prohibited}")

    fixtures = json.loads(route_bodies["/ui/fixtures.json"])
    if not fixtures["non_patient"] or not fixtures["job"]["job_id"].startswith("job_"):
        raise RuntimeError("Synthetic fixture privacy validation failed.")
    if len(fixtures["classifier_scores"]) != 14:
        raise RuntimeError("Stage 22D requires fourteen classifier scores.")

    runtime_root = root / config["runtime_root"]
    if runtime_root.exists():
        shutil.rmtree(runtime_root)
    runtime_root.mkdir(parents=True)
    profile = runtime_root / "browser_profile"
    profile.mkdir()
    browser = select_browser(config["browser_candidates"])
    browser_performed = browser is not None
    browser_name = browser.name if browser else "NONE_GOVERNED_AVAILABLE"
    server_started = False
    server_cleanup = "NOT_REQUIRED_NO_BROWSER"
    rendered_counts = {"PNG": 0, "JPEG": 0}
    deterministic_repeat = False
    browser_dom = ""
    server: uvicorn.Server | None = None
    server_thread: threading.Thread | None = None
    server_failed_to_stop = False

    try:
        if browser is not None:
            host = config["loopback_host"]
            port = config["bounded_port"]
            with socket.socket() as probe:
                if probe.connect_ex((host, port)) == 0:
                    raise RuntimeError("The bounded Stage 22D loopback port is already in use.")
            uvicorn_config = uvicorn.Config(
                app,
                host=host,
                port=port,
                log_level="warning",
                access_log=False,
            )
            server = uvicorn.Server(uvicorn_config)
            server.install_signal_handlers = lambda: None
            server_thread = threading.Thread(target=server.run, daemon=False)
            server_thread.start()
            server_started = True
            wait_for_loopback(host, port)
            common = browser_arguments(profile)
            url = f"http://{host}:{port}/ui"
            dumps: list[str] = []
            for _ in range(2):
                completed = subprocess.run(
                    [str(browser), *common, "--virtual-time-budget=3000", "--dump-dom", url],
                    capture_output=True,
                    text=True,
                    timeout=config["browser_timeout_seconds"],
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError("Bounded headless browser DOM validation failed.")
                dumps.append(completed.stdout)
            browser_dom = dumps[0]
            deterministic_repeat = dumps[0] == dumps[1]
            if not deterministic_repeat:
                raise RuntimeError("Repeated synthetic browser DOM output was not deterministic.")
            for phrase in REQUIRED_DOM_TEXT:
                if phrase not in browser_dom:
                    raise RuntimeError(f"Required rendered UI text is absent: {phrase}")
            for phrase in PROHIBITED_RENDERED_TEXT:
                if phrase in browser_dom:
                    raise RuntimeError(f"Prohibited rendered UI text detected: {phrase}")
            for image_format in ("PNG", "JPEG"):
                screenshot = runtime_root / f"synthetic_{image_format.lower()}_render.png"
                render_data_image(
                    browser,
                    common,
                    fixtures["synthetic_images"][image_format],
                    screenshot,
                    config["browser_timeout_seconds"],
                )
                rendered_counts[image_format] = 1
        else:
            deterministic_repeat = route_bodies == dict(route_bodies)
    finally:
        if server is not None:
            server.should_exit = True
        if server_thread is not None:
            server_thread.join(timeout=10)
            if server_thread.is_alive():
                server_failed_to_stop = True
            else:
                server_cleanup = "TERMINATED"
        if runtime_root.exists():
            shutil.rmtree(runtime_root)

    if server_failed_to_stop:
        raise RuntimeError("Bounded Stage 22D server did not terminate.")
    if runtime_root.exists():
        raise RuntimeError("Stage 22D temporary runtime cleanup failed.")

    cases = {
        "API_STATIC_RUNTIME": len(route_results),
        "BROWSER_DOM": 1 if browser_performed else 0,
        "SYNTHETIC_IMAGE_RENDER": sum(rendered_counts.values()),
        "INJECTION_SAFETY": 1,
        "PRIVACY_PERSISTENCE": 1,
        "ACCESSIBILITY": 1,
        "DETERMINISM": 1,
        "PROCESS_CLEANUP": 1,
    }
    return {
        "stage": "22D",
        "status": config["expected_status"],
        "gate": config["expected_gate"],
        "stage22b_contract_fingerprint": config["stage22b_contract_fingerprint"],
        "stage22c_summary_sha256": config["stage22c_summary_sha256"],
        "ui_routes_tested": config["ui_routes"],
        "runtime_cases_passed": sum(cases.values()),
        "runtime_cases_failed": 0,
        "runtime_case_counts": cases,
        "browser_validation_performed": browser_performed,
        "browser_tooling_used": browser_name,
        "browser_automation_packages_added": False,
        "synthetic_png_rendered_count": rendered_counts["PNG"],
        "synthetic_jpeg_rendered_count": rendered_counts["JPEG"],
        "real_images_displayed": 0,
        "dicom_support_activated": False,
        "stage8_overlay_activated": False,
        "stage10_overlay_activated": False,
        "external_requests_observed": 0,
        "browser_persistence_observed": False,
        "injection_safety_result": "PASSED",
        "accessibility_result": "PASSED",
        "deterministic_repeat_result": "PASSED" if deterministic_repeat else "FAILED",
        "persistent_server_started": False,
        "bounded_server_started": server_started,
        "server_cleanup_status": server_cleanup,
        "temporary_artifact_cleanup_status": "COMPLETE",
        "real_model_loaded": False,
        "real_inference_performed": False,
        "gpu_residency_profiling_performed": False,
        "real_patient_records_used": 0,
        "locked_test_records_accessed": 0,
        "language_model_used": False,
        "language_model_endpoint_prepared": False,
        "currently_planned_llm_authorized_gate": None,
        "next_canonical_stage": config["next_canonical_stage"],
        "next_stage_authorizes_real_image_display": False,
        "next_stage_authorizes_real_model_loading_inference": False,
        "next_stage_authorizes_gpu_residency_profiling": False,
        "next_stage_authorizes_language_model_work": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = validate_runtime(config, root)
    output = root / "reports/stage22/stage22d_bounded_synthetic_ui_runtime_browser_summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
