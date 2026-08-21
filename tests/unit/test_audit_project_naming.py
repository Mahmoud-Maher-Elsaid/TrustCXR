from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "audit_project_naming", ROOT / "scripts/qa/audit_project_naming.py"
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules["audit_project_naming"] = MODULE
SPEC.loader.exec_module(MODULE)


def test_naming_auditor_classifies_future_extension_terms(tmp_path: Path) -> None:
    (tmp_path / "docs/research_extensions").mkdir(parents=True)
    (tmp_path / "docs/research_extensions/roadmap.md").write_text(
        "Grad-CAM is NOT IMPLEMENTED.\n", encoding="utf-8"
    )
    original = MODULE.tracked_files
    original_gh = MODULE.github_metadata
    MODULE.tracked_files = lambda _root: [tmp_path / "docs/research_extensions/roadmap.md"]
    MODULE.github_metadata = lambda: None
    try:
        report = MODULE.audit(tmp_path)
    finally:
        MODULE.tracked_files = original
        MODULE.github_metadata = original_gh
    assert report["errors"] == 0
    assert report["counts"]["ALLOWED_FUTURE_WORK"] >= 1


def test_naming_auditor_rejects_machine_path(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("Data: F:\\AI\\TrustCXR\\TrustCXR-Data\n", encoding="utf-8")
    original = MODULE.tracked_files
    original_gh = MODULE.github_metadata
    MODULE.tracked_files = lambda _root: [tmp_path / "README.md"]
    MODULE.github_metadata = lambda: None
    try:
        report = MODULE.audit(tmp_path)
    finally:
        MODULE.tracked_files = original
        MODULE.github_metadata = original_gh
    assert report["errors"] == 1
    assert report["findings"][0]["rule"] == "PRIVATE_MACHINE_PATH"


def run_audit(tmp_path: Path, relative_path: str, content: str) -> dict:
    target = tmp_path / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    original = MODULE.tracked_files
    original_gh = MODULE.github_metadata
    MODULE.tracked_files = lambda _root: [target]
    MODULE.github_metadata = lambda: None
    try:
        return MODULE.audit(tmp_path)
    finally:
        MODULE.tracked_files = original
        MODULE.github_metadata = original_gh


def test_canonical_brand_is_not_a_variant(tmp_path: Path) -> None:
    report = run_audit(tmp_path, "README.md", "TrustCXR\ntrustcxr package\n")
    assert report["errors"] == 0
    assert report["counts"]["WARNING"] == 0


def test_noncanonical_brand_variants_are_detected(tmp_path: Path) -> None:
    report = run_audit(tmp_path, "README.md", "Trust CXR\nTrust-CXR\nTRUST-CXR\n")
    assert report["counts"]["WARNING"] == 3


def test_rule_literals_are_not_self_scanned(tmp_path: Path) -> None:
    report = run_audit(
        tmp_path, "scripts/qa/audit_project_naming.py", "F:\\AI\\TrustCXR\nTrust CXR\n"
    )
    assert report["errors"] == 0
    assert report["findings"] == []


def test_test_fixture_machine_paths_are_allowed(tmp_path: Path) -> None:
    report = run_audit(tmp_path, "tests/unit/example.py", "F:\\AI\\TrustCXR\n")
    assert report["errors"] == 0
    assert report["counts"]["ALLOWED_TEST_FIXTURE"] == 1


def test_intentional_test_claims_are_allowed_fixtures(tmp_path: Path) -> None:
    report = run_audit(tmp_path, "tests/unit/example.py", "TrustCXR implements Grad-CAM\n")
    assert report["errors"] == 0
    assert report["counts"]["ALLOWED_TEST_FIXTURE"] >= 1


def test_historical_machine_paths_are_nonblocking(tmp_path: Path) -> None:
    report = run_audit(tmp_path, "reports/stage9/evidence.md", "F:\\AI\\TrustCXR\n")
    assert report["errors"] == 0
    assert report["counts"]["WARNING"] == 1


def test_current_runtime_machine_paths_are_errors(tmp_path: Path) -> None:
    report = run_audit(tmp_path, "src/trustcxr/serving/api.py", "F:\\AI\\TrustCXR\n")
    assert report["errors"] == 1


def test_future_and_current_capability_claims_are_distinguished(tmp_path: Path) -> None:
    future = run_audit(
        tmp_path, "docs/research_extensions/roadmap.md", "Grad-CAM is NOT IMPLEMENTED.\n"
    )
    current = run_audit(tmp_path, "README.md", "TrustCXR implements Grad-CAM.\n")
    assert future["errors"] == 0
    assert future["counts"]["ALLOWED_FUTURE_WORK"] >= 1
    assert current["errors"] == 1


def test_prohibited_claims_list_does_not_trigger_overclaim(tmp_path: Path) -> None:
    content = """The release must never be described as:
- using an LLM, VLM, or implemented Grad-CAM extension;
"""
    report = run_audit(tmp_path, "docs/release/FINAL_CLAIMS_MATRIX.md", content)
    assert report["errors"] == 0
    assert not any(finding["rule"] == "OVERCLAIMED_CAPABILITY" for finding in report["findings"])
