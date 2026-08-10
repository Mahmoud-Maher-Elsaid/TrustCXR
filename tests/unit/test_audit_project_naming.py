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
