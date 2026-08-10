"""Read-only TrustCXR branding, path, and capability-claim auditor.

The auditor intentionally treats historical stage evidence and the post-release
roadmap differently from current user-facing claims.  It never edits files.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = {
    "brand": "TrustCXR",
    "package": "trustcxr",
    "core_release": "TRUSTCXR_FROZEN_RESEARCH_RELEASE",
    "ui": "FROZEN_CORE_RESEARCH_REVIEW_UI",
    "tag": "v1.0.0-research",
    "branch": "research-extension/explainability-grounded-llm",
}
TEXT_SUFFIXES = {
    ".md",
    ".txt",
    ".json",
    ".py",
    ".ps1",
    ".toml",
    ".yml",
    ".yaml",
    ".html",
    ".css",
    ".js",
    ".tsx",
    ".xml",
}
CURRENT_PUBLIC_DOCS = {
    "README.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "docs/data/DATA_GOVERNANCE.md",
    "docs/execution/LOCAL_EXECUTION_GUIDE.md",
    "docs/execution/QUICK_START.md",
}
CURRENT_DOC_PREFIXES = ("docs/release/", "docs/research_extensions/")
CURRENT_RUNTIME_PREFIXES = ("src/trustcxr/serving/",)
HISTORICAL_PREFIXES = (
    "configs/",
    "reports/",
    "scripts/",
    "docs/execution/stages/",
    "docs/fusion/",
    "docs/training/",
    "docs/research_protocol/",
    "src/trustcxr/segmentation/",
)
AUDITOR_DEFINITION_FILES = {
    "scripts/qa/audit_project_naming.py",
}


@dataclass(frozen=True)
class Finding:
    severity: str
    rule: str
    path: str
    line: int | None
    message: str


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    return [root / item for item in result.stdout.decode("utf-8", "replace").split("\0") if item]


def github_metadata() -> dict[str, Any] | None:
    try:
        result = subprocess.run(
            [
                "gh",
                "repo",
                "view",
                "Mahmoud-Maher-Elsaid/TrustCXR",
                "--json",
                "nameWithOwner,isPrivate,defaultBranchRef,description,repositoryTopics",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, json.JSONDecodeError):
        return None


def is_future_or_historical(root: Path, path: Path, line: str) -> tuple[bool, bool]:
    rel = path.relative_to(root).as_posix()
    future = "research_extensions" in rel or "NOT IMPLEMENTED" in line.upper()
    historical = (
        rel.startswith(HISTORICAL_PREFIXES) or rel == "docs/execution/complete_stage_registry.json"
    )
    return future, historical


def machine_path_severity(rel: str) -> str:
    if rel in CURRENT_PUBLIC_DOCS or rel.startswith(CURRENT_DOC_PREFIXES):
        return "ERROR"
    if rel.startswith(CURRENT_RUNTIME_PREFIXES):
        return "ERROR"
    if rel.startswith("tests/"):
        return "ALLOWED_TEST_FIXTURE"
    if rel.startswith(HISTORICAL_PREFIXES) or rel == "docs/execution/complete_stage_registry.json":
        return "WARNING"
    return "WARNING"


def is_prohibited_context(lines: list[str], line_number: int) -> bool:
    """Recognize nearby prose that explicitly negates or prohibits a claim."""
    start = max(0, line_number - 12)
    context = " ".join(lines[start:line_number])
    return bool(
        re.search(
            r"must\s+never\s+be\s+described|prohibited\s+claims?|must\s+not|do\s+not|"
            r"does\s+not|not\s+(?:implemented|supported|available)|without\s+",
            context,
            re.I,
        )
    )


def audit(root: Path = ROOT) -> dict[str, Any]:
    findings: list[Finding] = []
    files = tracked_files(root)
    for path in files:
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8-sig")
        except (UnicodeDecodeError, OSError):
            continue
        rel = path.relative_to(root).as_posix()
        definition_file = rel in AUDITOR_DEFINITION_FILES
        test_fixture = rel.startswith("tests/")
        lines = text.splitlines()
        for number, line in enumerate(lines, 1):
            future, historical = is_future_or_historical(root, path, line)
            prohibited = is_prohibited_context(lines, number)
            if definition_file:
                continue
            if re.search(r"F:\\AI\\TrustCXR|C:\\Users\\[^\\\s]+", line, re.I):
                findings.append(
                    Finding(
                        machine_path_severity(rel),
                        "PRIVATE_MACHINE_PATH",
                        rel,
                        number,
                        "machine-specific path in tracked text",
                    )
                )
            if "..\\venv" in line or "../.venv" in line:
                findings.append(
                    Finding(
                        "ALLOWED_TEST_FIXTURE" if test_fixture else "ERROR",
                        "VENV_PATH_TYPO",
                        rel,
                        number,
                        "obsolete or incorrect virtual-environment path",
                    )
                )
            if re.search(
                r"current\s+stage\s*:\s*stage\s*1\b|Stage 1 - Repository Foundation", line, re.I
            ):
                findings.append(
                    Finding(
                        "ALLOWED_TEST_FIXTURE" if test_fixture else "ERROR",
                        "STALE_CURRENT_STAGE",
                        rel,
                        number,
                        "stale Stage 1 current-status claim",
                    )
                )
            extension_claim = (
                r"vision[- ]language system|implemented LLM|LLM report generation|implemented VLM|"
                r"(?:implements?|implemented|available|supports?)\s+Grad-CAM|Grad-CAM\s+is\s+implemented"
            )
            if re.search(extension_claim, line, re.I):
                qualified = r"not implemented|withheld|future|planned"
                if future or historical or prohibited or re.search(qualified, line, re.I):
                    findings.append(
                        Finding(
                            "ALLOWED_FUTURE_WORK"
                            if future
                            else ("ALLOWED_HISTORICAL" if historical else "INFO"),
                            "CAPABILITY_CONTEXT",
                            rel,
                            number,
                            "capability term is qualified",
                        )
                    )
                else:
                    findings.append(
                        Finding(
                            "ALLOWED_TEST_FIXTURE" if test_fixture else "ERROR",
                            "OVERCLAIMED_CAPABILITY",
                            rel,
                            number,
                            "unqualified extension capability claim",
                        )
                    )
            if re.search(r"trust(?:\s+|-)cxr", line, re.I) and rel in {
                "README.md",
                "CONTRIBUTING.md",
                "SECURITY.md",
            }:
                findings.append(
                    Finding(
                        "ALLOWED_TEST_FIXTURE" if test_fixture else "WARNING",
                        "BRAND_VARIANT",
                        rel,
                        number,
                        "non-canonical TrustCXR spelling",
                    )
                )
            extension_terms = r"pathology localization|grounded LLM|multimodal VLM"
            if "Grad-CAM" in line or re.search(extension_terms, line, re.I):
                findings.append(
                    Finding(
                        "ALLOWED_TEST_FIXTURE"
                        if test_fixture
                        else (
                            "ALLOWED_FUTURE_WORK"
                            if future
                            else ("ALLOWED_HISTORICAL" if historical else "INFO")
                        ),
                        "EXTENSION_TERM",
                        rel,
                        number,
                        "extension term retained with context",
                    )
                )

    for path in files:
        rel = path.relative_to(root).as_posix()
        name = path.name.lower()
        if re.search(
            r"(^|[_.-])(final_final|copy\d*|new_new|backup-temp|old2|tmp|scratch)([_.-]|$)", name
        ):
            findings.append(
                Finding(
                    "WARNING",
                    "SUSPICIOUS_PATH_NAME",
                    rel,
                    None,
                    "review path naming; historical evidence may justify retention",
                )
            )

    metadata = github_metadata()
    if metadata:
        description = metadata.get("description") or ""
        if re.search(
            r"vision[- ]language|\bLLM\b|\bVLM\b|true pathology localization", description, re.I
        ):
            findings.append(
                Finding(
                    "ERROR",
                    "GITHUB_OVERCLAIM",
                    "<github-about>",
                    None,
                    "About description overclaims an unimplemented extension",
                )
            )
    severities = (
        "ERROR",
        "WARNING",
        "INFO",
        "ALLOWED_HISTORICAL",
        "ALLOWED_FUTURE_WORK",
        "ALLOWED_TEST_FIXTURE",
    )
    counts = {
        severity: sum(item.severity == severity for item in findings) for severity in severities
    }
    return {
        "canonical": CANONICAL,
        "files_scanned": len(files),
        "github_metadata_available": metadata is not None,
        "findings": [asdict(item) for item in findings],
        "counts": counts,
        "errors": counts["ERROR"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    report = audit(args.root.resolve())
    counts = report["counts"]
    print("Project Naming Audit\n====================\n")
    print(f"Files scanned: {report['files_scanned']}")
    for key in (
        "ERROR",
        "WARNING",
        "INFO",
        "ALLOWED_HISTORICAL",
        "ALLOWED_FUTURE_WORK",
        "ALLOWED_TEST_FIXTURE",
    ):
        print(f"{key}: {counts[key]}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Project Naming Audit", "", f"Files scanned: {report['files_scanned']}", ""]
        lines.extend(
            f"- **{key}:** {counts[key]}"
            for key in (
                "ERROR",
                "WARNING",
                "INFO",
                "ALLOWED_HISTORICAL",
                "ALLOWED_FUTURE_WORK",
                "ALLOWED_TEST_FIXTURE",
            )
        )
        lines.extend(["", "## Findings", ""])
        for finding in report["findings"]:
            location = (
                f"{finding['path']}:{finding['line']}" if finding["line"] else finding["path"]
            )
            lines.append(
                "- **{severity}** `{rule}` — `{location}`: {message}".format(
                    severity=finding["severity"],
                    rule=finding["rule"],
                    location=location,
                    message=finding["message"],
                )
            )
        args.markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if args.strict and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
