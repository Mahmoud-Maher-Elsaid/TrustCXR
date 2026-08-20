from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from packaging.utils import canonicalize_name

FINAL_LOCK = Path("requirements/lock-final-research-windows-cu130.txt")
FINAL_LOCK_SHA256 = "cc63ac8bfb8dd6cc0f15469c4e7dfd6f620ec3747931ebd63c85fb11a8dc0786"
EXPECTED_PYTHON = "3.12.10"
EXPECTED_CUDA = "13.0"
EXPECTED_GPU = "NVIDIA GeForce RTX 3070 Ti Laptop GPU"
BOOTSTRAP_DISTRIBUTIONS = {"pip": "26.2", "setuptools": "78.1.0", "wheel": "0.47.0"}
EXT4H_GPU_RUNTIME_MANIFEST = Path(
    "configs/research_extensions/ext4h/gpu_runtime_dependencies_v1.json"
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_lock(path: Path) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s]+)", line.strip())
        if not match:
            continue
        name, version = match.groups()
        canonical = canonicalize_name(name)
        if canonical in pins and pins[canonical] != version:
            raise RuntimeError(f"Conflicting final-lock pin: {canonical}")
        pins[canonical] = version
    return pins


def tracked_import_roots(root: Path) -> set[str]:
    imports: set[str] = set()
    for source_root in (root / "src", root / "scripts"):
        for path in source_root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imports.add(node.module.split(".")[0])
    return imports


def scoped_extension_import_roots(root: Path) -> set[str]:
    """Imports governed by an explicit research-extension runtime manifest."""
    imports: set[str] = set()
    scoped_paths = [root / "src/trustcxr/grounded_llm", root / "scripts/research_extensions"]
    for source_root in scoped_paths:
        for path in (
            source_root.glob("ext4h*.py")
            if source_root.name == "grounded_llm"
            else source_root.glob("*.py")
        ):
            tree = ast.parse(path.read_text(encoding="utf-8-sig"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    imports.add(node.module.split(".")[0])
    return imports


def audit_imports(root: Path, pins: dict[str, str]) -> dict[str, Any]:
    package_map = importlib.metadata.packages_distributions()
    local_roots = {"trustcxr", "scripts"}
    local_roots.update(path.stem for path in (root / "scripts").rglob("*.py"))
    third_party: dict[str, list[str]] = {}
    ungoverned: list[str] = []
    scoped_manifest_path = root / EXT4H_GPU_RUNTIME_MANIFEST
    scoped_manifest: dict[str, str] = {}
    if scoped_manifest_path.is_file():
        payload = json.loads(scoped_manifest_path.read_text(encoding="utf-8"))
        scoped_manifest = {
            canonicalize_name(name): version
            for name, version in payload.get("dependencies", {}).items()
        }
    scoped_imports = scoped_extension_import_roots(root)
    for imported in sorted(tracked_import_roots(root)):
        if imported in sys.stdlib_module_names or imported in local_roots:
            continue
        distributions = package_map.get(imported, [])
        governed = sorted(
            name for name in {canonicalize_name(value) for value in distributions} if name in pins
        )
        if not governed and imported in scoped_imports:
            distribution = canonicalize_name(imported)
            if distribution in scoped_manifest:
                actual = importlib.metadata.version(distribution)
                if actual == scoped_manifest[distribution]:
                    governed = [f"EXT4H_SCOPED:{distribution}=={actual}"]
        if not governed:
            ungoverned.append(imported)
        else:
            third_party[imported] = governed
    return {
        "tracked_third_party_import_roots": third_party,
        "ungoverned_runtime_imports": ungoverned,
        "passed": not ungoverned,
    }


def verify(root: Path) -> dict[str, Any]:
    lock = root / FINAL_LOCK
    if sha256(lock) != FINAL_LOCK_SHA256:
        raise RuntimeError("Final environment lock SHA-256 mismatch.")
    pins = parse_lock(lock)
    if len(pins) != 53:
        raise RuntimeError("Final environment dependency count mismatch.")
    mismatches = []
    for name, expected in sorted({**pins, **BOOTSTRAP_DISTRIBUTIONS}.items()):
        actual = importlib.metadata.version(name)
        if actual != expected:
            mismatches.append({"distribution": name, "expected": expected, "actual": actual})
    if mismatches:
        raise RuntimeError(f"Installed distribution mismatch: {mismatches[0]['distribution']}")
    if platform.python_version() != EXPECTED_PYTHON:
        raise RuntimeError("Python version mismatch.")

    import torch

    if torch.version.cuda != EXPECTED_CUDA:
        raise RuntimeError("PyTorch CUDA runtime mismatch.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    if torch.cuda.get_device_name(0) != EXPECTED_GPU:
        raise RuntimeError("Validated GPU model mismatch.")
    import_audit = audit_imports(root, pins)
    if not import_audit["passed"]:
        raise RuntimeError(
            f"Ungoverned runtime import: {import_audit['ungoverned_runtime_imports'][0]}"
        )
    return {
        "status": "PASSED_FINAL_ENVIRONMENT_VERIFICATION",
        "canonical_lock": FINAL_LOCK.as_posix(),
        "canonical_lock_sha256": FINAL_LOCK_SHA256,
        "locked_dependencies": len(pins),
        "bootstrap_distributions": BOOTSTRAP_DISTRIBUTIONS,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": importlib.metadata.version("torch"),
        "torchvision": importlib.metadata.version("torchvision"),
        "cuda_runtime": torch.version.cuda,
        "cuda_available": True,
        "gpu": torch.cuda.get_device_name(0),
        "conflicting_pins": [],
        "import_audit": import_audit,
        "model_inference_performed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    print(json.dumps(verify(args.project_root.resolve()), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
