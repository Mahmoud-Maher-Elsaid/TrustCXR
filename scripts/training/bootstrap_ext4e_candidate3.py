"""Zero-inference Candidate #3 identity/runtime bootstrap preflight."""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json"
MODEL_ROOT = ROOT / "cache/research_extensions/ext4e_candidate3/models"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: dict) -> None:
    if config["candidate_id"] != 3:
        raise RuntimeError("CANDIDATE3_IDENTITY_MISMATCH")
    if config["repository"] != "microsoft/Phi-4-mini-instruct":
        raise RuntimeError("CANDIDATE3_REPOSITORY_MISMATCH")
    if config["resolved_revision"] != "cfbefacb99257ffa30c83adab238a50856ac3083":
        raise RuntimeError("CANDIDATE3_REVISION_NOT_IMMUTABLE")
    if config["partitions"]["development_cases_accessed"] != 0:
        raise RuntimeError("CANDIDATE3_DEVELOPMENT_ACCESS_FORBIDDEN")
    if config["partitions"]["final_cases_accessed"] != 0:
        raise RuntimeError("CANDIDATE3_FINAL_ACCESS_FORBIDDEN")
    if config["partitions"]["locked_test_accessed"] is not False:
        raise RuntimeError("CANDIDATE3_LOCKED_TEST_FORBIDDEN")
    if config["generation"]["request_count"] != 1 or config["generation"]["retry_count"] != 0:
        raise RuntimeError("CANDIDATE3_GENERATION_POLICY_MISMATCH")


def validate_runtime(config: dict) -> dict:
    try:
        transformers = importlib.import_module("transformers")
        torch = importlib.import_module("torch")
        phi3 = transformers.Phi3ForCausalLM
    except (ImportError, AttributeError) as exc:
        raise RuntimeError("CANDIDATE3_TRANSFORMERS_RUNTIME_UNAVAILABLE") from exc
    return {
        "kind": "transformers",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "phi3_class_available": phi3.__name__ == "Phi3ForCausalLM",
        "model_load_performed": False,
        "inference_requests": 0,
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    runtime = validate_runtime(config)
    artifact_records = []
    for artifact in config["artifacts"]:
        path = MODEL_ROOT / artifact["filename"]
        record = {
            "filename": artifact["filename"],
            "path": str(path),
            "exists": path.is_file(),
            "size_bytes": path.stat().st_size if path.is_file() else None,
            "sha256": sha256_file(path) if path.is_file() else None,
            "expected_sha256": artifact["sha256"],
        }
        if path.is_file() and artifact["sha256"] and record["sha256"] != artifact["sha256"]:
            raise RuntimeError("CANDIDATE3_MODEL_SHA256_MISMATCH")
        artifact_records.append(record)
    report = {
        "status": "CANDIDATE3_IDENTITY_PREFLIGHT_READY"
        if all(item["exists"] for item in artifact_records)
        else "CANDIDATE3_DOWNLOAD_REQUIRED",
        "candidate_id": 3,
        "config": str(CONFIG_PATH),
        "runtime": runtime,
        "artifacts": artifact_records,
        "development_cases_accessed": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
        "inference_requests": 0,
        "next_action": "DOWNLOAD_OFFICIAL_ARTIFACTS_THEN_REEXECUTE_PREFLIGHT",
    }
    evidence = ROOT / "artifacts/research_extensions/ext4e_candidate3"
    evidence.mkdir(parents=True, exist_ok=True)
    (evidence / "identity_preflight.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "CANDIDATE3_IDENTITY_PREFLIGHT_READY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
