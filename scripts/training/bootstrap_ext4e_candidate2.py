"""Fail-closed Candidate #2 identity, load-only, and synthetic smoke fast path."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPOSITORY = "mistralai/Ministral-3-8B-Instruct-2512-GGUF"
FILENAME = "Ministral-3-8B-Instruct-2512-Q4_K_M.gguf"
RUNTIME_RELEASE = "b8233"
RUNTIME_COMMIT_PREFIX = "c5a7788"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def gguf_header(path: Path) -> bool:
    with path.open("rb") as stream:
        return stream.read(4) == b"GGUF"


def official_metadata() -> dict:
    url = f"https://huggingface.co/api/models/{REPOSITORY}"
    with urllib.request.urlopen(url, timeout=30) as response:
        data = json.loads(response.read())
    revision = data.get("sha")
    if not isinstance(revision, str) or len(revision) < 7:
        raise RuntimeError("Candidate #2 revision was not resolved from official metadata.")
    siblings = {item.get("rfilename") for item in data.get("siblings", [])}
    if FILENAME not in siblings:
        raise RuntimeError("Pinned Candidate #2 GGUF filename is absent from official metadata.")
    return {
        "revision": revision,
        "license": data.get("cardData", {}).get("license") or data.get("license"),
    }


def download_model(path: Path, revision: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{REPOSITORY}/resolve/{revision}/{FILENAME}?download=true"
    temporary = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as source, temporary.open("wb") as target:
        while chunk := source.read(1024 * 1024):
            target.write(chunk)
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default="configs/research_extensions/ext4e_candidate2_ministral.json"
    )
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if config["repository"] != REPOSITORY or config["filename"] != FILENAME:
        raise RuntimeError("Candidate #2 identity contract mismatch.")
    if config["revision"] != "MUST_BE_RESOLVED_FROM_OFFICIAL_HF_METADATA":
        raise RuntimeError("Candidate #2 revision must be resolved at bootstrap time.")
    metadata = official_metadata()
    evidence = ROOT / "artifacts/research_extensions/ext4e_candidate2"
    model = ROOT / "cache/research_extensions/ext4e_candidate2/models" / FILENAME
    evidence.mkdir(parents=True, exist_ok=True)
    identity = {
        "candidate_id": 2,
        "model_family": config["model_family"],
        "model_variant": config["model_variant"],
        "repository": REPOSITORY,
        "resolved_revision": metadata["revision"],
        "filename": FILENAME,
        "quantization": "Q4_K_M",
        "license": metadata["license"],
        "timestamp": datetime.now(UTC).isoformat(),
        "source": "official_huggingface_metadata",
    }
    (evidence / "candidate2_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    if not model.is_file():
        download_model(model, metadata["revision"])
    if not gguf_header(model):
        raise RuntimeError("Candidate #2 artifact is not GGUF.")
    identity.update({"sha256": sha256_file(model), "file_size_bytes": model.stat().st_size})
    (evidence / "candidate2_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    runtime = (
        ROOT
        / "cache/research_extensions/ext4e_candidate2/llama_cpp_b8233"
        / "build_cuda/bin/Release/llama-server.exe"
    )
    if not runtime.is_file():
        raise RuntimeError("PINNED_RUNTIME_INCOMPATIBLE_WITH_CANDIDATE2")
    version = subprocess.run(
        [str(runtime), "--version"], capture_output=True, text=True, check=False
    )
    identity_text = version.stdout + version.stderr
    if (
        version.returncode != 0
        or "8233" not in identity_text
        or RUNTIME_COMMIT_PREFIX not in identity_text
    ):
        raise RuntimeError("PINNED_RUNTIME_INCOMPATIBLE_WITH_CANDIDATE2")
    server = subprocess.Popen(
        [
            str(runtime),
            "--model",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            "18080",
            "--parallel",
            "1",
            "--ctx-size",
            "2048",
            "--n-gpu-layers",
            "999",
            "--cors-origins",
            "localhost",
            "--no-webui",
            "--reasoning",
            "off",
        ],
        stdout=(evidence / "llama_server.log").open("w"),
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as response:
                    if response.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            raise RuntimeError("LOAD_TIMEOUT")
        (evidence / "load_only_metadata.json").write_text(
            json.dumps(
                {
                    "status": "LOAD_ONLY_PASS",
                    "model_sha256": identity["sha256"],
                    "file_size_bytes": identity["file_size_bytes"],
                    "runtime_release": RUNTIME_RELEASE,
                    "runtime_commit_prefix": RUNTIME_COMMIT_PREFIX,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        schema = {
            "type": "object",
            "properties": {
                "status": {"const": "PASS"},
                "message": {"const": "structured output compatibility"},
            },
            "required": ["status", "message"],
            "additionalProperties": False,
        }
        payload = {
            "model": FILENAME,
            "messages": [{"role": "user", "content": "Return the required synthetic object."}],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260806,
            "max_tokens": 768,
            "stream": False,
            "reasoning_effort": "none",
            "response_format": {"type": "json_object", "schema": schema},
        }
        (evidence / "synthetic_request.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        request = urllib.request.Request(
            "http://127.0.0.1:18080/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            raw = response.read()
        (evidence / "raw_http_response.json").write_bytes(raw)
        envelope = json.loads(raw)
        content = envelope["choices"][0]["message"]["content"]
        (evidence / "raw_model_content.txt").write_text(content, encoding="utf-8")
        parsed = json.loads(content)
        (evidence / "parsed_output.json").write_text(json.dumps(parsed, indent=2), encoding="utf-8")
        if parsed != {"status": "PASS", "message": "structured output compatibility"}:
            raise RuntimeError("SYNTHETIC_SCHEMA_VALIDATION_FAILURE")
        (evidence / "synthetic_validation.json").write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "request_count": 1,
                    "retry_count": 0,
                    "development_cases_accessed": 0,
                    "frozen_final_cases_accessed": 0,
                    "locked_test_accessed": False,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (evidence / "run_metadata.json").write_text(
            json.dumps(
                {
                    "status": "CANDIDATE2_FAST_BOOTSTRAP_PASS",
                    "generation_completed": True,
                    "cleanup": True,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return 0
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()


if __name__ == "__main__":
    raise SystemExit(main())
