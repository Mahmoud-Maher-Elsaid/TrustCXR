"""Fail-closed Candidate #2 identity, load-only, and synthetic smoke fast path."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
RUNTIME_COMMIT = "c5a778891ba0ddbd4cbb507c823f970595b1adc2"
CUDA_ROOT = Path(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.1")
RUNTIME_SOURCE = ROOT / "cache/research_extensions/ext4e_candidate2/llama_cpp_b8233"
RUNTIME_EXE = RUNTIME_SOURCE / "build_cuda/bin/Release/llama-server.exe"


def runtime_environment() -> dict[str, str]:
    env = dict(os.environ)
    paths = [
        str(RUNTIME_EXE.parent),
        str(CUDA_ROOT / "bin"),
        str(CUDA_ROOT / "bin" / "x64"),
    ]
    env["PATH"] = ";".join(paths + [env.get("PATH", "")])
    return env


def startup_probe() -> dict:
    if not RUNTIME_EXE.is_file():
        return {"passed": False, "classification": "CANDIDATE2_CUDA_RUNTIME_MISSING"}
    result = subprocess.run(
        [str(RUNTIME_EXE), "--help"],
        capture_output=True,
        text=True,
        check=False,
        env=runtime_environment(),
    )
    return {
        "passed": result.returncode == 0,
        "classification": None
        if result.returncode == 0
        else "CANDIDATE2_EXECUTABLE_STARTUP_FAILED",
        "exit_code": result.returncode,
        "stdout": result.stdout[-4000:],
        "stderr": result.stderr[-4000:],
        "path": runtime_environment()["PATH"],
        "inference_requests": 0,
    }


def classify_server_failure(log_path: Path, process: subprocess.Popen) -> str:
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
    if "invalid argument" in text:
        return "INVALID_SERVER_ARGUMENT"
    if "out of memory" in text.lower() or "cuda" in text.lower() and "alloc" in text.lower():
        return "GPU_OOM"
    if process.poll() is not None:
        return "SERVER_EXITED_DURING_LOAD"
    return "MODEL_LOAD_TIMEOUT"


def build_server_argv(model: Path, port: int = 18080) -> list[str]:
    """The sole authoritative b8233 server argv construction path."""

    return [
        str(RUNTIME_EXE),
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--ctx-size",
        "2048",
        "--n-gpu-layers",
        "999",
        "--parallel",
        "1",
    ]


def runtime_source_identity() -> dict:
    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-c", "safe.directory=*", "-C", str(RUNTIME_SOURCE), *args],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError("CANDIDATE2_RUNTIME_SOURCE_GIT_FAILURE")
        return result.stdout.strip()

    return {
        "source_path": str(RUNTIME_SOURCE),
        "commit_actual": git("rev-parse", "HEAD"),
        "tag_actual": git("tag", "--points-at", "HEAD"),
    }


def write_preflight_diagnostic(config: dict, model: Path, evidence: Path) -> int:
    source = runtime_source_identity()
    branch = subprocess.run(
        ["git", "-C", str(ROOT), "branch", "--show-current"],
        capture_output=True,
        text=True,
        check=False,
    )
    nvcc = CUDA_ROOT / "bin" / "nvcc.exe"
    dlls = sorted(p.name for p in RUNTIME_EXE.parent.glob("*.dll"))
    required_dlls = {"ggml.dll", "ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "llama.dll"}
    missing_dlls = sorted(required_dlls - set(dlls))
    probe = startup_probe()
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    diagnostic = {
        "repository_branch": branch.stdout.strip(),
        "model_exists": model.is_file(),
        "model_size_actual": model.stat().st_size if model.is_file() else None,
        "model_size_expected": 5198911904,
        "model_sha256_actual": sha256_file(model) if model.is_file() else None,
        "model_sha256_expected": "33e7a72cf5e6e2cfc2f2847075acc013d68bba023e35310cef86b5cf8fdca761",
        "runtime_source_exists": RUNTIME_SOURCE.is_dir(),
        "runtime_tag_actual": source["tag_actual"],
        "runtime_tag_expected": "b8233",
        "runtime_commit_actual": source["commit_actual"],
        "runtime_commit_expected": RUNTIME_COMMIT,
        "cuda_root_exists": CUDA_ROOT.is_dir(),
        "nvcc_exists_absolute": nvcc.is_file(),
        "nvcc_path": str(nvcc),
        "cuda_server_exists": RUNTIME_EXE.is_file(),
        "cuda_server_sha256_current": sha256_file(RUNTIME_EXE) if RUNTIME_EXE.is_file() else None,
        "cuda_server_directory": str(RUNTIME_EXE.parent),
        "required_runtime_dlls": dlls,
        "runtime_environment_ready": RUNTIME_EXE.is_file() and not missing_dlls,
        "runtime_search_path": [
            str(RUNTIME_EXE.parent),
            str(CUDA_ROOT / "bin"),
            str(CUDA_ROOT / "bin" / "x64"),
        ],
        "missing_runtime_dlls": missing_dlls,
        "startup_probe": probe,
        "server_argv": build_server_argv(model),
        "gpu_detected": gpu.returncode == 0,
        "gpu_identity": gpu.stdout.strip(),
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
        "status": "PREFLIGHT_PASS",
    }
    checks = (
        diagnostic["model_exists"]
        and diagnostic["model_size_actual"] == diagnostic["model_size_expected"]
        and diagnostic["model_sha256_actual"] == diagnostic["model_sha256_expected"]
        and diagnostic["runtime_source_exists"]
        and diagnostic["runtime_tag_expected"] in diagnostic["runtime_tag_actual"]
        and diagnostic["runtime_commit_actual"] == diagnostic["runtime_commit_expected"]
        and diagnostic["cuda_root_exists"]
        and diagnostic["nvcc_exists_absolute"]
        and diagnostic["cuda_server_exists"]
        and diagnostic["gpu_detected"]
        and not missing_dlls
        and probe["passed"]
    )
    if not checks:
        diagnostic["status"] = "PREFLIGHT_FAILED"
        diagnostic["failure_classification"] = (
            probe.get("classification")
            if not probe.get("passed")
            else "CANDIDATE2_PREFLIGHT_FAILURE"
        )
    (evidence / "preflight_diagnostic.json").write_text(
        json.dumps(diagnostic, indent=2), encoding="utf-8"
    )
    if not checks:
        raise RuntimeError("CANDIDATE2_PREFLIGHT_FAILED")
    return 0


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
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    if config["repository"] != REPOSITORY or config["filename"] != FILENAME:
        raise RuntimeError("Candidate #2 identity contract mismatch.")
    if config["revision"] != "0102285ad796bd99af90f58de616092e5630e970":
        raise RuntimeError("CANDIDATE2_MODEL_REVISION_MISMATCH")
    evidence = ROOT / "artifacts/research_extensions/ext4e_candidate2"
    model = ROOT / "cache/research_extensions/ext4e_candidate2/models" / FILENAME
    evidence.mkdir(parents=True, exist_ok=True)
    if args.preflight_only:
        return write_preflight_diagnostic(config, model, evidence)
    metadata = official_metadata()
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
    runtime = RUNTIME_EXE
    if not runtime.is_file():
        raise RuntimeError("CANDIDATE2_CUDA_RUNTIME_MISSING")
    nvcc = CUDA_ROOT / "bin" / "nvcc.exe"
    if not nvcc.is_file():
        raise RuntimeError("CANDIDATE2_CUDA_TOOLKIT_MISSING")
    identity.update(
        {
            "runtime_release": RUNTIME_RELEASE,
            "runtime_commit": RUNTIME_COMMIT,
            "cuda_root": str(CUDA_ROOT),
            "nvcc_path": str(nvcc),
            "cuda_server_sha256": sha256_file(runtime),
        }
    )
    (evidence / "candidate2_identity.json").write_text(
        json.dumps(identity, indent=2), encoding="utf-8"
    )
    probe = startup_probe()
    if not probe["passed"]:
        raise RuntimeError(probe["classification"])
    server_argv = build_server_argv(model)
    (evidence / "server_argv.json").write_text(json.dumps(server_argv, indent=2), encoding="utf-8")
    server = subprocess.Popen(
        server_argv,
        stdout=(evidence / "llama_server.log").open("w"),
        stderr=subprocess.STDOUT,
        text=True,
        env=runtime_environment(),
    )
    try:
        timeout_seconds = 180
        deadline = time.monotonic() + timeout_seconds
        health_last_status = None
        while time.monotonic() < deadline:
            if server.poll() is not None:
                raise RuntimeError(classify_server_failure(evidence / "llama_server.log", server))
            try:
                with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as response:
                    health_last_status = response.status
                    if response.status == 200:
                        break
            except urllib.error.HTTPError as error:
                health_last_status = error.code
                time.sleep(1)
            except (urllib.error.URLError, TimeoutError):
                time.sleep(1)
        else:
            diagnostic = {
                "server_pid": server.pid,
                "server_alive_at_timeout": server.poll() is None,
                "server_exit_code": server.poll(),
                "elapsed_seconds": timeout_seconds,
                "configured_timeout_seconds": timeout_seconds,
                "health_last_status": health_last_status,
                "failure_classification": classify_server_failure(
                    evidence / "llama_server.log", server
                ),
                "inference_requests": 0,
                "development_cases_accessed": 0,
                "final_cases_accessed": 0,
                "locked_test_accessed": False,
            }
            (evidence / "load_failure_diagnostic.json").write_text(
                json.dumps(diagnostic, indent=2), encoding="utf-8"
            )
            raise RuntimeError(diagnostic["failure_classification"])
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
