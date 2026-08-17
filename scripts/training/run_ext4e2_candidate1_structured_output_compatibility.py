"""Run one synthetic request-level structured-output compatibility smoke."""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def request_json(url: str, payload: dict, timeout: int) -> bytes:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            json.dumps(
                {
                    "status_code": exc.code,
                    "reason": exc.reason,
                    "body": body,
                    "endpoint": url,
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                sort_keys=True,
            )
        ) from exc


def main() -> int:
    config = json.loads(
        (
            ROOT / "configs/research_extensions/ext4e2d0_structured_output_compatibility.json"
        ).read_text(encoding="utf-8")
    )
    model_path = ROOT / "cache/research_extensions/ext4e2/models/Qwen3-8B-Q4_K_M.gguf"
    server_path = ROOT / "cache/research_extensions/ext4e2/llama.cpp/llama-server.exe"
    if not model_path.is_file() or not server_path.is_file():
        raise RuntimeError("Pinned model or llama-server executable is missing.")
    if sha256_file(model_path) != config["model_sha256"]:
        raise RuntimeError("Pinned model SHA-256 mismatch.")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / config["evidence_root"] / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    schema = config["synthetic_schema"]
    (run_root / "synthetic_schema.json").write_text(
        json.dumps(schema, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    log_path = run_root / "llama_server.log"
    server_args = [
        str(server_path),
        "--model",
        str(model_path),
        "--ctx-size",
        "2048",
        "--n-gpu-layers",
        "999",
        "--host",
        "127.0.0.1",
        "--port",
        "18080",
        "--cors-origins",
        "localhost",
        "--no-webui",
        "--parallel",
        "1",
        "--reasoning",
        "off",
    ]
    started = time.monotonic()
    process = subprocess.Popen(server_args, stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    request_count = 0
    status = "MODEL_SERVER_FAILURE"
    generation_started = False
    generation_completed = False
    raw_response: bytes | None = None
    error: str | None = None
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("llama-server exited before readiness.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:18080/health", timeout=2) as health:
                    if health.status == 200:
                        break
            except (urllib.error.URLError, TimeoutError):
                time.sleep(2)
        else:
            status = "MODEL_TIMEOUT"
            raise RuntimeError("llama-server readiness timeout.")

        payload = {
            "model": config["model_filename"],
            "messages": [
                {
                    "role": "system",
                    "content": "Return only the requested synthetic compatibility object.",
                },
                {"role": "user", "content": "Return the required object."},
            ],
            "temperature": config["generation"]["temperature"],
            "top_p": config["generation"]["top_p"],
            "seed": config["generation"]["seed"],
            "max_tokens": config["generation"]["max_tokens"],
            "stream": False,
            "reasoning_effort": config["request_reasoning_effort"],
            "response_format": {"type": "json_schema", "schema": schema},
        }
        (run_root / "request.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        request_count = 1
        raw_response = request_json(config["endpoint"], payload, 180)
        generation_started = True
        response = json.loads(raw_response)
        content = response["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        if parsed != {"message": "structured output compatibility", "status": "PASS"}:
            raise RuntimeError("Synthetic structured output failed deterministic validation.")
        generation_completed = True
        status = "STRUCTURED_OUTPUT_COMPATIBILITY_PASS"
        (run_root / "raw_response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_root / "parsed_content.json").write_text(
            json.dumps(parsed, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except Exception as exc:
        error = str(exc)
        if status == "MODEL_SERVER_FAILURE":
            status = "STRUCTURED_OUTPUT_COMPATIBILITY_FAILURE"
    finally:
        process.terminate()
        try:
            process.wait(timeout=10)
            cleanup = process.returncode is not None
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)
            cleanup = False
        evidence = {
            "status": status,
            "run_id": run_id,
            "runtime_release": config["runtime_release"],
            "runtime_commit_prefix": config["runtime_commit_prefix"],
            "model_filename": config["model_filename"],
            "model_sha256": config["model_sha256"],
            "server_arguments": server_args[1:],
            "endpoint": config["endpoint"],
            "structured_output_mechanism": config["structured_output_mechanism"],
            "response_format_type": config["response_format_type"],
            "schema_sha256": canonical_sha256(schema),
            "request_reasoning_effort": config["request_reasoning_effort"],
            "request_count": request_count,
            "retry_count": 0,
            "generation_started": generation_started,
            "generation_completed": generation_completed,
            "error": error,
            "server_cleanup_confirmed": cleanup,
            "latency_seconds": round(time.monotonic() - started, 3),
            "development_cases_accessed": 0,
            "frozen_final_cases_accessed": 0,
            "locked_test_accessed": False,
            "patient_or_medical_payload": False,
        }
        (run_root / "structured_output_compatibility.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if status == "STRUCTURED_OUTPUT_COMPATIBILITY_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
