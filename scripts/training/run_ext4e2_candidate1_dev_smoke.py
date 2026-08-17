"""Run exactly one local EXT-4E2D development-case inference attempt."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trustcxr.grounded_llm.benchmark import score_case  # noqa: E402
from trustcxr.grounded_llm.contracts import (  # noqa: E402
    GroundedOutputEnvelope,
    build_synthetic_case,
)
from trustcxr.grounded_llm.partition_guard import validate_development_partition  # noqa: E402


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    return sha256_bytes(payload.encode("utf-8"))


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class HttpRequestFailure(RuntimeError):
    """A non-successful local inference request with preserved response evidence."""

    def __init__(self, endpoint: str, status_code: int, reason: str, body: str) -> None:
        self.endpoint = endpoint
        self.status_code = status_code
        self.reason = reason
        self.body = body
        self.timestamp = datetime.now(UTC).isoformat()
        super().__init__(f"HTTP {status_code} {reason} from {endpoint}: {body}")

    def as_dict(self) -> dict:
        return {
            "endpoint": self.endpoint,
            "status_code": self.status_code,
            "reason": self.reason,
            "body": self.body,
            "timestamp": self.timestamp,
        }


def request_json(url: str, payload: dict, timeout: int) -> bytes:
    body = json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        raise HttpRequestFailure(url, exc.code, exc.reason, response_body) from exc


def main() -> int:
    config = load_json(ROOT / "configs/research_extensions/ext4e2d_candidate1_dev_smoke.json")
    prompt_path = ROOT / config["prompt_template"]
    prompt_bytes = prompt_path.read_bytes()
    if sha256_bytes(prompt_bytes) != config["prompt_sha256"]:
        raise RuntimeError("Prompt template SHA-256 mismatch.")
    if config["partition"] != "development" or config["case_id"] != "dev_supported":
        raise RuntimeError("EXT-4E2D is restricted to the frozen development case.")
    selected = validate_development_partition(
        "development",
        ({"case_id": config["case_id"], "category": config["case_category"]},),
    )
    if len(selected) != 1:
        raise RuntimeError("Exactly one development case is required.")

    envelope = build_synthetic_case("supported")
    input_payload = envelope.model_dump(mode="json")
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = ROOT / config["evidence_root"] / run_id
    run_root.mkdir(parents=True, exist_ok=False)
    (run_root / "input_envelope.json").write_text(
        json.dumps(input_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_root / "prompt_template.txt").write_bytes(prompt_bytes)
    schema_path = run_root / "ext4c_output_schema.json"
    schema_path.write_text(
        json.dumps(GroundedOutputEnvelope.model_json_schema(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    model_path = ROOT / "cache/research_extensions/ext4e2/models/Qwen3-8B-Q4_K_M.gguf"
    server_path = ROOT / "cache/research_extensions/ext4e2/llama.cpp/llama-server.exe"
    if not model_path.is_file() or not server_path.is_file():
        raise RuntimeError("Pinned model or llama-server executable is missing.")
    if sha256_file(model_path) != config["model_sha256"]:
        raise RuntimeError("Pinned model SHA-256 mismatch.")

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
        "--no-prefill-assistant",
        "--json-schema-file",
        str(schema_path),
    ]
    started = time.monotonic()
    process = subprocess.Popen(server_args, stdout=log_path.open("w"), stderr=subprocess.STDOUT)
    request_count = 0
    status = "MODEL_SERVER_FAILURE"
    raw_response: bytes | None = None
    request_error: dict | None = None
    generation_started = False
    generation_completed = False
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

        output_payload = {
            "messages": [
                {"role": "system", "content": prompt_bytes.decode("utf-8")},
                {"role": "user", "content": json.dumps(input_payload, sort_keys=True)},
            ],
            "temperature": 0.0,
            "top_p": 1.0,
            "seed": 20260806,
            "max_tokens": 768,
            "stream": False,
        }
        request_count = 1
        raw_response = request_json(
            "http://127.0.0.1:18080/v1/chat/completions", output_payload, 180
        )
        generation_started = True
        response = json.loads(raw_response)
        choice = response["choices"][0]["message"]
        if choice.get("reasoning_content") or response.get("reasoning_content"):
            status = "SAFETY_GATE_FAILURE"
            raise RuntimeError("Reasoning content was returned despite non-thinking mode.")
        candidate = json.loads(choice["content"])
        validated = GroundedOutputEnvelope.model_validate(candidate)
        references = {item.evidence_id: item.status.value for item in envelope.evidence_items}
        if any(ref.evidence_id not in references for ref in validated.evidence_references):
            raise RuntimeError("Output references evidence outside the supplied envelope.")
        case = {
            "case_id": "dev_supported",
            "category": "COMPLETE_SUPPORTED_EVIDENCE",
            "grounding_kind": "supported",
            "expected_statuses": ["COMPLETED"],
        }
        score = score_case(case, candidate)
        status = "DEV_CASE_SMOKE_PASS" if score["case_passed"] else "SAFETY_GATE_FAILURE"
        generation_completed = True
        (run_root / "raw_response.json").write_text(
            json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_root / "parsed_output.json").write_text(
            json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        (run_root / "score.json").write_text(
            json.dumps(score, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except HttpRequestFailure as exc:
        status = "MODEL_REQUEST_FAILURE"
        request_error = exc.as_dict()
        (run_root / "http_error.json").write_text(
            json.dumps(request_error, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise RuntimeError(
            f"HTTP request failed with {exc.status_code} {exc.reason}; response body preserved."
        ) from exc
    except json.JSONDecodeError as exc:
        status = "STRUCTURED_OUTPUT_INVALID"
        raise RuntimeError("Model response was not valid JSON.") from exc
    except (KeyError, TypeError, ValueError) as exc:
        status = "GROUNDING_VALIDATION_FAILURE"
        raise RuntimeError("Model response failed EXT-4C/grounding validation.") from exc
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
            "case_id": "dev_supported",
            "case_category": "COMPLETE_SUPPORTED_EVIDENCE",
            "partition": "development",
            "input_envelope_sha256": canonical_sha256(input_payload),
            "prompt_sha256": config["prompt_sha256"],
            "model_sha256": config["model_sha256"],
            "runtime_release": config["runtime_release"],
            "runtime_commit_prefix": config["runtime_commit_prefix"],
            "generation": config["generation"],
            "inference_request_count": request_count,
            "retry_count": 0,
            "inference_request_attempted": request_count == 1,
            "generation_started": generation_started,
            "generation_completed": generation_completed,
            "http_error": request_error,
            "development_cases_accessed": 1,
            "frozen_final_cases_accessed": 0,
            "locked_test_accessed": False,
            "server_cleanup_confirmed": cleanup,
            "latency_seconds": round(time.monotonic() - started, 3),
            "reasoning_content_stored": False,
        }
        (run_root / "run_metadata.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return 0 if status == "DEV_CASE_SMOKE_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
