"""Execute the governed five-case remainder of the EXT-4E1 development run."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from trustcxr.grounded_llm.contracts import (  # noqa: E402
    GroundedOutputEnvelope,
    build_synthetic_case,
)
from trustcxr.grounded_llm.development_evaluation import (  # noqa: E402
    aggregate_evidence,
    build_evaluation_plan,
    scoring_case,
    sha256_file,
)


def _load_runner():
    path = ROOT / "scripts/training/run_ext4e2_candidate1_dev_smoke.py"
    spec = importlib.util.spec_from_file_location("ext4e2d_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


CASES_PATH = ROOT / "tests/fixtures/ext4d_benchmark_cases.json"
CONFIG_PATH = ROOT / "configs/research_extensions/ext4e2d_candidate1_dev_smoke.json"
HISTORICAL_ROOT = ROOT / "artifacts/research_extensions/ext4e2_candidate1/dev_case_smoke"
MODEL_PATH = ROOT / "cache/research_extensions/ext4e2/models/Qwen3-8B-Q4_K_M.gguf"
SERVER_PATH = ROOT / "cache/research_extensions/ext4e2/llama.cpp/llama-server.exe"


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    runner = _load_runner()
    config = runner.load_json(CONFIG_PATH)
    runner.validate_config(config)
    if not MODEL_PATH.is_file() or not SERVER_PATH.is_file():
        raise RuntimeError("Pinned model or llama-server executable is missing.")
    if sha256_file(MODEL_PATH) != config["model_sha256"]:
        raise RuntimeError("Pinned model SHA-256 mismatch.")

    plan = build_evaluation_plan(CASES_PATH, CONFIG_PATH, HISTORICAL_ROOT)
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))["development_cases"]
    by_id = {case["case_id"]: case for case in cases}
    remaining = plan["remaining_case_ids"]
    if len(remaining) != 5 or "dev_supported" in remaining:
        raise RuntimeError("The governed remaining-case set is invalid.")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    evaluation_root = (
        ROOT
        / config.get(
            "evaluation_artifact_root",
            "artifacts/research_extensions/ext4e_candidate1/development_evaluation",
        )
        / run_id
    )
    evaluation_root.mkdir(parents=True, exist_ok=False)
    prompt_path = ROOT / config["prompt_template"]
    prompt_bytes = prompt_path.read_bytes()
    if runner.sha256_bytes(prompt_bytes) != config["prompt_sha256"]:
        raise RuntimeError("Prompt template SHA-256 mismatch.")
    schema = GroundedOutputEnvelope.model_json_schema()
    _write_json(evaluation_root / "evaluation_plan.json", plan)
    (evaluation_root / "prompt_template.txt").write_bytes(prompt_bytes)
    _write_json(evaluation_root / "ext4c_output_schema.json", schema)

    server = config["server"]
    server_args = [
        str(SERVER_PATH),
        "--model",
        str(MODEL_PATH),
        "--ctx-size",
        str(server["context_size"]),
        "--n-gpu-layers",
        str(server["gpu_layers"]),
        "--host",
        server["host"],
        "--port",
        str(server["port"]),
        "--cors-origins",
        server["cors_origins"],
        "--no-webui",
        "--parallel",
        str(server["parallel_slots"]),
        "--reasoning",
        server["reasoning"],
    ]
    process = None
    evidence_paths = {"dev_supported": HISTORICAL_ROOT / "20260817T091916Z"}
    try:
        process = subprocess.Popen(
            server_args,
            stdout=(evaluation_root / "llama_server.log").open("w"),
            stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("llama-server exited before readiness.")
            try:
                with runner.urllib.request.urlopen(
                    runner.build_readiness_url(config), timeout=2
                ) as health:
                    if health.status == 200:
                        break
            except (runner.urllib.error.URLError, TimeoutError):
                time.sleep(2)
        else:
            raise RuntimeError("llama-server readiness timeout.")

        for case_id in remaining:
            case = by_id[case_id]
            case_root = evaluation_root / case_id
            case_root.mkdir()
            envelope = build_synthetic_case(case["grounding_kind"])
            input_payload = envelope.model_dump(mode="json")
            _write_json(case_root / "input_envelope.json", input_payload)
            _write_json(case_root / "ext4c_output_schema.json", schema)
            started = time.monotonic()
            request_count = 1
            metadata = {
                "case_id": case_id,
                "case_category": case["category"],
                "partition": "development",
                "status": "MODEL_REQUEST_FAILURE",
                "generation_started": False,
                "generation_completed": False,
                "response_parse_valid": False,
                "inference_request_count": request_count,
                "retry_count": 0,
                "development_cases_accessed": 1,
                "frozen_final_cases_accessed": 0,
                "locked_test_accessed": False,
            }
            output_payload = runner.build_request_payload(
                config, prompt_bytes.decode("utf-8"), input_payload, schema
            )
            _write_json(case_root / "request.json", output_payload)
            try:
                raw = runner.request_json(runner.build_completion_url(config), output_payload, 180)
                metadata["generation_started"] = True
                metadata["generation_completed"] = True
                (case_root / "raw_http_response.json").write_bytes(raw)
                response, content, candidate = runner.extract_model_content(raw)
                metadata["response_parse_valid"] = True
                (case_root / "raw_model_content.txt").write_text(content, encoding="utf-8")
                validated = GroundedOutputEnvelope.model_validate(candidate)
                references = {item.evidence_id for item in envelope.evidence_items}
                if any(ref.evidence_id not in references for ref in validated.evidence_references):
                    raise RuntimeError("Output references evidence outside the supplied envelope.")
                (case_root / "raw_response.json").write_text(
                    json.dumps(response, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                _write_json(case_root / "parsed_output.json", candidate)
                score = runner.score_case(scoring_case(case), candidate)
                _write_json(case_root / "score.json", score)
                metadata["status"] = (
                    "DEV_CASE_SMOKE_PASS" if score["case_passed"] else "SCIENTIFIC_CASE_FAIL"
                )
                metadata["score_case_passed"] = score["case_passed"]
                metadata["score_valid"] = score["valid"]
                metadata["violation_counts"] = score["violations"]
            except runner.HttpRequestFailure as exc:
                metadata["status"] = "MODEL_REQUEST_FAILURE"
                _write_json(case_root / "http_error.json", exc.as_dict())
                raise RuntimeError(f"Technical request failure for {case_id}.") from exc
            except (
                runner.ResponseProcessingFailure,
                runner.ConfigContractFailure,
                runner.RequestConstructionFailure,
            ) as exc:
                metadata["status"] = "TECHNICAL_CASE_FAILURE"
                metadata["error"] = str(exc)
                raise RuntimeError(f"Technical case failure for {case_id}.") from exc
            finally:
                metadata["latency_seconds"] = round(time.monotonic() - started, 3)
                _write_json(case_root / "run_metadata.json", metadata)
            evidence_paths[case_id] = case_root

        aggregate = aggregate_evidence(CASES_PATH, evidence_paths)
        _write_json(evaluation_root / "development_summary.json", aggregate)
        _write_json(evaluation_root / "development_case_results.json", aggregate["case_results"])
        _write_json(
            evaluation_root / "development_violation_totals.json", aggregate["violation_counts"]
        )
        print(
            json.dumps(
                {
                    "status": "DEVELOPMENT_EVALUATION_COMPLETE_REQUIRES_GOVERNANCE_DECISION",
                    **aggregate,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


if __name__ == "__main__":
    raise SystemExit(main())
