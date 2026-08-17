"""Offline EXT-4E2D contract validation; never starts a server or sends HTTP."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    path = ROOT / "scripts/training/run_ext4e2_candidate1_dev_smoke.py"
    spec = importlib.util.spec_from_file_location("ext4e2d_runner", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main() -> int:
    runner = _load_runner()
    config = json.loads(
        (ROOT / "configs/research_extensions/ext4e2d_candidate1_dev_smoke.json").read_text(
            encoding="utf-8"
        )
    )
    runner.validate_config(config)
    for fields, section in (
        (runner.REQUIRED_TOP_LEVEL_FIELDS, config),
        (runner.REQUIRED_SERVER_FIELDS, config["server"]),
        (runner.REQUIRED_GENERATION_FIELDS, config["generation"]),
    ):
        missing = sorted(set(fields) - set(section))
        if missing:
            raise RuntimeError(f"Required config fields are missing: {missing}")
    for key, value in runner.FROZEN_TOP_LEVEL_VALUES.items():
        if config[key] != value:
            raise RuntimeError(f"Frozen top-level value differs for {key}.")
    for key, value in runner.FROZEN_SERVER_VALUES.items():
        if config["server"][key] != value:
            raise RuntimeError(f"Frozen server value differs for {key}.")
    for key, value in runner.FROZEN_GENERATION_VALUES.items():
        if config["generation"][key] != value:
            raise RuntimeError(f"Frozen generation value differs for {key}.")
    if runner.build_readiness_url(config) != "http://127.0.0.1:18080/health":
        raise RuntimeError("Unexpected readiness endpoint.")
    if runner.build_completion_url(config) != "http://127.0.0.1:18080/v1/chat/completions":
        raise RuntimeError("Unexpected completion endpoint.")
    schema = runner.GroundedOutputEnvelope.model_json_schema()
    payload = runner.build_request_payload(config, "offline preflight", {"offline": True}, schema)
    if payload["reasoning_effort"] != "none":
        raise RuntimeError("Unexpected request reasoning policy.")
    if payload["response_format"]["type"] != "json_object":
        raise RuntimeError("Unexpected structured-output response format.")
    if payload["response_format"]["schema"] != schema:
        raise RuntimeError("Request schema differs from EXT-4C schema.")
    if any(key in payload for key in ("json_schema", "grammar", "grammar_file")):
        raise RuntimeError("Duplicate structured-output mechanism found.")
    print("EXT-4E2D offline pre-inference contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
