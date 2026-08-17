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
