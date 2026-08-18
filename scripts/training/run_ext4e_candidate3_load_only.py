"""Candidate #3 local-files-only model load gate; never generates or forwards."""

from __future__ import annotations

import gc
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
from typing import Any

import psutil
import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs/research_extensions/ext4e_candidate3_phi4_mini.json"
MODEL_ROOT = ROOT / "cache/research_extensions/ext4e_candidate3/models"
REPORT_PATH = ROOT / "reports/research_extensions/ext4e/EXT4E_CANDIDATE3_LOAD_ONLY.json"
EXPECTED_SCHEMA_SHA256 = "7e28f42cc574cf40d45a725ffac526fc469ac834ab86a574ac613ae79923c650"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def process_rss() -> int:
    return psutil.Process().memory_info().rss


def gpu_memory() -> dict[str, int | bool | str | None]:
    if not torch.cuda.is_available():
        return {"available": False, "device": None, "total_bytes": None, "free_bytes": None}
    free, total = torch.cuda.mem_get_info()
    return {
        "available": True,
        "device": torch.cuda.get_device_name(0),
        "total_bytes": total,
        "free_bytes": free,
        "allocated_bytes": torch.cuda.memory_allocated(0),
        "reserved_bytes": torch.cuda.memory_reserved(0),
    }


def validate_identity(config: dict[str, Any]) -> dict[str, Any]:
    if (
        config["candidate_id"] != 3
        or config["resolved_revision"] != "cfbefacb99257ffa30c83adab238a50856ac3083"
    ):
        raise RuntimeError("CANDIDATE3_WEIGHT_IDENTITY_MISMATCH")
    if config["load_only"] != {
        "dtype": "bfloat16",
        "device_strategy": "CPU_ONLY_BFLOAT16",
        "device_map": {"": "cpu"},
        "quantization": "none",
        "local_files_only": True,
        "trust_remote_code": False,
        "forward_pass": False,
        "generation": False,
    }:
        raise RuntimeError("CANDIDATE3_LOAD_STRATEGY_MISMATCH")
    records = []
    for artifact in config["artifacts"]:
        path = MODEL_ROOT / artifact["filename"]
        if not path.is_file():
            raise RuntimeError("CANDIDATE3_REQUIRED_LOCAL_FILE_MISSING")
        record = {
            "filename": artifact["filename"],
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if record["size_bytes"] != artifact["size_bytes"] or record["sha256"] != artifact["sha256"]:
            raise RuntimeError("CANDIDATE3_WEIGHT_IDENTITY_MISMATCH")
        records.append(record)
    try:
        llguidance_version = importlib.metadata.version("llguidance")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("CANDIDATE3_LLGUIDANCE_NOT_INSTALLED") from exc
    if llguidance_version != "1.8.0":
        raise RuntimeError("CANDIDATE3_LLGUIDANCE_VERSION_MISMATCH")
    if config["partitions"] != {
        "development_case_ids": config["partitions"]["development_case_ids"],
        "development_cases_accessed": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
    }:
        raise RuntimeError("CANDIDATE3_PARTITION_ACCESS_FORBIDDEN")
    return {
        "artifacts": records,
        "llguidance_version": llguidance_version,
    }


def load_model_only() -> tuple[Any, dict[str, Any]]:
    """Load weights on CPU in BF16. This function intentionally has no generate/forward call."""

    started = time.perf_counter()
    model, loading_info = AutoModelForCausalLM.from_pretrained(
        MODEL_ROOT,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        output_loading_info=True,
    )
    model.to("cpu")
    if model.__class__.__name__ != "Phi3ForCausalLM":
        raise RuntimeError("CANDIDATE3_MODEL_CLASS_MISMATCH")
    if loading_info.get("missing_keys") or loading_info.get("unexpected_keys"):
        raise RuntimeError("CANDIDATE3_MODEL_WEIGHT_KEYS_MISMATCH")
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})
    devices = sorted({str(parameter.device) for parameter in model.parameters()})
    return model, {
        "model_class": model.__class__.__name__,
        "parameter_count": parameter_count,
        "loaded_parameter_count": parameter_count,
        "dtypes": dtypes,
        "devices": devices,
        "device_map": {"": "cpu"},
        "missing_keys": loading_info.get("missing_keys", []),
        "unexpected_keys": loading_info.get("unexpected_keys", []),
        "load_duration_seconds": time.perf_counter() - started,
    }


def main() -> int:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    identity = validate_identity(config)
    baseline_cpu = process_rss()
    baseline_gpu = gpu_memory()
    model = None
    result: dict[str, Any] = {
        "status": "CANDIDATE3_LOAD_ONLY_FAILED",
        "candidate_id": 3,
        "revision": config["resolved_revision"],
        "identity": identity,
        "dtype": "bfloat16",
        "device_strategy": config["load_only"],
        "cpu_rss_before_bytes": baseline_cpu,
        "gpu_before": baseline_gpu,
        "development_cases_accessed": 0,
        "final_cases_accessed": 0,
        "locked_test_accessed": False,
        "real_model_inference_requests": 0,
        "generated_tokens": 0,
        "generation_called": False,
    }
    try:
        model, load_info = load_model_only()
        result.update(load_info)
        result["cpu_rss_after_load_bytes"] = process_rss()
        result["gpu_after_load"] = gpu_memory()
        result["status"] = "CANDIDATE3_LOAD_ONLY_PASS"
    except RuntimeError as exc:
        result["failure_classification"] = str(exc)
    except (MemoryError, torch.cuda.OutOfMemoryError) as exc:
        result["failure_classification"] = "CANDIDATE3_GPU_OOM"
        result["error"] = str(exc)
    except Exception as exc:
        result["failure_classification"] = "CANDIDATE3_MODEL_LOAD_ERROR"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        result["gpu_after_cleanup"] = gpu_memory()
        result["cleanup_confirmed"] = True
    REPORT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "CANDIDATE3_LOAD_ONLY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
