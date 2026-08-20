"""EXT-4G.1 Gemma identity, integrity, and load-only audit helpers.

This module deliberately contains no generation or forward-inference path.
It can download only the pinned repository revision when explicitly asked by
the local runner, then audits files, configuration, tokenizer identity, and
optional CPU-only load state.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

EXT4G1_CANDIDATE_ID = "EXT4G_CANDIDATE_GEMMA3_4B_IT_V1"
EXT4G1_REPOSITORY = "google/gemma-3-4b-it"
EXT4G1_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
EXT4G1_BACKEND = "llguidance"
EXT4G1_BACKEND_VERSION = "1.8.0"
EXT4G1_SCHEMA_SHA256 = "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
EXT4G1_BENCHMARK_SHA256 = "671a04d2d859f1b1ffb9414a8c0f636596949748a00548e45abcbbfdb752db61"
EXT4G1_MODEL_PATH = Path("cache/research_extensions/ext4g_candidate_gemma3_4b_it/models")


class Ext4g1IdentityError(RuntimeError):
    """Raised when an EXT-4G.1 identity or load-only gate fails closed."""


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_revision(repository: str, revision: str) -> None:
    if repository != EXT4G1_REPOSITORY or revision != EXT4G1_REVISION:
        raise Ext4g1IdentityError("EXT4G1_IMMUTABLE_REVISION_MISMATCH")


def download_pinned_snapshot(destination: Path) -> str:
    """Download only the pinned snapshot; credentials remain in hub config."""
    verify_revision(EXT4G1_REPOSITORY, EXT4G1_REVISION)
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=EXT4G1_REPOSITORY,
            revision=EXT4G1_REVISION,
            local_dir=str(destination),
            repo_type="model",
        )
    except Exception as exc:  # pragma: no cover - exercised in governed runtime
        raise Ext4g1IdentityError("EXT4G1_PINNED_DOWNLOAD_FAILED") from exc
    return str(destination)


def enumerate_pinned_revision_files() -> list[dict[str, Any]]:
    """Enumerate the authenticated Hub manifest before any download."""
    verify_revision(EXT4G1_REPOSITORY, EXT4G1_REVISION)
    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(EXT4G1_REPOSITORY, revision=EXT4G1_REVISION)
    except Exception as exc:  # pragma: no cover - exercised in governed runtime
        raise Ext4g1IdentityError("EXT4G1_AUTHENTICATED_HUB_METADATA_FAILED") from exc
    if info.sha != EXT4G1_REVISION:
        raise Ext4g1IdentityError("EXT4G1_HUB_REVISION_RESOLUTION_MISMATCH")
    return [
        {
            "filename": sibling.rfilename,
            "hub_size": sibling.size,
            "lfs_oid": getattr(getattr(sibling, "lfs", None), "oid", None),
        }
        for sibling in sorted(info.siblings or [], key=lambda item: item.rfilename)
    ]


def build_integrity_manifest(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        raise Ext4g1IdentityError("EXT4G1_MODEL_CACHE_MISSING")
    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise Ext4g1IdentityError("EXT4G1_MODEL_CACHE_EMPTY")
    return [
        {
            "filename": path.relative_to(root).as_posix(),
            "byte_size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]


def verify_safetensor_index(root: Path) -> dict[str, Any]:
    index_path = root / "model.safetensors.index.json"
    if not index_path.is_file():
        raise Ext4g1IdentityError("EXT4G1_SAFETENSOR_INDEX_MISSING")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise Ext4g1IdentityError("EXT4G1_SAFETENSOR_INDEX_INVALID")
    referenced = sorted(set(weight_map.values()))
    missing = [name for name in referenced if not (root / name).is_file()]
    if missing:
        raise Ext4g1IdentityError("EXT4G1_SAFETENSOR_SHARD_MISSING")
    return {
        "index": index_path.name,
        "tensor_count": len(weight_map),
        "referenced_shards": referenced,
        "metadata": index.get("metadata", {}),
    }


def audit_config(root: Path) -> dict[str, Any]:
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
    except Exception as exc:
        raise Ext4g1IdentityError("EXT4G1_CONFIG_LOAD_FAILED") from exc
    text_config = getattr(config, "text_config", None)
    return {
        "model_type": getattr(config, "model_type", None),
        "architectures": list(getattr(config, "architectures", None) or []),
        "vocab_size": getattr(text_config or config, "vocab_size", None),
        "hidden_size": getattr(text_config or config, "hidden_size", None),
        "num_hidden_layers": getattr(text_config or config, "num_hidden_layers", None),
        "num_attention_heads": getattr(text_config or config, "num_attention_heads", None),
        "torch_dtype": str(getattr(config, "torch_dtype", None)),
        "vision_config_present": getattr(config, "vision_config", None) is not None,
        "eos_token_id": getattr(text_config or config, "eos_token_id", None),
        "bos_token_id": getattr(text_config or config, "bos_token_id", None),
        "pad_token_id": getattr(text_config or config, "pad_token_id", None),
    }


def audit_processor(root: Path, probe_text: str) -> dict[str, Any]:
    try:
        from transformers import AutoProcessor

        processor = AutoProcessor.from_pretrained(
            str(root), local_files_only=True, trust_remote_code=False
        )
        tokenizer = getattr(processor, "tokenizer", processor)
        encoded = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "text", "text": probe_text}]}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    except Exception as exc:
        raise Ext4g1IdentityError("EXT4G1_TEXT_PROCESSOR_AUDIT_FAILED") from exc
    vocab = tokenizer.get_vocab()
    return {
        "processor_class": type(processor).__name__,
        "tokenizer_class": type(tokenizer).__name__,
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", len(vocab)),
        "tokenizer_len": len(tokenizer),
        "tokenizer_max_id": max(vocab.values(), default=-1),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "probe_token_count": int(encoded.shape[-1]),
        "chat_template_sha256": hashlib.sha256(
            str(getattr(tokenizer, "chat_template", "")).encode("utf-8")
        ).hexdigest(),
    }


def verify_runtime_versions() -> dict[str, str]:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - governed runtime
        raise Ext4g1IdentityError("EXT4G1_TORCH_MISSING") from exc
    versions = {
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "llguidance": importlib.metadata.version("llguidance"),
    }
    if versions["python"] != "3.12.10":
        raise Ext4g1IdentityError("EXT4G1_PYTHON_VERSION_MISMATCH")
    if versions["torch"] != "2.12.1+cu130":
        raise Ext4g1IdentityError("EXT4G1_TORCH_VERSION_MISMATCH")
    if versions["transformers"] != "4.57.6":
        raise Ext4g1IdentityError("EXT4G1_TRANSFORMERS_VERSION_MISMATCH")
    if versions["llguidance"] != EXT4G1_BACKEND_VERSION:
        raise Ext4g1IdentityError("EXT4G1_LLGUIDANCE_VERSION_MISMATCH")
    return versions


def assert_cpu_bfloat16_model(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    buffers = list(model.buffers())
    devices = sorted({str(parameter.device) for parameter in parameters})
    buffer_devices = sorted({str(buffer.device) for buffer in buffers})
    dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    if devices != ["cpu"] or any(device != "cpu" for device in buffer_devices):
        raise Ext4g1IdentityError("EXT4G1_NON_CPU_PLACEMENT")
    if dtypes != ["torch.bfloat16"]:
        raise Ext4g1IdentityError("EXT4G1_DTYPE_MISMATCH")
    return {
        "model_class": type(model).__name__,
        "parameter_count": sum(parameter.numel() for parameter in parameters),
        "parameter_devices": devices,
        "buffer_devices": buffer_devices,
        "parameter_dtypes": dtypes,
    }
