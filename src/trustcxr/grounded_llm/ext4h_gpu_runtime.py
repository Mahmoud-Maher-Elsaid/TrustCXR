"""Governed Gemma INT8 CUDA runtime gates for EXT-4H.

This module contains only runtime identity and fail-closed placement checks.
It does not alter the semantic planner, slot contracts, or tokenizer policy.
"""

from __future__ import annotations

from typing import Any

GPU_RUNTIME_ID = "EXT4H_GEMMA3_GPU_INT8_V1"
MODEL_REVISION = "093f9f388b31de276ce2de164bdc2081324b9767"
MODEL_VOCAB_SIZE = 262208
CONSTRAINT_VOCAB_SIZE = 262145
TAIL_START = 262145
TAIL_END = 262207
TAIL_COUNT = 63
BNB_VERSION = "0.50.0"


class Ext4hGpuRuntimeError(RuntimeError):
    """Fail-closed GPU runtime error."""


def cuda_preflight(torch_module: Any) -> dict[str, Any]:
    cuda = torch_module.cuda
    if not cuda.is_available():
        raise Ext4hGpuRuntimeError("EXT4HG1_CUDA_UNAVAILABLE")
    count = int(cuda.device_count())
    if count < 1:
        raise Ext4hGpuRuntimeError("EXT4HG1_CUDA_DEVICE_MISSING")
    device = torch_module.device("cuda:0")
    probe = torch_module.ones((1,), device=device)
    probe_value = float(probe.item())
    del probe
    if probe_value != 1.0:
        raise Ext4hGpuRuntimeError("EXT4HG1_CUDA_PROBE_FAILED")
    props = cuda.get_device_properties(0)
    return {
        "cuda_available": True,
        "gpu_count": count,
        "device": "cuda:0",
        "gpu_name": props.name,
        "compute_capability": f"{props.major}.{props.minor}",
        "total_vram_bytes": int(props.total_memory),
        "torch_cuda_version": torch_module.version.cuda,
    }


def build_int8_quantization_config() -> Any:
    try:
        from transformers import BitsAndBytesConfig
    except ImportError as exc:  # pragma: no cover - production gate
        raise Ext4hGpuRuntimeError("EXT4HG1_BITSANDBYTES_INTEGRATION_MISSING") from exc
    return BitsAndBytesConfig(load_in_8bit=True)


def verify_bitsandbytes_cuda() -> dict[str, Any]:
    try:
        import bitsandbytes as bnb
    except ImportError as exc:  # pragma: no cover - production gate
        raise Ext4hGpuRuntimeError("EXT4HG1_BITSANDBYTES_MISSING") from exc
    version = getattr(bnb, "__version__", "unknown")
    if version != BNB_VERSION:
        raise Ext4hGpuRuntimeError("EXT4HG1_BITSANDBYTES_VERSION_MISMATCH")
    try:
        import torch

        layer = bnb.nn.Linear8bitLt(4, 4).to("cuda:0")
        result = layer(torch.ones((1, 4), device="cuda:0"))
        kernel_device = str(result.device)
        del layer, result
        torch.cuda.empty_cache()
    except Exception as exc:  # pragma: no cover - hardware-dependent
        raise Ext4hGpuRuntimeError("EXT4HG1_BNB_CUDA_KERNEL_FAILED") from exc
    if kernel_device != "cuda:0":
        raise Ext4hGpuRuntimeError("EXT4HG1_BNB_CPU_FALLBACK")
    return {
        "version": version,
        "backend": "bitsandbytes CUDA Linear8bitLt",
        "linear8bitlt_cuda_smoke": "PASS",
        "kernel_device": kernel_device,
    }


def validate_int8_model_placement(model: Any) -> dict[str, Any]:
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    buffer_devices = sorted({str(buffer.device) for buffer in model.buffers()})
    quantized_modules = sum(
        1 for module in model.modules() if module.__class__.__name__ == "Linear8bitLt"
    )
    if parameter_devices != ["cuda:0"] or any(device != "cuda:0" for device in buffer_devices):
        raise Ext4hGpuRuntimeError("EXT4HG1_CPU_MODEL_FALLBACK")
    if quantized_modules <= 0:
        raise Ext4hGpuRuntimeError("EXT4HG1_NO_INT8_MODULES")
    input_device = str(model.get_input_embeddings().weight.device)
    if input_device != "cuda:0":
        raise Ext4hGpuRuntimeError("EXT4HG1_INPUT_EMBEDDING_NOT_CUDA")
    return {
        "parameter_devices": parameter_devices,
        "buffer_devices": buffer_devices,
        "quantized_linear8bitlt_modules": quantized_modules,
        "input_embedding_device": input_device,
        "gpu_execution_verified": True,
    }


def validate_gemma_vocab_alignment(tokenizer: Any) -> dict[str, Any]:
    vocabulary = tokenizer.get_vocab()
    ids = sorted(vocabulary.values())
    if ids != list(range(CONSTRAINT_VOCAB_SIZE)):
        raise Ext4hGpuRuntimeError("EXT4HG1_VOCAB_ALIGNMENT_FAILED")
    return {
        "model_vocab_size": MODEL_VOCAB_SIZE,
        "constraint_vocab_size": CONSTRAINT_VOCAB_SIZE,
        "identity_prefix": [0, CONSTRAINT_VOCAB_SIZE - 1],
        "model_only_tail": [TAIL_START, TAIL_END],
        "tail_count": TAIL_COUNT,
        "tail_policy": "always_forbidden",
    }
