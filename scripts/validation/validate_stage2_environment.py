from __future__ import annotations

import json
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = PROJECT_ROOT / "reports" / "stage2" / "environment_validation.json"
CHECKPOINT_PATH = PROJECT_ROOT / "artifacts" / "stage2" / "checkpoint_smoke.pt"


def bytes_to_gib(value: int) -> float:
    return round(value / (1024**3), 3)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def write_report(report: dict[str, Any]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def run_gpu_validation() -> dict[str, Any]:
    require(
        sys.version_info[:2] == (3, 12),
        f"Python 3.12 is required, but Python {platform.python_version()} is active.",
    )

    require(
        torch.cuda.is_available(),
        "PyTorch cannot access the NVIDIA CUDA GPU.",
    )

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)

    require(
        properties.total_memory >= 7 * 1024**3,
        "Less than 7 GiB of GPU memory is available.",
    )

    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    matrix_start = time.perf_counter()

    matrix_a = torch.randn(
        (2048, 2048),
        device=device,
        dtype=torch.float16,
    )
    matrix_b = torch.randn(
        (2048, 2048),
        device=device,
        dtype=torch.float16,
    )
    matrix_result = matrix_a @ matrix_b
    torch.cuda.synchronize(device)

    matrix_seconds = time.perf_counter() - matrix_start

    require(
        bool(torch.isfinite(matrix_result).all().item()),
        "The CUDA matrix multiplication produced invalid values.",
    )

    del matrix_a
    del matrix_b
    del matrix_result

    model = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(32, 4),
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda")

    inputs = torch.randn(
        (8, 3, 256, 256),
        device=device,
    )
    targets = torch.randint(
        low=0,
        high=4,
        size=(8,),
        device=device,
    )

    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
    ):
        predictions = model(inputs)
        loss = criterion(predictions, targets)

    require(
        bool(torch.isfinite(loss).item()),
        "The mixed-precision loss is not finite.",
    )

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize(device)

    CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "loss": float(loss.detach().cpu().item()),
        },
        CHECKPOINT_PATH,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=True,
    )

    require(
        "model" in checkpoint and "optimizer" in checkpoint,
        "The checkpoint save and load test failed.",
    )

    CHECKPOINT_PATH.unlink(missing_ok=True)

    cpu_images = torch.randn((32, 3, 64, 64))
    cpu_labels = torch.randint(0, 4, (32,))
    dataset = TensorDataset(cpu_images, cpu_labels)
    loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    batch_images, batch_labels = next(iter(loader))
    gpu_images = batch_images.to(device, non_blocking=True)
    gpu_labels = batch_labels.to(device, non_blocking=True)

    require(
        gpu_images.is_cuda and gpu_labels.is_cuda,
        "The DataLoader-to-GPU transfer test failed.",
    )

    torch.cuda.synchronize(device)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    loss_value = float(loss.detach().cpu().item())

    del model
    del optimizer
    del inputs
    del targets
    del predictions
    del loss
    del gpu_images
    del gpu_labels

    torch.cuda.empty_cache()

    return {
        "status": "PASSED",
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "project_root": str(PROJECT_ROOT),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "pytorch": {
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cudnn_available": torch.backends.cudnn.is_available(),
            "cudnn_version": torch.backends.cudnn.version(),
        },
        "gpu": {
            "name": properties.name,
            "compute_capability": (f"{properties.major}.{properties.minor}"),
            "total_memory_gib": bytes_to_gib(properties.total_memory),
            "peak_allocated_gib": bytes_to_gib(peak_allocated),
            "peak_reserved_gib": bytes_to_gib(peak_reserved),
        },
        "tests": {
            "cuda_matrix_multiplication": "PASSED",
            "matrix_seconds": round(matrix_seconds, 4),
            "mixed_precision_forward_backward": "PASSED",
            "mixed_precision_loss": round(loss_value, 6),
            "optimizer_step": "PASSED",
            "checkpoint_save_load": "PASSED",
            "pinned_dataloader_transfer": "PASSED",
        },
    }


def main() -> int:
    report: dict[str, Any] = {
        "status": "STARTED",
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }

    try:
        report = run_gpu_validation()
        write_report(report)

        print(json.dumps(report, indent=2, sort_keys=True))
        print()
        print("STAGE 2 GPU VALIDATION: PASSED")
        return 0

    except Exception as error:
        report["status"] = "FAILED"
        report["error_type"] = type(error).__name__
        report["error"] = str(error)
        write_report(report)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
