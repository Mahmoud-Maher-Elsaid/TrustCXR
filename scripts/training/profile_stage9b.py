from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
from torch import nn

from trustcxr.integration.stage9b_ablation import (
    CohortIndex,
    Stage9Dataset,
    build_loader,
    build_model,
    deterministic_subset,
)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def profile_candidate(
    *,
    config: dict,
    cohort_index: CohortIndex,
    variant: str,
    workers: int,
    records: int,
    batches: int,
) -> dict:
    training = config["training"]
    identifiers = deterministic_subset(
        cohort_index.identifiers("train"), records, int(training["seed"]) + 901
    )
    dataset = Stage9Dataset(
        cohort_index,
        Path(config["cohort"]["segmentation_database_path"]),
        identifiers,
        variant=variant,
        image_size=int(training["image_size"]),
        augment=True,
        seed=int(training["seed"]),
        horizontal_flip_probability=float(training["horizontal_flip_probability"]),
        brightness_jitter=float(training["brightness_jitter"]),
        contrast_jitter=float(training["contrast_jitter"]),
    )
    loader = build_loader(
        dataset,
        batch_size=int(training["batch_size"]),
        shuffle=False,
        seed=int(training["seed"]) + 902,
        num_workers=workers,
    )
    iterator = iter(loader)
    batch_times: list[float] = []
    first_batch = None
    for _ in range(batches):
        started = time.perf_counter()
        batch = next(iterator)
        batch_times.append(time.perf_counter() - started)
        if first_batch is None:
            first_batch = batch
    assert first_batch is not None
    images, targets, _ = first_batch
    result = {
        "variant": variant,
        "num_workers": workers,
        "records": len(identifiers),
        "batches_measured": len(batch_times),
        "first_batch_seconds": batch_times[0],
        "later_batch_mean_seconds": (
            sum(batch_times[1:]) / len(batch_times[1:]) if len(batch_times) > 1 else None
        ),
        "loader_records_per_second": images.shape[0] * len(batch_times) / sum(batch_times),
        "batch_shape": list(images.shape),
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.reset_peak_memory_stats()
        model = build_model(
            len(config["labels"]), input_channels=images.shape[1], pretrained=False
        ).to(device)
        criterion = nn.BCEWithLogitsLoss()
        synchronize()
        started = time.perf_counter()
        device_images = images.to(device, non_blocking=True)
        device_targets = targets.to(device, non_blocking=True)
        synchronize()
        result["host_to_device_seconds"] = time.perf_counter() - started
        model.zero_grad(set_to_none=True)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(device_images)
            loss = criterion(logits, device_targets)
        synchronize()
        result["forward_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        loss.backward()
        synchronize()
        result["backward_seconds"] = time.perf_counter() - started
        result["peak_vram_allocated_bytes"] = torch.cuda.max_memory_allocated()
        result["peak_vram_reserved_bytes"] = torch.cuda.max_memory_reserved()
        del model, device_images, device_targets, logits, loss
        torch.cuda.empty_cache()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a bounded non-learning Stage 9B profile.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--records", type=int, default=96)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 4])
    arguments = parser.parse_args()
    config = json.loads(arguments.config.read_text(encoding="utf-8"))
    cohort_index = CohortIndex(Path(config["cohort"]["database_path"]))
    results = []
    for variant in config["variants"]:
        for workers in arguments.workers:
            print(f"Profiling {variant} with {workers} workers.", flush=True)
            results.append(
                profile_candidate(
                    config=config,
                    cohort_index=cohort_index,
                    variant=variant,
                    workers=workers,
                    records=arguments.records,
                    batches=arguments.batches,
                )
            )
    payload = {
        "profile_type": "BOUNDED_NON_LEARNING",
        "test_records_accessed": 0,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "results": results,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
