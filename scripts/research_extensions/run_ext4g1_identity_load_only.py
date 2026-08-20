"""Run the governed EXT-4G.1 Gemma identity/integrity/load-only gate.

Use ``--download`` only in the authenticated production environment. This
script never calls ``forward`` or ``generate`` and never opens benchmark data.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from trustcxr.grounded_llm.ext4g1_identity import (
    EXT4G1_MODEL_PATH,
    EXT4G1_REPOSITORY,
    EXT4G1_REVISION,
    assert_cpu_bfloat16_model,
    audit_config,
    audit_processor,
    build_integrity_manifest,
    download_pinned_snapshot,
    enumerate_pinned_revision_files,
    verify_revision,
    verify_runtime_versions,
    verify_safetensor_index,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--root", type=Path, default=EXT4G1_MODEL_PATH)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    verify_revision(EXT4G1_REPOSITORY, EXT4G1_REVISION)
    remote_manifest = enumerate_pinned_revision_files()
    if args.download:
        download_pinned_snapshot(args.root)
    versions = verify_runtime_versions()
    manifest = build_integrity_manifest(args.root)
    shard_index = verify_safetensor_index(args.root)
    config = audit_config(args.root)
    processor = audit_processor(args.root, "EXT-4G.1 harmless text-only identity probe.")

    load_result = None
    if args.load_only:
        import torch
        from transformers import Gemma3ForConditionalGeneration

        model = Gemma3ForConditionalGeneration.from_pretrained(
            str(args.root),
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.bfloat16,
        )
        if not isinstance(model, Gemma3ForConditionalGeneration):
            raise RuntimeError("EXT4G1_MODEL_CLASS_MISMATCH")
        load_result = assert_cpu_bfloat16_model(model)
        del model
        gc.collect()

    result = {
        "stage": "EXT-4G.1",
        "candidate_id": "EXT4G_CANDIDATE_GEMMA3_4B_IT_V1",
        "repository": EXT4G1_REPOSITORY,
        "revision": EXT4G1_REVISION,
        "model_path": str(args.root),
        "remote_manifest": remote_manifest,
        "runtime_versions": versions,
        "integrity_manifest": manifest,
        "safetensor_index": shard_index,
        "config": config,
        "processor": processor,
        "load_only": load_result,
        "model_forward_calls": 0,
        "model_generate_calls": 0,
        "development_cases_accessed": 0,
        "frozen_final_cases_accessed": 0,
        "locked_test_accessed": False,
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
