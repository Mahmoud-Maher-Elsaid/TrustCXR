from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Short read-only Stage 9 runtime probes.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fingerprint = subparsers.add_parser("fingerprint")
    fingerprint.add_argument("--project-root", type=Path, required=True)
    fingerprint.add_argument("--config", type=Path, required=True)
    checkpoints = subparsers.add_parser("checkpoints")
    checkpoints.add_argument("paths", nargs="+", type=Path)
    subparsers.add_parser("cuda")
    args = parser.parse_args()
    if args.command == "fingerprint":
        sys.path.insert(0, str(args.project_root.resolve() / "src"))
        from trustcxr.integration.stage9b_ablation import config_fingerprint

        config = json.loads(args.config.read_text(encoding="utf-8"))
        print(
            config_fingerprint(
                args.config,
                Path(config["cohort"]["database_path"]),
                Path(config["cohort"]["segmentation_database_path"]),
            )
        )
        return 0
    import torch

    if args.command == "cuda":
        print(int(torch.cuda.is_available()))
        return 0
    for path in args.paths:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            result = {
                "path": str(path),
                "fingerprint": payload.get("config_fingerprint"),
                "variant": payload.get("variant"),
                "epoch": payload.get("epoch"),
            }
        except Exception as exc:  # noqa: BLE001 - probe must report corrupt artifacts
            result = {"path": str(path), "error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
