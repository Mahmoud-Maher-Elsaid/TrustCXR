from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.features.rad_dino import run_extraction


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrustCXR Stage 7B RAD-DINO CLS extraction.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    project_root = arguments.project_root.resolve()
    config_path = arguments.config or (
        project_root / "configs" / "features" / "stage7b_rad_dino_cls.json"
    )
    run_extraction(project_root, config_path.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
