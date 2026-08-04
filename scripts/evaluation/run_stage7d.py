from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.comparison.stage7d import run_comparison


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TrustCXR Stage 7D formal model comparison.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_args()
    run_comparison(
        arguments.project_root.resolve(),
        arguments.config.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
