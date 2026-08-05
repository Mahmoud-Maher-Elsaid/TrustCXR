from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.integration.stage9b_ablation import run_ablation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_ablation(arguments.project_root, arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
