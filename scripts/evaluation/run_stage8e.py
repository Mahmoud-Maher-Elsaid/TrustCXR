from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.segmentation.stage8e_final_evaluation import run_final_evaluation


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()

    run_final_evaluation(arguments.project_root, arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
