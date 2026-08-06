from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.integration.stage9_final_evaluation import run_final_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the frozen Stage 9 test evaluation once.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    return run_final_evaluation(args.project_root.resolve(), args.freeze.resolve(), args.smoke_test)


if __name__ == "__main__":
    raise SystemExit(main())
