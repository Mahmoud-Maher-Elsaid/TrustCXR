from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.integration.stage9c_comparison import run_stage9c


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only Stage 9C comparison.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-only", action="store_true", required=True)
    parser.add_argument("--paired-patient-bootstrap", action="store_true", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    return run_stage9c(args.project_root.resolve(), args.config.resolve(), args.smoke_test)


if __name__ == "__main__":
    raise SystemExit(main())
