from __future__ import annotations

import argparse
import json
from pathlib import Path

from trustcxr.reliability.stage16d_evaluation import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 16D validation reliability evaluation.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    return run(config, args.project_root.resolve())


if __name__ == "__main__":
    raise SystemExit(main())
