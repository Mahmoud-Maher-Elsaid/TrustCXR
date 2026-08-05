from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.segmentation.stage8d_comparison import run_comparison


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()

    run_comparison(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
