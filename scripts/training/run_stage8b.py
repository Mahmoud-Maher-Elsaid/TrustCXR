from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.segmentation.stage8b_unet import run_training_only


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    arguments = parser.parse_args()
    run_training_only(arguments.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
