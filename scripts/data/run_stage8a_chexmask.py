from __future__ import annotations

import argparse
from pathlib import Path

from trustcxr.segmentation.chexmask import load_records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    arguments = parser.parse_args()

    for split in ("train", "validation", "test"):
        records = load_records(arguments.database, split)
        print(f"{split}: {len(records)} records")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
