from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    config = project_root / "configs" / "training" / "stage6_nih_densenet121.json"
    return subprocess.call(
        [
            sys.executable,
            "-m",
            "trustcxr.classification.train",
            "--project-root",
            str(project_root),
            "--config",
            str(config),
        ],
        cwd=project_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
