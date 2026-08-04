from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "training" / "stage7c_rad_dino_probes.json"
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "trustcxr.probes.rad_dino",
            "--project-root",
            str(project_root),
            "--config",
            str(config_path),
        ],
        cwd=project_root,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
