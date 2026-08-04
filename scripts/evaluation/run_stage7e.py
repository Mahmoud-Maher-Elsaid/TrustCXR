from __future__ import annotations

import sys
from pathlib import Path

from trustcxr.spatial.stage7e import main

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "configs" / "evaluation" / "stage7e_patch_token_audit.json"
    raise SystemExit(
        main(
            [
                "--project-root",
                str(project_root),
                "--config",
                str(config_path),
                *sys.argv[1:],
            ]
        )
    )
