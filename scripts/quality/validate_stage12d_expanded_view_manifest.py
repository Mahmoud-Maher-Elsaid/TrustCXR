from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__:
    from scripts.quality.run_stage12d_annotation_cohort_readiness import inspect_manifest
else:
    from run_stage12d_annotation_cohort_readiness import inspect_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Stage 12D expanded-view manifest.")
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(
        (root / "configs/quality/stage12d_annotation_cohort_readiness.json").read_text(
            encoding="utf-8"
        )
    )
    specification = config["manifests"]["expanded_view"]
    manifest = root / config["local_manifest_root"] / specification["filename"]
    result = inspect_manifest(
        manifest,
        specification,
        config["protocol_version"],
        set(config["allowed_development_splits"]),
    )
    counts = result.get("label_counts", {})
    if not result["ready"] or counts.get("UNKNOWN", 0) < 17:
        raise RuntimeError("Expanded-view manifest did not validate approved UNKNOWN records.")
    if counts.get("OTHER", 0) != 0:
        raise RuntimeError("Expanded-view manifest contains unsupported OTHER annotations.")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
