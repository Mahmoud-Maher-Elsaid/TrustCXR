from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def close(root: Path = ROOT) -> dict[str, object]:
    config = json.loads(
        (root / "configs/release/stage27b_final_release_closure.json").read_text(encoding="utf-8")
    )
    stage27a = json.loads((root / config["stage27a_report"]).read_text(encoding="utf-8"))
    if stage27a["status"] != config["required_stage27a_status"]:
        raise ValueError("Stage 27A status does not authorize closure")
    if stage27a["gate"] != config["required_stage27a_gate"]:
        raise ValueError("Stage 27A gate does not authorize closure")
    manifest = json.loads((root / config["manifest"]).read_text(encoding="utf-8"))
    if manifest["repository_visibility_expectation"] != "PRIVATE":
        raise ValueError("Repository visibility contract changed")
    extension = manifest["post_release_extension"]
    if any(extension[name] for name in ("grad_cam", "true_localization", "llm", "vlm")):
        raise ValueError("Optional extension was activated in the core release")
    return {
        "stage": "27B",
        "status": config["expected_status"],
        "closure": config["closure"],
        "mandatory_core_stages_remaining": 0,
        "ui_designation": config["ui_designation"],
        "external_validation_performed": False,
        "active_learning_activated": False,
        "training_performed": False,
        "locked_test_records_accessed": 0,
        "grad_cam_implemented": False,
        "true_localization_implemented": False,
        "language_model_implemented": False,
        "multimodal_vlm_implemented": False,
        "recommended_post_release_branch": "research-extension/explainability-grounded-llm",
    }


def main() -> None:
    result = close()
    output = ROOT / "reports/stage27/stage27b_final_release_closure.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
