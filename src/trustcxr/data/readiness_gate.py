"""Final data readiness gate and training dataset selection."""

from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    value = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return value


def validate_selection(selection: dict[str, Any]) -> None:
    """Validate the final training selection configuration."""
    datasets = selection.get("datasets")
    sequence = selection.get("training_sequence")
    first_stage = selection.get("first_training_stage")
    split_policy = selection.get("split_policy")

    if not isinstance(datasets, list) or len(datasets) != 10:
        raise ValueError("Exactly ten dataset decisions are required.")

    if not isinstance(sequence, list) or not sequence:
        raise ValueError("The training sequence is empty.")

    if not isinstance(first_stage, dict):
        raise ValueError("The first training stage is missing.")

    if not isinstance(split_policy, dict):
        raise ValueError("The split policy is missing.")

    dataset_ids = [str(item["id"]) for item in datasets]

    if len(dataset_ids) != len(set(dataset_ids)):
        raise ValueError("Dataset IDs must be unique.")

    if split_policy.get("allow_image_level_random_split") is not False:
        raise ValueError("Image-level random splitting must remain disabled.")

    ready_datasets = [item for item in datasets if item.get("decision") == "TRAINING_READY"]

    withheld_datasets = [item for item in datasets if item.get("decision") == "WITHHELD_IDENTITY"]

    if len(ready_datasets) != 4:
        raise ValueError("Exactly four datasets must be training-ready.")

    if len(withheld_datasets) != 6:
        raise ValueError("Exactly six datasets must remain safely withheld.")

    for item in ready_datasets:
        if item.get("patient_level_ready") is not True:
            raise ValueError(f"Training-ready dataset lacks patient readiness: {item['id']}")

    for item in withheld_datasets:
        if item.get("approved_tasks"):
            raise ValueError(f"Withheld dataset has approved tasks: {item['id']}")

    if first_stage.get("gate") != "APPROVED":
        raise ValueError("The first training stage is not approved.")

    if first_stage.get("dataset_id") not in dataset_ids:
        raise ValueError("The first training dataset is unknown.")


def build_summary(
    *,
    selection: dict[str, Any],
    stage4_2: dict[str, Any],
    stage4_3: dict[str, Any],
) -> dict[str, Any]:
    """Build the privacy-safe final data readiness summary."""
    validate_selection(selection)

    datasets = selection["datasets"]
    sequence = selection["training_sequence"]

    ready = [item for item in datasets if item["decision"] == "TRAINING_READY"]
    withheld = [item for item in datasets if item["decision"] == "WITHHELD_IDENTITY"]

    leakage_stage4_2 = int(stage4_2.get("total_leakage_violations", -1))
    leakage_stage4_3 = int(stage4_3.get("total_leakage_violations", -1))

    if leakage_stage4_2 != 0 or leakage_stage4_3 != 0:
        raise ValueError("Patient leakage violations are not zero.")

    return {
        "schema_version": "1.0",
        "status": "PASSED",
        "overall_gate": "GO_FOR_STAGE_5",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "audit_mode": "READ_ONLY",
        "dataset_count": len(datasets),
        "training_ready_dataset_count": len(ready),
        "safely_withheld_dataset_count": len(withheld),
        "patient_level_complete_count": int(stage4_3["overall_patient_level_complete_count"]),
        "patient_leakage_violations": 0,
        "first_training_stage": selection["first_training_stage"],
        "training_sequence": sequence,
        "training_ready_datasets": [
            {
                "id": item["id"],
                "name": item["name"],
                "approved_tasks": item["approved_tasks"],
                "role": item["role"],
            }
            for item in ready
        ],
        "safely_withheld_datasets": [
            {
                "id": item["id"],
                "name": item["name"],
                "reason": "PATIENT_IDENTITY_UNRESOLVED",
                "role": item["role"],
            }
            for item in withheld
        ],
        "safety_gates": {
            "patient_level_split_required": True,
            "image_level_random_split_allowed": False,
            "unresolved_identity_action": "WITHHOLD",
            "raw_patient_values_committed": False,
            "dataset_files_committed": False,
        },
    }


def build_markdown(summary: dict[str, Any]) -> str:
    """Build the final readiness report."""
    first_stage = summary["first_training_stage"]

    lines = [
        "# TrustCXR Final Data Readiness Gate",
        "",
        f"Generated at UTC: `{summary['generated_at_utc']}`",
        "",
        "## Final decision",
        "",
        f"- Overall gate: `{summary['overall_gate']}`",
        f"- Status: `{summary['status']}`",
        (f"- Training-ready datasets: `{summary['training_ready_dataset_count']}`"),
        (f"- Safely withheld datasets: `{summary['safely_withheld_dataset_count']}`"),
        (f"- Patient leakage violations: `{summary['patient_leakage_violations']}`"),
        "",
        "## First approved training stage",
        "",
        f"- Stage: `{first_stage['stage']}`",
        f"- Task: `{first_stage['task']}`",
        f"- Dataset: `{first_stage['dataset_id']}`",
        f"- Model: `{first_stage['model']}`",
        "",
        "## Approved training sequence",
        "",
        "| Order | Stage | Task | Dataset | Model | Status |",
        "|---:|---|---|---|---|---|",
    ]

    for item in summary["training_sequence"]:
        lines.append(
            "| "
            f"{item['order']} | "
            f"{item['stage']} | "
            f"{item['task']} | "
            f"{item['dataset_id']} | "
            f"{item['model']} | "
            f"{item['status']} |"
        )

    lines.extend(
        [
            "",
            "## Training-ready datasets",
            "",
            "| Dataset | Approved tasks | Role |",
            "|---|---|---|",
        ]
    )

    for item in summary["training_ready_datasets"]:
        tasks = ", ".join(item["approved_tasks"])
        lines.append(f"| {item['name']} | {tasks} | {item['role']} |")

    lines.extend(
        [
            "",
            "## Safely withheld datasets",
            "",
            "| Dataset | Reason | Intended future role |",
            "|---|---|---|",
        ]
    )

    for item in summary["safely_withheld_datasets"]:
        lines.append(f"| {item['name']} | {item['reason']} | {item['role']} |")

    lines.extend(
        [
            "",
            "## Safety policy",
            "",
            "- Patient-level splitting remains mandatory.",
            "- Image-level random splitting remains prohibited.",
            "- Unresolved patient identity causes withholding, not failure.",
            "- Dataset files and patient-level manifests remain local-only.",
            "- Stage 5 may begin without waiting for withheld datasets.",
            "",
        ]
    )

    return "\n".join(lines)


def write_outputs(
    *,
    report_root: Path,
    summary: dict[str, Any],
) -> None:
    """Write aggregate Stage 4.4 reports."""
    report_root.mkdir(parents=True, exist_ok=True)

    (report_root / "data_readiness_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    (report_root / "FINAL_DATA_READINESS.md").write_text(
        build_markdown(summary),
        encoding="utf-8",
    )

    with (report_root / "training_sequence.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "order",
                "stage",
                "task",
                "dataset_id",
                "model",
                "input_size",
                "status",
                "fallback_model",
            ],
        )
        writer.writeheader()
        writer.writerows(summary["training_sequence"])
