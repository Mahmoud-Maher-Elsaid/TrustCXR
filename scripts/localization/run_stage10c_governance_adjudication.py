from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

LICENSE_DECISIONS = {"PENDING_MANUAL_REVIEW", "APPROVED_FOR_RESEARCH", "REJECTED"}


def load_matrix(path: Path) -> dict[str, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return {row["dataset"]: row for row in csv.DictReader(handle)}


def adjudicate(config: dict[str, Any], matrix: dict[str, dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for decision in config["datasets"]:
        name = decision["name"]
        if name not in matrix:
            raise RuntimeError(f"Stage 10B evidence is missing for {name}.")
        license_decision = decision["license_decision"]
        if license_decision not in LICENSE_DECISIONS:
            raise RuntimeError(f"Invalid license decision for {name}: {license_decision}")
        if license_decision != "PENDING_MANUAL_REVIEW" and not all(
            decision.get(field) for field in ("reviewer", "reviewed_at", "review_note")
        ):
            raise RuntimeError(
                f"A completed license decision for {name} requires reviewer evidence."
            )
        identity_resolved = decision["identity_decision"] == "ACCEPT_STAGE10B_PATIENT_TRACKING"
        if identity_resolved and matrix[name]["identity_status"] != "PATIENT_TRACKING_RESOLVED":
            raise RuntimeError(f"Identity acceptance for {name} contradicts Stage 10B evidence.")
        ready = identity_resolved and license_decision == "APPROVED_FOR_RESEARCH"
        rows.append(
            {
                "dataset": name,
                "identity_decision": decision["identity_decision"],
                "license_decision": license_decision,
                "license_source": decision["license_source"],
                "ready_for_split_design": ready,
                "manual_action": (
                    "Review authoritative terms and record approval or rejection"
                    if license_decision == "PENDING_MANUAL_REVIEW"
                    else "None"
                ),
            }
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Adjudicate Stage 10C governance decisions.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if (
        config.get("training_permitted") is not False
        or config.get("test_access_permitted") is not False
    ):
        raise RuntimeError("Stage 10C must prohibit training and test access.")
    summary10b = json.loads(
        (root / "reports/stage10/stage10b_governance_identity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        summary10b.get("status") != "PASSED_RESOLUTION_AUDIT"
        or summary10b.get("test_records_accessed") != 0
    ):
        raise RuntimeError("Stage 10C refused: Stage 10B evidence is invalid.")
    rows = adjudicate(
        config,
        load_matrix(root / "reports/stage10/stage10b_governance_identity_matrix.csv"),
    )
    ready = [row for row in rows if row["ready_for_split_design"]]
    pending = [row for row in rows if row["license_decision"] == "PENDING_MANUAL_REVIEW"]
    summary = {
        "stage": "10C",
        "status": "PASSED_GOVERNANCE_ADJUDICATION",
        "gate": (
            "GO_FOR_STAGE_10D_PATIENT_SAFE_SPLIT_DESIGN"
            if ready and not pending
            else "HOLD_FOR_MANUAL_LICENSE_DECISIONS"
        ),
        "datasets_reviewed": len(rows),
        "datasets_ready_for_split_design": len(ready),
        "manual_license_decisions_pending": len(pending),
        "training_permitted": False,
        "test_records_accessed": 0,
        "patient_level_rows_tracked": False,
    }
    report_root = root / "reports/stage10"
    report_root.mkdir(parents=True, exist_ok=True)
    (report_root / "stage10c_governance_adjudication_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (report_root / "stage10c_governance_decisions.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Stage 10C Manual Governance Actions",
        "",
        f"- Gate: `{summary['gate']}`",
        "- Training permitted: `false`",
        "- Test records accessed: `0`",
        "",
        "Review each authoritative source in the Stage 10C config. For each dataset, set "
        "`license_decision` to `APPROVED_FOR_RESEARCH` or `REJECTED`, and record reviewer, "
        "ISO-8601 review time, and a concise evidence note. This is a human legal/governance "
        "decision; the script cannot approve terms automatically.",
        "",
    ]
    for row in rows:
        lines.append(
            f"- **{row['dataset']}** — identity: `{row['identity_decision']}`; "
            f"license: `{row['license_decision']}`; source: {row['license_source']}"
        )
    (report_root / "STAGE10C_MANUAL_GOVERNANCE_ACTIONS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
