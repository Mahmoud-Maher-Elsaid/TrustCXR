from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def pair_id(study: str) -> str:
    return hashlib.sha256(f"trustcxr-stage13c:pair:{study}".encode()).hexdigest()


def design(config: dict[str, Any], root: Path) -> dict[str, Any]:
    prohibited = (
        config["heuristic_pairing_permitted"],
        config["duplicate_view_selection_permitted"],
        config["training_permitted"],
        config["inference_permitted"],
        config["locked_test_access_permitted"],
        config["frozen_results_may_be_modified"],
    )
    if any(prohibited) or not config["exact_pair_requires_one_frontal_and_one_lateral"]:
        raise RuntimeError("Stage 13C safety contract changed.")
    stage13b = json.loads((root / config["stage13b_evidence"]).read_text(encoding="utf-8"))
    if stage13b.get("gate") != "GO_FOR_STAGE_13C_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN":
        raise RuntimeError("Stage 13C requires the completed Stage 13B gate.")
    source = root / config["study_identity_index"]
    connection = sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT record_key_hash, patient_key_hash, study_key_hash, split, view_label, "
        "identity_evidence "
        "FROM study_records ORDER BY study_key_hash, record_key_hash"
    ).fetchall()
    connection.close()
    allowed_splits = set(config["allowed_splits"])
    if any(row[3] not in allowed_splits for row in rows):
        raise RuntimeError("Stage 13C detected a locked split.")
    allowed_identity_evidence = set(config["allowed_identity_evidence"])
    if any(row[5] not in allowed_identity_evidence for row in rows):
        raise RuntimeError("Stage 13C detected a record without approved explicit study identity.")

    studies: dict[str, list[tuple[str, str, str, str, str]]] = defaultdict(list)
    for row in rows:
        studies[row[2]].append(row)
    pairs: list[tuple[str, str, str, str, str, str, str, str]] = []
    dispositions: list[tuple[str, str, str, str, int]] = []
    disposition_counts: Counter[str] = Counter()
    pair_split_counts: Counter[str] = Counter()
    for study, records in studies.items():
        patients = {row[1] for row in records}
        splits = {row[3] for row in records}
        if len(patients) != 1 or len(splits) != 1:
            raise RuntimeError("Stage 13C study contains multiple patients or splits.")
        patient = next(iter(patients))
        split = next(iter(splits))
        frontals = [row for row in records if row[4] in config["frontal_views"]]
        laterals = [row for row in records if row[4] in config["lateral_views"]]
        unknown = [row for row in records if row[4] in config["unknown_views"]]
        if len(records) == 2 and len(frontals) == 1 and len(laterals) == 1 and not unknown:
            status = "EXACT_FRONTAL_LATERAL_PAIR"
            pairs.append(
                (
                    pair_id(study),
                    patient,
                    study,
                    split,
                    frontals[0][0],
                    frontals[0][4],
                    laterals[0][0],
                    laterals[0][4],
                )
            )
            pair_split_counts[split] += 1
        elif len(records) == 1 and len(frontals) == 1:
            status = "SINGLE_FRONTAL"
        elif len(records) == 1 and len(laterals) == 1:
            status = "SINGLE_LATERAL"
        elif unknown:
            status = "WITHHELD_UNKNOWN_VIEW_PRESENT"
        else:
            status = "WITHHELD_DUPLICATE_OR_AMBIGUOUS_VIEWS"
        disposition_counts[status] += 1
        dispositions.append((study, patient, split, status, len(records)))

    destination = root / config["output_pair_index"]
    if destination.exists():
        raise RuntimeError(f"Stage 13C output exists; preserve it before rerun: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    output = sqlite3.connect(destination)
    try:
        output.execute(
            "CREATE TABLE pair_records (pair_key_hash TEXT PRIMARY KEY, "
            "patient_key_hash TEXT NOT NULL, study_key_hash TEXT NOT NULL UNIQUE, "
            "split TEXT NOT NULL, frontal_record_key_hash TEXT NOT NULL, "
            "frontal_view TEXT NOT NULL, lateral_record_key_hash TEXT NOT NULL, "
            "lateral_view TEXT NOT NULL)"
        )
        output.execute(
            "CREATE TABLE study_dispositions (study_key_hash TEXT PRIMARY KEY, "
            "patient_key_hash TEXT NOT NULL, split TEXT NOT NULL, disposition TEXT NOT NULL, "
            "record_count INTEGER NOT NULL)"
        )
        output.executemany("INSERT INTO pair_records VALUES (?, ?, ?, ?, ?, ?, ?, ?)", pairs)
        output.executemany("INSERT INTO study_dispositions VALUES (?, ?, ?, ?, ?)", dispositions)
        output.commit()
        patient_leakage = output.execute(
            "SELECT COUNT(*) FROM (SELECT patient_key_hash FROM pair_records "
            "GROUP BY patient_key_hash HAVING COUNT(DISTINCT split) > 1)"
        ).fetchone()[0]
        record_reuse = output.execute(
            "SELECT COUNT(*) FROM (SELECT record_key FROM ("
            "SELECT frontal_record_key_hash AS record_key FROM pair_records UNION ALL "
            "SELECT lateral_record_key_hash AS record_key FROM pair_records) "
            "GROUP BY record_key HAVING COUNT(*) > 1)"
        ).fetchone()[0]
    finally:
        output.close()
    valid = bool(pairs) and patient_leakage == 0 and record_reuse == 0
    return {
        "stage": "13C",
        "status": "PASSED_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN" if valid else "HOLD_PAIR_DESIGN",
        "gate": (
            "GO_FOR_STAGE_13D_MULTIVIEW_BASELINE_PREPARATION"
            if valid
            else "HOLD_FOR_MULTIVIEW_PAIR_EVIDENCE"
        ),
        "source_records": len(rows),
        "source_studies": len(studies),
        "exact_pairs": len(pairs),
        "pair_split_counts": dict(sorted(pair_split_counts.items())),
        "study_disposition_counts": dict(sorted(disposition_counts.items())),
        "patient_leakage_violations": patient_leakage,
        "record_reuse_violations": record_reuse,
        "heuristic_pairs_created": 0,
        "duplicate_views_selected": 0,
        "unknown_views_paired": 0,
        "other_view_withheld": config["other_view_withheld"],
        "locked_test_records_accessed": 0,
        "training_performed": False,
        "inference_performed": False,
        "frozen_results_modified": False,
        "local_pair_index": str(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Stage 13C patient-safe pair design.")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    result = design(config, root)
    report_root = root / "reports/stage13"
    (report_root / "stage13c_patient_safe_pair_design_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = "\n".join(
        [
            "# Stage 13C Patient-Safe Multi-View Pair Design",
            "",
            f"- Status: `{result['status']}`",
            f"- Gate: `{result['gate']}`",
            f"- Exact frontal/lateral pairs: `{result['exact_pairs']}`",
            f"- Patient leakage violations: `{result['patient_leakage_violations']}`",
            f"- Record reuse violations: `{result['record_reuse_violations']}`",
            "- Heuristic pairs created: `0`",
            "- UNKNOWN records paired: `0`",
            "- Locked-test records accessed: `0`",
            "- Training performed: `false`",
            "",
            "Only studies with exactly one AP/PA frontal and one LATERAL record are paired. "
            "Ambiguous and duplicate structures remain withheld.",
            "",
        ]
    )
    (report_root / "STAGE13C_PATIENT_SAFE_PAIR_DESIGN_REPORT.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "PASSED_PATIENT_SAFE_MULTIVIEW_PAIR_DESIGN" else 2


if __name__ == "__main__":
    raise SystemExit(main())
