"""Dataset adapter contracts and privacy-safe identity resolution."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class IdentityRule:
    """A single configured source for one canonical identifier."""

    source: str
    confidence: str
    column: str | None = None
    tag: str | None = None
    regex: str | None = None
    keyword: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> IdentityRule:
        """Build an identity rule from the generated adapter plan."""
        return cls(
            source=str(value.get("source", "UNRESOLVED")),
            confidence=str(value.get("confidence", "NONE")),
            column=(str(value["column"]) if value.get("column") else None),
            tag=(str(value["tag"]) if value.get("tag") else None),
            regex=(str(value["regex"]) if value.get("regex") else None),
            keyword=(str(value["keyword"]) if value.get("keyword") else None),
        )


@dataclass(frozen=True)
class DatasetAdapterContract:
    """Generated mapping contract for one dataset."""

    dataset_id: str
    folder: str
    adapter_status: str
    patient_split_ready: bool
    patient_rule: IdentityRule
    study_rule: IdentityRule
    image_rule: IdentityRule

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
    ) -> DatasetAdapterContract:
        """Build a contract from generated JSON-compatible data."""
        rules = value.get("identity_rules", {})

        return cls(
            dataset_id=str(value["dataset_id"]),
            folder=str(value["folder"]),
            adapter_status=str(value["adapter_status"]),
            patient_split_ready=bool(value["patient_split_ready"]),
            patient_rule=IdentityRule.from_mapping(rules.get("patient", {})),
            study_rule=IdentityRule.from_mapping(rules.get("study", {})),
            image_rule=IdentityRule.from_mapping(rules.get("image", {})),
        )


def load_adapter_contracts(
    path: Path,
) -> dict[str, DatasetAdapterContract]:
    """Load generated adapter contracts keyed by dataset ID."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    datasets = raw.get("datasets")

    if not isinstance(datasets, list):
        raise ValueError("Adapter plan does not contain datasets.")

    contracts = [DatasetAdapterContract.from_mapping(item) for item in datasets]

    return {contract.dataset_id: contract for contract in contracts}


def canonicalize_identifier(
    *,
    dataset_id: str,
    namespace: str,
    value: str,
) -> str:
    """Create a stable privacy-safe canonical identifier."""
    normalized = value.strip()

    if not normalized:
        raise ValueError("Identifier value must not be empty.")

    digest = hashlib.sha256((f"{dataset_id}\x1f{namespace}\x1f{normalized}").encode()).hexdigest()

    return f"{dataset_id}:{namespace}:{digest[:24]}"


def _resolve_path_component(
    relative_path: str,
) -> str | None:
    for component in relative_path.replace("\\", "/").split("/"):
        lowered = component.lower()

        if "patient" in lowered or "subject" in lowered:
            return component

    return None


def resolve_raw_identifier(
    *,
    rule: IdentityRule,
    metadata_row: Mapping[str, Any] | None = None,
    relative_path: str | None = None,
    dicom_header: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve one raw identifier using a generated rule."""
    metadata_row = metadata_row or {}
    dicom_header = dicom_header or {}

    if rule.source == "METADATA_COLUMN" and rule.column:
        value = metadata_row.get(rule.column)
        return str(value).strip() if value is not None else None

    if rule.source == "DICOM_TAG" and rule.tag:
        value = dicom_header.get(rule.tag)
        return str(value).strip() if value is not None else None

    if rule.source == "FILENAME_REGEX" and rule.regex and relative_path:
        match = re.match(
            rule.regex,
            Path(relative_path).name,
            re.IGNORECASE,
        )
        return match.group(1) if match else None

    if rule.source == "PATH_COMPONENT" and relative_path:
        return _resolve_path_component(relative_path)

    if rule.source == "RELATIVE_PATH" and relative_path:
        return relative_path.replace("\\", "/")

    return None


def resolve_canonical_identifier(
    *,
    dataset_id: str,
    namespace: str,
    rule: IdentityRule,
    metadata_row: Mapping[str, Any] | None = None,
    relative_path: str | None = None,
    dicom_header: Mapping[str, Any] | None = None,
) -> str | None:
    """Resolve and hash one canonical identifier."""
    raw_value = resolve_raw_identifier(
        rule=rule,
        metadata_row=metadata_row,
        relative_path=relative_path,
        dicom_header=dicom_header,
    )

    if not raw_value:
        return None

    return canonicalize_identifier(
        dataset_id=dataset_id,
        namespace=namespace,
        value=raw_value,
    )


def resolve_record_identity(
    *,
    contract: DatasetAdapterContract,
    metadata_row: Mapping[str, Any] | None = None,
    relative_path: str | None = None,
    dicom_header: Mapping[str, Any] | None = None,
) -> dict[str, str | None]:
    """Resolve canonical patient, study, and image identifiers."""
    return {
        "patient_id": resolve_canonical_identifier(
            dataset_id=contract.dataset_id,
            namespace="patient",
            rule=contract.patient_rule,
            metadata_row=metadata_row,
            relative_path=relative_path,
            dicom_header=dicom_header,
        ),
        "study_id": resolve_canonical_identifier(
            dataset_id=contract.dataset_id,
            namespace="study",
            rule=contract.study_rule,
            metadata_row=metadata_row,
            relative_path=relative_path,
            dicom_header=dicom_header,
        ),
        "image_id": resolve_canonical_identifier(
            dataset_id=contract.dataset_id,
            namespace="image",
            rule=contract.image_rule,
            metadata_row=metadata_row,
            relative_path=relative_path,
            dicom_header=dicom_header,
        ),
    }
