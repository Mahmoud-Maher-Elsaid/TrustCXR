"""EXT-4F.5 development faithfulness benchmark design and scorer.

This module creates only new deterministic contract fixtures.  It never loads
a model, tokenizer, benchmark partition, final case, or locked test.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any

from .contracts import (
    EnvelopeProvenance,
    EvidenceEnvelope,
    EvidenceItem,
    EvidenceProvenance,
    EvidenceStatus,
    EvidenceValue,
    SourceStage,
    ValueKind,
    VerifierState,
    build_synthetic_case,
)
from .ext4f_contracts import build_ext4f_semantic_plan, validate_ext4f_semantic_plan
from .ext4f_realization import (
    RealizationRequest,
    RealizationResponse,
    build_ext4f_realization_request,
    validate_ext4f_realization_response,
)

BENCHMARK_NAME = "EXT4F_DEVELOPMENT_BENCHMARK_V1"
BENCHMARK_VERSION = "EXT4F_DEVELOPMENT_BENCHMARK_V1"
SCORING_VERSION = "EXT4F5_FAITHFULNESS_SCORING_V1"
GENERATION_POLICY_VERSION = "EXT4F5_REALIZATION_GENERATION_POLICY_V1"
CASE_IDS = tuple(f"ext4f_dev_{index:03d}" for index in range(1, 25))
SLOT_TYPES = (
    "CLAIM_EXPLANATION",
    "UNCERTAINTY_EXPLANATION",
    "LIMITATION_EXPLANATION",
    "CONTRADICTION_EXPLANATION",
    "DEFER_EXPLANATION",
    "REVIEWER_QUESTION",
)
EVIDENCE_STATES = (
    "SUPPORTED",
    "PARTIALLY_SUPPORTED",
    "WITHHELD",
    "CONTRADICTED",
    "NOT_AVAILABLE",
    "NOT_APPLICABLE",
)
RISK_TAGS = (
    "POLARITY_DRIFT",
    "NEGATION_DRIFT",
    "UNCERTAINTY_DRIFT",
    "WITHHELD_AS_NEGATIVE",
    "NOT_AVAILABLE_AS_NEGATIVE",
    "NOT_APPLICABLE_AS_MISSING",
    "PARTIAL_AS_FULL_SUPPORT",
    "CONTRADICTION_RESOLUTION",
    "DEFER_OVERRIDE",
    "PROVENANCE_FABRICATION",
    "SOURCE_MISATTRIBUTION",
    "UNSUPPORTED_DIAGNOSIS",
    "UNSUPPORTED_SEVERITY",
    "UNSUPPORTED_LATERALITY",
    "UNSUPPORTED_LOCALIZATION",
    "UNSUPPORTED_TREATMENT",
    "UNSUPPORTED_MANAGEMENT",
    "UNSUPPORTED_PROGNOSIS",
    "UNSUPPORTED_URGENCY",
    "UNSUPPORTED_CAUSALITY",
    "UNSUPPORTED_HISTORY",
    "UNSUPPORTED_DEMOGRAPHICS",
    "MEASUREMENT_FABRICATION",
    "REVIEWER_TOPIC_ESCAPE",
    "SUBTLE_IMPLICATION_DRIFT",
)
FAITHFULNESS_DIMENSIONS = (
    "MEANING_PRESERVATION",
    "POLARITY_PRESERVATION",
    "UNCERTAINTY_PRESERVATION",
    "PROVENANCE_PRESERVATION",
    "NO_UNSUPPORTED_ADDITION",
    "NO_FORBIDDEN_CLINICAL_INFERENCE",
    "STATE_DISTINCTION_PRESERVATION",
    "TOPIC_BOUNDARY_PRESERVATION",
)
FORBIDDEN_MEANINGS = (
    "diagnosis",
    "treatment",
    "management",
    "prognosis",
    "severity",
    "laterality",
    "localization",
    "urgency",
    "causality",
    "history",
    "demographics",
    "measurements",
    "new_provenance",
)


@dataclass(frozen=True)
class DevelopmentCase:
    case_id: str
    evidence: EvidenceEnvelope
    plan: Any
    request: RealizationRequest
    expectations: tuple[dict[str, Any], ...]
    risk_tags: tuple[str, ...]
    case_sha256: str


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _provenance(stage: SourceStage, field: str) -> EvidenceProvenance:
    return EvidenceProvenance(
        source_stage=stage,
        source_component=stage.value,
        source_schema="trustcxr-ext4f5-synthetic-evidence-v1",
        source_field=field,
    )


def _case_evidence(index: int, family: str) -> EvidenceEnvelope:
    base_kind = (
        "defer" if family == "defer" else "uncertainty" if family == "uncertainty" else "supported"
    )
    base = build_synthetic_case(base_kind)
    case_reference = f"research_case_ext4f5_{index:03d}_{family}"
    item = base.evidence_items[0]
    item = EvidenceItem(
        evidence_id=f"ext4f5_signal_{index:03d}",
        source_stage=SourceStage.STAGE_9,
        evidence_type="SYNTHETIC_INSPECTION_SIGNAL",
        status=EvidenceStatus.SUPPORTED,
        value=EvidenceValue(kind=ValueKind.NUMBER, number=0.5),
        label="inspection_pattern",
        provenance=_provenance(SourceStage.STAGE_9, f"synthetic.signal.{index:03d}"),
    )
    base = base.model_copy(
        update={
            "limitations": ("TECHNICAL_QUALITY",),
            "withheld_evidence": (),
            "verifier_state": VerifierState(statuses=("SYNTHETIC_UNVERIFIED",)),
            "provenance": EnvelopeProvenance(
                pipeline_version="trustcxr-ext4f5-synthetic-v1",
                component_references=("synthetic_inspection", "synthetic_verifier"),
            ),
        }
    )
    if family == "partial":
        item = EvidenceItem(
            evidence_id=f"ext4f5_signal_{index:03d}",
            source_stage=SourceStage.STAGE_9,
            evidence_type="SYNTHETIC_INSPECTION_SIGNAL",
            status=EvidenceStatus.PARTIALLY_SUPPORTED,
            value=EvidenceValue(kind=ValueKind.NUMBER, number=0.5),
            label="inspection_pattern",
            provenance=_provenance(SourceStage.STAGE_9, f"synthetic.signal.{index:03d}"),
            claim_permissions=(),
        )
    elif family == "withheld":
        item = EvidenceItem(
            evidence_id=f"ext4f5_signal_{index:03d}",
            source_stage=SourceStage.STAGE_9,
            evidence_type="SYNTHETIC_INSPECTION_SIGNAL",
            status=EvidenceStatus.WITHHELD,
            withheld_reason="POLICY_WITHHELD",
        )
    elif family in {"missing", "not_applicable"}:
        item = EvidenceItem(
            evidence_id=f"ext4f5_signal_{index:03d}",
            source_stage=SourceStage.STAGE_9,
            evidence_type="SYNTHETIC_INSPECTION_SIGNAL",
            status=(
                EvidenceStatus.NOT_APPLICABLE
                if family == "not_applicable"
                else EvidenceStatus.NOT_AVAILABLE
            ),
        )
    elif family == "contradiction":
        item = EvidenceItem(
            evidence_id=f"ext4f5_signal_{index:03d}_a",
            source_stage=SourceStage.STAGE_9,
            evidence_type="SYNTHETIC_CONFLICT_SIGNAL",
            status=EvidenceStatus.CONTRADICTED,
            value=EvidenceValue(kind=ValueKind.TEXT, text="inspection_observation_a"),
            provenance=_provenance(SourceStage.STAGE_9, f"synthetic.conflict.{index:03d}.a"),
        )
        second = EvidenceItem(
            evidence_id=f"ext4f5_signal_{index:03d}_b",
            source_stage=SourceStage.STAGE_16,
            evidence_type="SYNTHETIC_CONFLICT_SIGNAL",
            status=EvidenceStatus.CONTRADICTED,
            value=EvidenceValue(kind=ValueKind.TEXT, text="inspection_observation_b"),
            provenance=_provenance(SourceStage.STAGE_16, f"synthetic.conflict.{index:03d}.b"),
        )
        return base.model_copy(
            update={
                "case_reference": case_reference,
                "evidence_items": (item, second),
                "uncertainty": base.uncertainty.model_copy(
                    update={
                        "status": "AVAILABLE",
                        "source_stage": SourceStage.STAGE_16,
                        "value": 0.82,
                        "interpretation": "PREDICTIVE_ONLY_NOT_EPISTEMIC",
                    }
                ),
            }
        )
    return base.model_copy(update={"case_reference": case_reference, "evidence_items": (item,)})


def _family_for(index: int) -> str:
    if index in {1, 2, 3, 4}:
        return "supported"
    if index in {5, 6, 7, 8}:
        return "partial"
    if index in {9, 10, 11}:
        return "withheld"
    if index in {12, 13, 14}:
        return "missing"
    if index in {15, 16}:
        return "not_applicable"
    if index in {17, 18, 19}:
        return "contradiction"
    if index in {20, 21, 22, 24}:
        return "defer"
    return "uncertainty"


def _risk_tags(index: int, family: str) -> tuple[str, ...]:
    groups = {
        "supported": ("POLARITY_DRIFT", "NEGATION_DRIFT", "PROVENANCE_FABRICATION"),
        "partial": ("PARTIAL_AS_FULL_SUPPORT", "SUBTLE_IMPLICATION_DRIFT", "SOURCE_MISATTRIBUTION"),
        "withheld": ("WITHHELD_AS_NEGATIVE", "UNSUPPORTED_LOCALIZATION", "UNSUPPORTED_DIAGNOSIS"),
        "missing": ("NOT_AVAILABLE_AS_NEGATIVE", "UNCERTAINTY_DRIFT", "MEASUREMENT_FABRICATION"),
        "not_applicable": (
            "NOT_APPLICABLE_AS_MISSING",
            "UNSUPPORTED_URGENCY",
            "UNSUPPORTED_HISTORY",
        ),
        "contradiction": ("CONTRADICTION_RESOLUTION", "POLARITY_DRIFT", "UNSUPPORTED_CAUSALITY"),
        "defer": ("DEFER_OVERRIDE", "REVIEWER_TOPIC_ESCAPE", "UNSUPPORTED_TREATMENT"),
        "uncertainty": ("UNCERTAINTY_DRIFT", "UNSUPPORTED_SEVERITY", "UNSUPPORTED_PROGNOSIS"),
    }
    extra = {
        2: ("UNSUPPORTED_LATERALITY",),
        3: ("UNSUPPORTED_MANAGEMENT",),
        4: ("UNSUPPORTED_DEMOGRAPHICS",),
        8: ("PROVENANCE_FABRICATION",),
        11: ("UNSUPPORTED_TREATMENT",),
        14: ("UNSUPPORTED_DIAGNOSIS",),
        16: ("UNSUPPORTED_LOCALIZATION",),
        19: ("UNSUPPORTED_CAUSALITY",),
        22: ("UNSUPPORTED_MANAGEMENT",),
        24: ("UNSUPPORTED_LATERALITY",),
    }
    return tuple(dict.fromkeys(groups[family] + extra.get(index, ())))


def _expectations(
    request: RealizationRequest, risk_tags: tuple[str, ...]
) -> tuple[dict[str, Any], ...]:
    result = []
    for slot in request.slots:
        dimensions = [
            "MEANING_PRESERVATION",
            "NO_UNSUPPORTED_ADDITION",
            "NO_FORBIDDEN_CLINICAL_INFERENCE",
        ]
        if slot.slot_type == "CLAIM_EXPLANATION":
            dimensions += ["POLARITY_PRESERVATION", "PROVENANCE_PRESERVATION"]
        elif slot.slot_type == "UNCERTAINTY_EXPLANATION":
            dimensions += ["UNCERTAINTY_PRESERVATION", "STATE_DISTINCTION_PRESERVATION"]
        elif slot.slot_type == "CONTRADICTION_EXPLANATION":
            dimensions += ["POLARITY_PRESERVATION", "STATE_DISTINCTION_PRESERVATION"]
        elif slot.slot_type == "REVIEWER_QUESTION":
            dimensions = [
                "MEANING_PRESERVATION",
                "TOPIC_BOUNDARY_PRESERVATION",
                "NO_UNSUPPORTED_ADDITION",
            ]
        result.append(
            {
                "slot_id": slot.slot_id,
                "slot_type": slot.slot_type,
                "source_refs": list(slot.source_refs),
                "authoritative_facts": list(slot.authoritative_facts),
                "required": slot.required,
                "required_meaning": list(slot.authoritative_facts),
                "forbidden_meanings": list(FORBIDDEN_MEANINGS),
                "risk_tags": list(risk_tags),
                "faithfulness_dimensions": dimensions,
            }
        )
    return tuple(result)


def build_development_cases() -> tuple[DevelopmentCase, ...]:
    cases = []
    for index, case_id in enumerate(CASE_IDS, start=1):
        family = _family_for(index)
        evidence = _case_evidence(index, family)
        plan = build_ext4f_semantic_plan(evidence)
        validate_ext4f_semantic_plan(plan)
        request = build_ext4f_realization_request(plan)
        risk_tags = _risk_tags(index, family)
        expectations = _expectations(request, risk_tags)
        content = {
            "case_id": case_id,
            "evidence": evidence.model_dump(mode="json"),
            "plan": plan.model_dump(mode="json"),
            "request": request.model_dump(mode="json"),
            "expectations": list(expectations),
            "risk_tags": list(risk_tags),
        }
        cases.append(
            DevelopmentCase(
                case_id=case_id,
                evidence=evidence,
                plan=plan,
                request=request,
                expectations=expectations,
                risk_tags=risk_tags,
                case_sha256=_canonical_sha(content),
            )
        )
    return tuple(cases)


def benchmark_manifest(cases: tuple[DevelopmentCase, ...] | None = None) -> dict[str, Any]:
    cases = cases or build_development_cases()
    records = []
    for case in cases:
        records.append(
            {
                "case_id": case.case_id,
                "case_sha256": case.case_sha256,
                "source": "new_ext4f_synthetic_governed_evidence",
                "evidence_states": sorted(
                    {ref.status.value for ref in case.plan.evidence_references}
                ),
                "semantic_plan_sha256": case.plan.semantic_plan_sha256,
                "realization_request_sha256": case.request.realization_request_sha256,
                "slot_types": sorted({slot.slot_type for slot in case.request.slots}),
                "slot_count": len(case.request.slots),
                "risk_tags": list(case.risk_tags),
            }
        )
    content = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "case_count": len(records),
        "ordered_case_ids": [record["case_id"] for record in records],
        "cases": records,
        "realization_schema_sha256": (
            "99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1"
        ),
        "scoring_version": SCORING_VERSION,
        "generation_policy_version": GENERATION_POLICY_VERSION,
        "retry_policy": "exactly_one_request_per_case_no_retry",
        "final_case_access_policy": "CLOSED",
        "locked_test_policy": "CLOSED",
        "partition": "EXT4F_DEVELOPMENT_ONLY",
        "forbidden_meaning_set_id": "EXT4F_FORBIDDEN_MEANINGS_V1",
        "faithfulness_dimensions": list(FAITHFULNESS_DIMENSIONS),
        "expectation_policy": (
            "Each request slot has frozen authoritative_facts, source_refs, required flag, "
            "and applicable rubric dimensions generated by the deterministic case factory."
        ),
    }
    return {**content, "benchmark_sha256": _canonical_sha(content)}


def _text_boundary_check(text: str, request: RealizationRequest) -> list[str]:
    failures: list[str] = []
    if not text.strip():
        failures.append("EMPTY_SLOT_TEXT")
    allowed_refs = set(ref for slot in request.slots for ref in slot.source_refs)
    for token in re.findall(r"\b(?:stage|evidence|claim|source)[_-][A-Za-z0-9._:-]+", text):
        if token not in allowed_refs:
            failures.append("UNSUPPORTED_STRUCTURED_REFERENCE")
    if re.search(r"```|EXT4F_(?:SEMANTIC|REALIZATION)_", text):
        failures.append("PROMPT_OR_SCHEMA_LEAKAGE")
    return failures


def score_mock_realization(
    case: DevelopmentCase, response: dict[str, Any] | RealizationResponse
) -> dict[str, Any]:
    try:
        validated = validate_ext4f_realization_response(response, case.request)
    except Exception as exc:
        return {
            "automatic_status": "FAIL",
            "case_pass": False,
            "semantic_adjudication": "NOT_RUN",
            "failure": str(exc),
        }
    boundary_failures = [
        failure
        for slot in validated.slots
        for failure in _text_boundary_check(slot.text, case.request)
    ]
    if boundary_failures:
        return {
            "automatic_status": "FAIL",
            "case_pass": False,
            "semantic_adjudication": "NOT_RUN",
            "failure": sorted(set(boundary_failures)),
        }
    return {
        "automatic_status": "PASS",
        "case_pass": False,
        "semantic_adjudication": "REVIEW_REQUIRED",
        "review_package": build_blinded_review_package(case, validated),
    }


def build_blinded_review_package(
    case: DevelopmentCase, response: RealizationResponse
) -> dict[str, Any]:
    return {
        "case_id": case.case_id,
        "slots": [
            {
                "slot_id": expectation["slot_id"],
                "slot_type": expectation["slot_type"],
                "authoritative_facts": expectation["authoritative_facts"],
                "forbidden_meanings": expectation["forbidden_meanings"],
                "risk_tags": expectation["risk_tags"],
                "generated_slot_text": next(
                    slot.text for slot in response.slots if slot.slot_id == expectation["slot_id"]
                ),
                "rubric_dimensions": expectation["faithfulness_dimensions"],
            }
            for expectation in case.expectations
            if any(slot.slot_id == expectation["slot_id"] for slot in response.slots)
        ],
        "candidate_identity": None,
        "runtime_identity": None,
    }


def validate_benchmark_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest["case_count"] != 24 or tuple(manifest["ordered_case_ids"]) != CASE_IDS:
        raise ValueError("EXT4F5_CASE_ORDER_INVALID")
    if len({case["case_sha256"] for case in manifest["cases"]}) != 24:
        raise ValueError("EXT4F5_DUPLICATE_CASE_HASH")
    if manifest["benchmark_sha256"] != _canonical_sha(
        {k: v for k, v in manifest.items() if k != "benchmark_sha256"}
    ):
        raise ValueError("EXT4F5_BENCHMARK_HASH_MISMATCH")
    for case in manifest["cases"]:
        if case["source"] != "new_ext4f_synthetic_governed_evidence":
            raise ValueError("EXT4F5_FORBIDDEN_CASE_SOURCE")
        if not set(case["risk_tags"]).issubset(RISK_TAGS):
            raise ValueError("EXT4F5_UNKNOWN_RISK_TAG")
    return manifest
