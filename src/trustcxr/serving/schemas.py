from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SCHEMA_VERSION = "trustcxr-serving-contract-v1"
PIPELINE_VERSION = "trustcxr-frozen-pipeline-v1"
RESEARCH_DESIGNATION = "RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"
REPORT_IDENTITY = "AI_GENERATED_RESEARCH_REPORT_DRAFT_FOR_EXPERT_REVIEW"
RESEARCH_DISCLAIMER = "Research use only. Not a medical diagnosis. Expert review is required."

BoundedToken = Annotated[
    str, StringConstraints(min_length=1, max_length=256, pattern=r"^[A-Za-z0-9._:-]+$")
]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ReasonCode = Annotated[str, StringConstraints(pattern=r"^[A-Z][A-Z0-9_]*$", max_length=96)]


class StrictContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class JobState(StrEnum):
    SUBMITTED = "SUBMITTED"
    VALIDATING = "VALIDATING"
    QUEUED = "QUEUED"
    GPU_PROCESSING = "GPU_PROCESSING"
    CPU_POSTPROCESSING = "CPU_POSTPROCESSING"
    VERIFYING = "VERIFYING"
    DECIDING = "DECIDING"
    COMPLETED = "COMPLETED"
    DEFERRED = "DEFERRED"
    FAILED_SANITIZED = "FAILED_SANITIZED"


class ComponentId(StrEnum):
    STAGE5 = "stage5_quality_view"
    STAGE9 = "stage9_classifier"
    STAGE10_11 = "stage10_11_limited_localization_fusion"
    STAGE16 = "stage16_reliability"
    STAGE17 = "stage17_triage"
    STAGE18 = "stage18_report_renderer"
    STAGE19 = "stage19_verifier"
    STAGE20 = "stage20_decision_support"


class VerifierStatus(StrEnum):
    VERIFIED = "VERIFIED"
    PARTIALLY_VERIFIED = "PARTIALLY_VERIFIED"
    UNVERIFIED = "UNVERIFIED"
    CONTRADICTED = "CONTRADICTED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    WITHHELD = "WITHHELD_INSUFFICIENT_EVIDENCE"


class Decision(StrEnum):
    DEFER = "DEFER"
    REVISE = "REVISE_DETERMINISTICALLY"
    ACCEPT = "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW"


class JobSubmission(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    input_token: BoundedToken
    pipeline_version: Literal["trustcxr-frozen-pipeline-v1"] = PIPELINE_VERSION
    idempotency_key: BoundedToken


class JobStatus(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    state: JobState
    research_designation: Literal["RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"] = RESEARCH_DESIGNATION
    reason_codes: tuple[ReasonCode, ...] = ()


class ComponentProvenance(StrictContractModel):
    component_id: ComponentId
    server_model_version: BoundedToken
    config_sha256: Sha256
    checkpoint_sha256: Sha256 | None = None


class EvidenceItem(StrictContractModel):
    evidence_code: ReasonCode
    source_component: ComponentId
    structured_field: BoundedToken
    value: bool | int | float | tuple[ReasonCode, ...]


class StructuredEvidence(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    pipeline_version: Literal["trustcxr-frozen-pipeline-v1"] = PIPELINE_VERSION
    component_provenance: tuple[ComponentProvenance, ...]
    structured_evidence: tuple[EvidenceItem, ...]
    limitations: tuple[ReasonCode, ...]


class ReportStatement(StrictContractModel):
    template_id: BoundedToken
    canonical_text: Annotated[str, StringConstraints(min_length=1, max_length=512)]
    source_stage: BoundedToken
    source_version: BoundedToken
    evidence_code: ReasonCode
    structured_source_field: BoundedToken


class ResearchReportDraft(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    report_identity: Literal["AI_GENERATED_RESEARCH_REPORT_DRAFT_FOR_EXPERT_REVIEW"] = (
        REPORT_IDENTITY
    )
    research_use_disclaimer: Literal[
        "Research use only. Not a medical diagnosis. Expert review is required."
    ] = RESEARCH_DISCLAIMER
    statements: tuple[ReportStatement, ...]
    omitted_capabilities: tuple[ReasonCode, ...]


class StatementVerification(StrictContractModel):
    template_id: BoundedToken
    status: VerifierStatus
    evidence_references: tuple[BoundedToken, ...]
    reason_codes: tuple[ReasonCode, ...]


class VerifierResult(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    statement_results: tuple[StatementVerification, ...]
    evidence_references: tuple[BoundedToken, ...]
    limitations: tuple[ReasonCode, ...]


class DecisionResult(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    decision: Decision
    reason_codes: tuple[ReasonCode, ...]
    evidence_references: tuple[BoundedToken, ...]
    research_designation: Literal["RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"] = RESEARCH_DESIGNATION


class Disposition(StrEnum):
    SAFETY_DEFER = "SAFETY_DEFER"
    TECHNICAL_FAILURE = "TECHNICAL_FAILURE"


class SanitizedDisposition(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    state: Literal[JobState.DEFERRED, JobState.FAILED_SANITIZED]
    disposition: Disposition
    reason_codes: tuple[ReasonCode, ...]
    research_designation: Literal["RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"] = RESEARCH_DESIGNATION


class WorkerRequest(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    component_id: ComponentId
    input_token: BoundedToken
    server_model_version: BoundedToken
    request_fingerprint: Sha256
    config_sha256: Sha256
    checkpoint_sha256: Sha256 | None = None


class WorkerResponse(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    job_id: BoundedToken
    component_id: ComponentId
    status: Literal["SUCCESS", "DEFERRED", "FAILED_SANITIZED"]
    structured_output: dict[str, bool | int | float | str] = Field(default_factory=dict)
    provenance: ComponentProvenance
    reason_codes: tuple[ReasonCode, ...]


class HealthResponse(StrictContractModel):
    schema_version: Literal["trustcxr-serving-contract-v1"] = SCHEMA_VERSION
    status: Literal["READY_CONTRACT_ONLY"] = "READY_CONTRACT_ONLY"
    research_designation: Literal["RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED"] = RESEARCH_DESIGNATION
    pipeline_version: Literal["trustcxr-frozen-pipeline-v1"] = PIPELINE_VERSION
