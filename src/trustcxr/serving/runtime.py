from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from threading import RLock

from trustcxr.serving.registry import FrozenComponentRegistry
from trustcxr.serving.schemas import (
    ComponentProvenance,
    Disposition,
    JobState,
    JobStatus,
    JobSubmission,
    SanitizedDisposition,
    WorkerRequest,
    WorkerResponse,
)
from trustcxr.serving.state_machine import transition

FAILURE_STATES = {
    "INVALID_REQUEST": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "UNSUPPORTED_INPUT": (JobState.DEFERRED, Disposition.SAFETY_DEFER),
    "DEPENDENCY_UNAVAILABLE": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "CHECKPOINT_HASH_MISMATCH": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "CUDA_UNAVAILABLE": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "CUDA_OOM": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "MODEL_LOAD_FAILURE": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "INFERENCE_FAILURE": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "PROVENANCE_FAILURE": (JobState.DEFERRED, Disposition.SAFETY_DEFER),
    "VERIFIER_FAILURE": (JobState.DEFERRED, Disposition.SAFETY_DEFER),
    "DECISION_POLICY_FAILURE": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
    "CLEANUP_FAILURE": (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE),
}


def pseudonymous_job_id(submission: JobSubmission) -> str:
    payload = f"{submission.pipeline_version}:{submission.idempotency_key}:{submission.input_token}"
    return "job_" + hashlib.sha256(payload.encode()).hexdigest()[:24]


def sanitized_disposition(job_id: str, reason_code: str) -> SanitizedDisposition:
    state, disposition = FAILURE_STATES.get(
        reason_code, (JobState.FAILED_SANITIZED, Disposition.TECHNICAL_FAILURE)
    )
    safe_reason = reason_code if reason_code in FAILURE_STATES else "INVALID_REQUEST"
    return SanitizedDisposition(
        job_id=job_id,
        state=state,
        disposition=disposition,
        reason_codes=(safe_reason,),
    )


class SanitizedLogger:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self._logger = logger or logging.getLogger("trustcxr.serving")

    def state(self, job_id: str, state: JobState, reason_codes: tuple[str, ...] = ()) -> None:
        self._logger.info(
            "job_id=%s state=%s reason_codes=%s", job_id, state.value, ",".join(reason_codes)
        )


class JobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, JobStatus] = {}
        self._fingerprints: dict[str, str] = {}
        self._lock = RLock()

    def submit(self, submission: JobSubmission) -> JobStatus:
        job_id = pseudonymous_job_id(submission)
        fingerprint = hashlib.sha256(submission.model_dump_json().encode()).hexdigest()
        with self._lock:
            if job_id in self._jobs:
                if self._fingerprints[job_id] != fingerprint:
                    raise RuntimeError("IDEMPOTENCY_CONFLICT")
                return self._jobs[job_id]
            status = JobStatus(job_id=job_id, state=JobState.SUBMITTED)
            self._jobs[job_id] = status
            self._fingerprints[job_id] = fingerprint
            return status

    def get(self, job_id: str) -> JobStatus | None:
        with self._lock:
            return self._jobs.get(job_id)

    def advance(self, job_id: str, target: JobState, reasons: tuple[str, ...] = ()) -> JobStatus:
        with self._lock:
            current = self._jobs[job_id]
            status = JobStatus(
                job_id=job_id, state=transition(current.state, target), reason_codes=reasons
            )
            self._jobs[job_id] = status
            return status


class TemporaryArtifactManager:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def create(self, job_id: str) -> Path:
        if not job_id.startswith("job_") or any(
            char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in job_id
        ):
            raise ValueError("INVALID_PSEUDONYMOUS_JOB_ID")
        self.root.mkdir(parents=True, exist_ok=True)
        destination = (self.root / job_id).resolve()
        if destination.parent != self.root:
            raise ValueError("REQUEST_SCOPE_ESCAPE_REJECTED")
        destination.mkdir(exist_ok=False)
        return destination

    def cleanup(self, job_id: str) -> None:
        destination = (self.root / job_id).resolve()
        if destination.parent != self.root:
            raise ValueError("REQUEST_SCOPE_ESCAPE_REJECTED")
        if destination.exists():
            shutil.rmtree(destination)

    def crash_recovery_cleanup(self) -> int:
        if not self.root.exists():
            return 0
        removed = 0
        for child in sorted(self.root.iterdir()):
            if child.is_dir() and child.name.startswith("job_"):
                shutil.rmtree(child)
                removed += 1
        return removed


@dataclass(frozen=True)
class SyntheticRuntimeState:
    cuda_available: bool = True
    simulated_failure: str | None = None


class BoundedWorker:
    """Contract-validating worker boundary; Stage 21F never deserializes a model."""

    safety_sequence = (
        "VALIDATE_REQUEST",
        "RESOLVE_FROZEN_SERVER_COMPONENT",
        "VERIFY_CONFIG_FINGERPRINT",
        "VERIFY_CHECKPOINT_SHA256",
        "VERIFY_CUDA_RUNTIME_IF_REQUIRED",
        "DESERIALIZE_AFTER_HASH_VALIDATION",
        "APPLY_EXACT_FROZEN_PREPROCESSING",
        "EXECUTE_ONLY_AUTHORIZED_COMPONENT",
        "RELEASE_GPU_MODEL_BEFORE_NEXT_GPU_MODEL",
        "PRESERVE_STRUCTURED_PROVENANCE",
        "PROPAGATE_ALL_SAFETY_STATES",
    )

    def __init__(self, registry: FrozenComponentRegistry) -> None:
        self.registry = registry
        self.resident_models = 0

    def validate_synthetic(
        self, request: WorkerRequest, state: SyntheticRuntimeState
    ) -> WorkerResponse | SanitizedDisposition:
        component = self.registry.resolve(request.component_id)
        if request.server_model_version != component.server_model_version:
            return sanitized_disposition(request.job_id, "PROVENANCE_FAILURE")
        if request.config_sha256 != component.config_sha256:
            return sanitized_disposition(request.job_id, "PROVENANCE_FAILURE")
        if request.checkpoint_sha256 != component.checkpoint_sha256:
            return sanitized_disposition(request.job_id, "CHECKPOINT_HASH_MISMATCH")
        if "GPU" in component.compute and not state.cuda_available:
            return sanitized_disposition(request.job_id, "CUDA_UNAVAILABLE")
        if state.simulated_failure:
            return sanitized_disposition(request.job_id, state.simulated_failure)
        provenance = ComponentProvenance(
            component_id=component.component_id,
            server_model_version=component.server_model_version,
            config_sha256=component.config_sha256,
            checkpoint_sha256=component.checkpoint_sha256,
        )
        return WorkerResponse(
            job_id=request.job_id,
            component_id=request.component_id,
            status="SUCCESS",
            structured_output={"synthetic_contract_validated": True},
            provenance=provenance,
            reason_codes=("SYNTHETIC_NO_MODEL_EXECUTION",),
        )
