from __future__ import annotations

from trustcxr.serving.schemas import JobState

TRANSITIONS: dict[JobState, frozenset[JobState]] = {
    JobState.SUBMITTED: frozenset({JobState.VALIDATING}),
    JobState.VALIDATING: frozenset({JobState.QUEUED, JobState.DEFERRED, JobState.FAILED_SANITIZED}),
    JobState.QUEUED: frozenset({JobState.GPU_PROCESSING, JobState.FAILED_SANITIZED}),
    JobState.GPU_PROCESSING: frozenset(
        {JobState.CPU_POSTPROCESSING, JobState.DEFERRED, JobState.FAILED_SANITIZED}
    ),
    JobState.CPU_POSTPROCESSING: frozenset(
        {JobState.VERIFYING, JobState.DEFERRED, JobState.FAILED_SANITIZED}
    ),
    JobState.VERIFYING: frozenset(
        {JobState.DECIDING, JobState.DEFERRED, JobState.FAILED_SANITIZED}
    ),
    JobState.DECIDING: frozenset(
        {JobState.COMPLETED, JobState.DEFERRED, JobState.FAILED_SANITIZED}
    ),
    JobState.COMPLETED: frozenset(),
    JobState.DEFERRED: frozenset(),
    JobState.FAILED_SANITIZED: frozenset(),
}


class IllegalTransitionError(RuntimeError):
    pass


def transition(current: JobState, target: JobState) -> JobState:
    if target not in TRANSITIONS[current]:
        raise IllegalTransitionError("ILLEGAL_JOB_STATE_TRANSITION")
    return target
