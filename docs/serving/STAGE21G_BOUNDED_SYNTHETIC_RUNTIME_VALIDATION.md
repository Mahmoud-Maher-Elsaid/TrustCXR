# Stage 21G Bounded Synthetic Runtime Validation

Stage 21G exercises the real FastAPI ASGI application, job store, state machine, immutable
registry, bounded worker control flow, safety propagation, idempotency, privacy logging, and
temporary-artifact lifecycle using synthetic non-patient fixtures only.

The runtime uses an in-process ASGI harness, so no socket or server process is opened and no
additional HTTP-client dependency is required. It performs no checkpoint deserialization,
model loading, preprocessing, inference, GPU profiling, patient processing, locked-test
access, persistent worker execution, or LLM operation. CUDA failures are simulated control
states only.

Stage 21G records category-level pass/fail counts and fails closed on any failed case or
incomplete cleanup. Its output may open Stage 21H, a synthetic-runtime acceptance decision;
Stage 21H does not authorize real model loading, real inference, GPU residency profiling, or
LLM work.
