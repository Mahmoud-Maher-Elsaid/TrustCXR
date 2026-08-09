# Stage 21F Minimal Local Backend/Worker Implementation

Stage 21F is prepared under Stage 21E dependency fingerprint
`b6fa4c0ec10301f9d6018610b4bd18b6a45270955ad7087b5ec62f55c594463f` and
implementation protocol fingerprint
`57cdfdc6df679ffe9e6ea38c797940423df1cc15af3cb1688d9871f01990fbb7`.

The implementation contains strict Pydantic request, evidence, report, verifier, decision,
worker, and sanitized-disposition schemas; three minimal FastAPI routes; the frozen job
state machine; an immutable server-side component registry; a bounded synthetic worker
validation boundary; deterministic idempotency; sanitized logging and failures; and
request-scoped ignored temporary storage with completion and crash cleanup.

The worker implementation validates server-owned component versions, config hashes,
checkpoint hashes, and injected CUDA state but cannot deserialize or execute a model in
Stage 21F. Public registry metadata omits internal paths. The API accepts bounded tokens only
and rejects extra model, path, URL, code, patient, and PHI-like fields.

No persistent server or worker, model loading, inference, GPU profiling, patient processing,
locked-test access, report generation, training, or LLM functionality is authorized. The
manual Stage 21F launcher performs only bounded synthetic implementation validation and
cleanup; it does not start Uvicorn.
