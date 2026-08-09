# Stage 21E Serving Dependency and Implementation Protocol Freeze

Stage 21E is prepared as governance/freeze only. It proposes FastAPI 0.116.1, Pydantic
2.11.7, and Uvicorn 0.35.0 as the minimum direct local HTTP/schema/ASGI set. The complete
transitive closure is exact-version and wheel-hash pinned in
`requirements/lock-serving-stage21.txt`. A Python 3.12 no-install resolver dry run passed;
no serving package was installed.

The selected cohort is a conservative mutually compatible mid-2025 set rather than an
unreviewed latest-version choice. FastAPI supplies the bounded API layer, Pydantic supplies
deterministic schema validation, and Uvicorn supplies the required local ASGI server.
Python's standard library remains sufficient for bounded local coordination, temporary
storage, sanitized logging, shutdown, and cleanup. Redis, Celery, databases, brokers,
orchestration, and cloud services remain prohibited.

The frozen architecture is one local API process, one local GPU execution path, sequential
one-model loading, and deterministic CPU post-processing. Checkpoint hashes are verified
before deserialization, checkpoints remain immutable, CUDA is checked before GPU loading,
and each GPU model is released before the next is loaded. A residency profile is unnecessary
for this sequential architecture but remains mandatory before any future multi-model
residency authorization.

Stage 21E installs nothing and implements or starts no serving component. If its gate passes,
Stage 21F may implement only the frozen minimal local backend/worker architecture. It may not
perform GPU residency profiling or add any LLM capability.
