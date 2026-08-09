# Stage 21B Backend API and Worker Contract

Stage 21B completed with `PASSED_BACKEND_API_AND_GPU_WORKER_CONTRACT` and opened
`GO_FOR_STAGE_21C_SYNTHETIC_API_WORKER_CONTRACT_VALIDATION`. The frozen contract
fingerprint is `6d92a8ddab32c2669d9e41b0b3f98bcac7485188f3b0ead628a7c2f87ae15c5f`.

The contract defines versioned bounded schemas, a fail-closed job state machine, a
hash-verified single-resident-model worker contract, privacy rules, sanitized failure
semantics, and deterministic orchestration. It started no server or worker, performed no
inference, used no patient records, accessed no locked test records, and authorized no
language-model capability.

## Dependency provenance

`pandas` is not a TrustCXR or Stage 21B dependency. It is absent from tracked imports,
`pyproject.toml`, requirements and lock files, CI configuration, dependency documentation,
and relevant Git history. Its import was requested only by the environment-recovery
preflight. Stage 21B uses Python standard-library modules and tracked TrustCXR modules, so
no pandas installation or dependency pin was added.

PyTorch `2.12.1+cu130` and torchvision `0.27.1+cu130` match the exact versions recorded in
the stage lock files and are governed existing dependencies. Stage 21B does not import or
execute them.

Stage 21C is limited to synthetic, non-patient API/worker contract validation. It does not
authorize backend/worker implementation or language-model work.
