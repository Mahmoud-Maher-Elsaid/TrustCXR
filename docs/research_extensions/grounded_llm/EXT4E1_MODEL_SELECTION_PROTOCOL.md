# EXT-4E1 Model Selection Protocol and Local Baseline Preflight

EXT-4E1 freezes a bounded, model-independent selection protocol before any model is run. It
does not select a model, download weights, call an API, or execute inference. EXT-4 remains a
local-only structured-text evidence extension; raw CXR images, DICOM, patient data, and the
locked benchmark are out of scope.

## Local resources and runtime boundary

The observed machine has an NVIDIA GeForce RTX 3070 Ti Laptop GPU with 8192 MiB VRAM and driver
592.82. No Ollama or llama.cpp server executable was detected. The repository environment exposes
Transformers command-line executables, but its Python launcher currently points to an unavailable
interpreter; this is recorded as a preflight condition, not repaired here.

The preferred future runtime is an isolated local Transformers runtime. llama.cpp is only a
fallback if already installed and compatible. No package installation, model download, external
provider, or network execution is authorized by EXT-4E1. Weights remain in an ignored local cache
and are never tracked. OOM or unsupported structured output fails closed; it must not silently
change model, quantization, or runtime.

## Bounded candidate pool

The included shortlist is Qwen/Qwen3-8B, mistralai/Ministral-3-8B-Instruct-2512, and
microsoft/Phi-4-mini-instruct. Llama-3.1-8B-Instruct is recorded as an optional excluded
comparison because the bounded protocol does not require a fourth candidate. Exact revisions,
license acceptance, quantization support, and runtime compatibility must be frozen before any
candidate execution. All candidates are text-only for EXT-4; vision features are prohibited.

Selection is not based on popularity or general benchmark scores. Candidate ranking is frozen as:
contract compatibility and safety/faithfulness first; then structured validity, grounding,
stability/repeatability, and latency/resource efficiency. Speed cannot compensate for a safety
failure.

## Development gate and final separation

Only the six EXT-4D development cases may be used for runtime debugging, format validation,
bounded interface work, and candidate comparison. Every candidate must produce valid EXT-4C
structured output with zero prohibited claims, DEFER violations, withheld-evidence violations,
fabricated provenance, unsupported localization/laterality/severity, and a reproducible local
configuration. The 24 frozen EXT-4D final cases are unavailable for selection, tuning, debugging,
or retries that optimize a candidate.

If multiple candidates pass, the predefined ranking rule is applied once. A candidate that fails
the development gate is not rescued by prompt or generation-parameter tuning on final cases.
The frozen final benchmark remains untouched until a later separately authorized execution stage.

## Configuration and prompt policy

Before candidate execution, freeze runtime identity/version, exact model revision, quantization,
context size, temperature, top-p/top-k, seed where supported, output limit, repetition penalty,
chat template, and the EXT-4C constrained-output mechanism. Temperature `0.0` is the default
low-variance policy; it is not a result-driven tuning choice.

The future grounded interface must carry EXT-4A governance, EXT-4B evidence, and EXT-4C output
semantics; preserve DEFER and WITHHELD localization; prohibit diagnosis, treatment, localization,
severity, and laterality; and emit JSON/schema-constrained output where supported. Any prompt or
interface development uses development cases only. No model-specific prompt is created in EXT-4E1.

## Future execution record

Each local run must record runtime identity/version, model revision, quantization, configuration
fingerprints, seed, generation parameters, structured-output mechanism, wall time, peak memory,
exit status, and failures. A successful preflight proves only practical local execution and
reproducible capture; it is not scientific benchmark success. The next authorized action is
`LOCAL DEVELOPMENT CANDIDATE EXECUTION`.
