# EXT-4E2C Qwen3 Load-Only GPU Smoke

Status: `LOAD_ONLY_GPU_SMOKE_PREPARED`

Observed local result: `LOAD_ONLY_PASS` — the candidate is technically loadable and has not been
scientifically evaluated. The recorded load took 4.098 seconds, used approximately 5017 MiB of the
8192 MiB GPU, and shut down cleanly. No inference or benchmark access occurred.

This is technical load validation only. It does not evaluate language behavior, read benchmark
cases, generate tokens, or select a scientific model.

## Frozen attempt

- Model: `Qwen/Qwen3-8B-GGUF`, revision
  `6a569868d07d3bd59e8b97fb001bf8c0b254bb20`
- File: `Qwen3-8B-Q4_K_M.gguf`
- SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
- Runtime: llama.cpp `b10453`, commit prefix `3cb7ffb`, CUDA 12.4
- Context: `2048`
- GPU policy: first attempt uses `--n-gpu-layers 999`; no silent CPU fallback or retry
- Binding: `127.0.0.1:18080`
- Timeout: 180 seconds

The script starts `llama-server`, waits for the non-generative `/health` endpoint, samples
`nvidia-smi`, and then shuts the process down. It never sends a completion or chat request.

The pinned build's local `--help` output confirms `--cors-origins`, `--no-webui`, and `--parallel`.
Future first-case execution should use `--host 127.0.0.1 --cors-origins localhost --no-webui
--parallel 1`; API-key authentication is not added because the server remains localhost-only.

## Outcomes

The machine-readable evidence records one of `LOAD_ONLY_PASS`, `GPU_OOM`, `LOAD_TIMEOUT`,
`RUNTIME_FAILURE`, `MODEL_HASH_MISMATCH`, `MODEL_LOAD_FAILURE`, `CUDA_BACKEND_FAILURE`, or
`UNEXPECTED_PROCESS_EXIT`. An OOM is a technical result, not scientific model evidence.

Evidence is written under the ignored path
`artifacts/research_extensions/ext4e2_candidate1/load_only_smoke/` as
`load_only_smoke.json`, `llama_server.log`, and `vram_samples.csv`. Cleanup is required and is
recorded explicitly. Future JSON evidence records numeric VRAM fields. The tracked closure summary
labels the existing numeric values as `DERIVED_FROM_RECORDED_VRAM_SAMPLES` and preserves the raw
historical JSON/CSV unchanged.

## Isolation

The script does not read EXT-4D fixtures, development cases, final cases, patient data, or locked
test data. `inference_performed` remains false and all case-access counters remain zero.
