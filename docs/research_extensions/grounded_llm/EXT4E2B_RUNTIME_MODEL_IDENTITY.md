# EXT-4E2B Runtime and Model Identity Freeze

Status: `RUNTIME_AND_MODEL_IDENTITY_FROZEN`

This preparation freezes Candidate #1 for local development preflight. It does not select a final
model, download weights, load a model, or perform inference.

## Candidate identity

- Repository: `Qwen/Qwen3-8B-GGUF`
- Revision: `6a569868d07d3bd59e8b97fb001bf8c0b254bb20`
- File: `Qwen3-8B-Q4_K_M.gguf`
- Quantization: `Q4_K_M`
- SHA-256: `d98cdcbd03e17ce47681435b5150e34c1417f50b5c0019dd560e4882c5745785`
- License: `Apache-2.0`
- Mode: text-only, non-thinking
- Status: development candidate 1; final selection not made

## Runtime identity

- Project: `ggml-org/llama.cpp`
- Release: `b10453`
- Commit: `3cb7ffb`
- Platform/backend: Windows x64, CUDA 12.4
- Runtime asset: `llama-b10453-bin-win-cuda-12.4-x64.zip`
- CUDA runtime asset: `cudart-llama-bin-win-cuda-12.4-x64.zip`
- CUDA runtime SHA-256: `8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6`

The official release metadata identifies the pinned release, commit, and assets, but the main
Windows CUDA runtime asset digest was not exposed during this preparation. Its value remains
`TO_BE_RESOLVED_OFFICIAL_RELEASE_METADATA`. The bootstrap fails closed until the official digest is
resolved; it must not substitute another release, backend, or unverified local digest.

## Isolated paths

- Runtime: `cache/research_extensions/ext4e2/llama.cpp`
- Models: `cache/research_extensions/ext4e2/models`
- Downloads: `cache/research_extensions/ext4e2/downloads`
- Evidence: `artifacts/research_extensions/ext4e2_candidate1`

These paths are local/ignored. Runtime binaries, model weights, and logs are not tracked.

## Fail-closed bootstrap

Run from a clean compatible descendant of the preparation base:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
  ".\scripts\training\run_ext4e2_candidate1_preflight.ps1"
```

The script verifies the branch, EXT-4D fingerprints, pinned identities, and development-only
partition before downloading. It verifies every downloaded SHA-256 and stops before any model
inference. Because the main runtime digest is unresolved, this command currently stops before
network download, by design.

## Execution boundary

Only the six EXT-4D development cases may be used in later runtime work. The 24 frozen final cases
are rejected by the partition guard and are not read by this bootstrap. External APIs, patient data,
raw images, training, fine-tuning, and inference are outside this stage.
