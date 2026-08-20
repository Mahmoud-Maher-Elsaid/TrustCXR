# EXT-4G.1 — Gemma Identity, Integrity, and Load-Only Gate

The selected identity is `google/gemma-3-4b-it` at immutable revision
`093f9f388b31de276ce2de164bdc2081324b9767`. This repository is gated, and
the current Codex environment cannot perform the authenticated Hub download
or local CPU-BF16 load. The report therefore records
`EXT4G1_READY_FOR_GOVERNED_LOCAL_DOWNLOAD_AND_LOAD_ONLY`, not a fabricated
load pass.

The supplied runner downloads only that revision, hashes every file, checks
the safetensor index, audits the local processor/tokenizer and text-only chat
template, and optionally performs one CPU-only load-only check. It never calls
`forward` or `generate`, and it never opens EXT-4F benchmark or final data.

Frozen policy remains `torch.bfloat16`, native CPU-only placement, no
`device_map`, no quantization, `llguidance==1.8.0`, and realization schema
SHA-256
`99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1`.
