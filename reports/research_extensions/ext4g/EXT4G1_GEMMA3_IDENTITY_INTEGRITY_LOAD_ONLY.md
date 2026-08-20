# EXT-4G.1 — Gemma Identity, Integrity, and Load-Only Gate

The selected identity is `google/gemma-3-4b-it` at immutable revision
`093f9f388b31de276ce2de164bdc2081324b9767`. The governed local execution
completed the download and CPU-BF16 load-only gate. The final status is
`EXT4G1_GEMMA3_IDENTITY_INTEGRITY_AND_LOAD_ONLY_PASS`.

The candidate manifest contains exactly 15 revision files; Hugging Face
`.cache` bookkeeping is explicitly excluded. The safetensor index maps 883
tensors to the two verified shards. The local processor/tokenizer and
text-only chat template were deterministic. llguidance 1.8.0 compiled and
strictly validated the frozen realization schema, and the zero-inference
processor probe passed for logits width 262208.

The Gemma tokenizer is an identity prefix of 262145 IDs. Model-only IDs
262145–262207 (63 IDs) are classified as an unregistered vocabulary tail and
are permanently masked. The existing local model evidence records a
`Gemma3ForConditionalGeneration` CPU-BF16 load with 4,300,079,472 parameters,
zero forward calls, and zero generation calls.

Frozen policy remains `torch.bfloat16`, native CPU-only placement, no
`device_map`, no quantization, `llguidance==1.8.0`, and realization schema
SHA-256
`99a1c3a48a7bc262434bc52c116342b8dd81d741c74a549fa7ad4e2b6f4533a1`.
