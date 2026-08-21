# EXT-4H.G1 GPU INT8 Runtime Closure

The governed run `20260820T034507Z_f202cbd4` passed the Gemma GPU INT8
technical smoke. The complete quantized model executed on `cuda:0` with
400 `Linear8bitLt` modules and no CPU model fallback. All three deterministic
slots completed exactly once, with explicit CUDA attention masks, preserved
Gemma vocabulary-tail masking, zero authority mutations, and successful
cleanup.

The observed `MatMul8bitLt` message that BF16 inputs are cast to FP16 during
INT8 quantization is expected runtime behavior and is recorded, not hidden.
This is a distinct CPU-BF16/GPU-INT8 execution variant; no performance claim
is made between them.

EXT-4F historical cases, final cases, and locked data remained unopened.
The next authorized stage is EXT-4H.3 fresh development benchmark design and
freeze. No benchmark execution is authorized by this closure.
