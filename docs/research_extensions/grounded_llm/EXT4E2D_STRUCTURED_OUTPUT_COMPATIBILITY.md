# EXT-4E2D Structured-Output Compatibility Repair

The first three Candidate #1 development smoke attempts were technical request failures,
not scientific evaluations. Qwen3 loaded successfully, but the pinned llama.cpp b10453
server applied a startup JSON grammar while accepting the Qwen Chat Completions assistant
prefill. The prefill began with `<|im_start|>assistant` and thinking-template content, so
the JSON grammar failed before token generation.

The observed `response_format.type = json_schema` request was accepted by the pinned
runtime but returned a schema-violating object. This is a technical structured-output
compatibility defect, not prompt tuning or scientific model selection. The repaired
architecture uses the pinned documented Chat Completions form `json_object` with the
same schema object.

The repaired architecture separates the concerns:

- llama-server starts without `--json-schema-file`, `--json-schema`, `--grammar`, or
  `--grammar-file`.
- Chat Completions carries exactly one request-level constraint:

  ```json
  "response_format": {
    "type": "json_object",
    "schema": "<generated EXT-4C JSON Schema>"
  }
  ```

- The pinned server remains local-only, single-slot, context 2048, and `--reasoning off`.
- The request records `reasoning_effort = "none"` as the explicit non-thinking request
  setting; no reasoning content is stored.
- EXT-4C `GroundedOutputEnvelope` validation remains mandatory after JSON parsing. Native
  constrained decoding is only a transport/generation safeguard and does not replace
  EXT-4B grounding or EXT-4C validation.

`EXT-4E2D0` is a separate synthetic compatibility smoke. It uses one non-clinical request,
zero retries, no EXT-4D case, no patient data, and no locked-test material. A successful
result establishes only request-level structured-output transport compatibility; it does
not evaluate Candidate #1 scientifically and does not authorize the development-case
smoke automatically.
