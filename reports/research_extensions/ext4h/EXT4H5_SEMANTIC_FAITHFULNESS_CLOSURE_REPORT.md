# EXT-4H.5 semantic-faithfulness closure

The supplied 80-unit blinded review was imported against the frozen bundle
without changing ratings. There were 37 resolved units and 43 unresolved
units. Unresolved metadata was preserved and maps to failure for final
selection under the frozen protocol.

The import produced 765 applicable decisions, 513 PASS and 252 FAIL
(0.6705882352941176). Resolved failures alone affect 16 slots across 12
cases and impose 64 unavoidable applicable FAIL decisions. Even granting
every unresolved judgment PASS, the maximum semantic rate is
701/765 = 0.9163398692810457 and the maximum case rate is 12/24 = 0.5.
Selection therefore cannot be rescued by further adjudication.

H4 remains technically valid: `EXT4H4_AUTOMATIC_GATE_PASS_REVIEW_REQUIRED`.
H5 is `EXT4H5_SEMANTIC_FAITHFULNESS_FAILED`; the candidate is not
scientifically selected. No model was loaded or run during closure.
