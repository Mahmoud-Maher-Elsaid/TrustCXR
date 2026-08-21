# EXT-4H.4 Automatic Gate Closure

The completed run `20260820T070353Z_fc35947a` is valid through the frozen
automatic gate. The apparent 84/80 discrepancy is expected: the benchmark
contains 84 total request/semantic slots, while four DEFER reviewer-question
slots are explicitly `required=False` in the frozen realization request.
`build_ext4h_slot_manifest` includes required slots only, so exactly 80
model-authored `slot_text` generations were required and executed.

The four non-generative topics are `slot_reviewer_defer_reason` in cases
009, 010, 018, and 022. They are deterministic optional review topics, not
missing model generations. Automatic evidence is 80/80 slot completion and
contract validity, 24/24 assembled and final-valid cases, zero authority
mutations, zero protocol deviations, and zero retries.

Status remains `EXT4H4_AUTOMATIC_GATE_PASS_REVIEW_REQUIRED`. Gemma is not
scientifically selected; semantic review is the next governed stage.
