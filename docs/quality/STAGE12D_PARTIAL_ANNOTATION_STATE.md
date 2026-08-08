# Stage 12D Partial Annotation State

The protocol `1.0.0` input-rejection review contains exactly 12 class/split slots. Three slots are approved from objective adjudication evidence: train `INCOMPLETE_ANATOMY`, validation `INCOMPLETE_ANATOMY`, and validation `INADEQUATE_QUALITY`. The remaining nine slots are explicitly recorded as `INCOMPLETE_NO_DEFENSIBLE_EXAMPLE`.

This is a validated partial annotation state, not a complete cohort and not authorization to train. No annotation was inferred merely to fill a slot, no locked-test record was accessed, and no final input-rejection manifest was promoted.

The next approved step is the evidence search defined in `configs/quality/stage12d_remaining_candidate_discovery_plan.json`. It is limited to existing unlocked train/validation data and must retain an incomplete slot whenever objective evidence is unavailable.
