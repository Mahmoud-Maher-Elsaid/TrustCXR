# Stage 11G Deterministic Fusion Implementation

Stage 11G implements and validates the frozen evidence-fusion rule matrix without loading either model or consuming predictions. It verifies agreement, unreliable localization, model disagreement, invalid geometry, and unsupported-label behavior.

The maximum positive evidence status is `PARTIALLY_SUPPORTED`. Absence of reliable localization cannot negate classifier evidence. Disagreement remains explicit. The implementation is restricted to the repaired patient-safe train/validation contract and cannot access or authorize locked-test data.

This stage performs no training, inference, threshold selection, or clinical validation. Its success opens only preparation for a later validation-only record-level fusion evaluation.
