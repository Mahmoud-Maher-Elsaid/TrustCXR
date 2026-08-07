# Stage 10 Lesion Localization Completion Report

- Status: `FINALIZED_LOCALIZATION_ACCEPTANCE_DECISION`
- Decision: accept the Stage 10E model as a research baseline with mandatory limitations
- Operating threshold frozen: `false`
- Final-test evaluation authorized: `false`
- Clinical localization claim authorized: `false`
- Anatomical conclusion: image-geometry and thoracic-location proxy only
- Final test images accessed during Stages 10A–10N: `0`

## Mandatory limitations

- Small-lesion sensitivity was `0.036145` at reference score 0.5.
- No audited operating point satisfied the frozen sensitivity and false-positive criteria.
- The Stage 10J repair failed and must not replace the Stage 10E baseline.
- No matched anatomy masks support a lung-region localization claim.
- Evidence is internal validation only and is not clinical validation.

Downstream fusion must mark unsupported localization as `UNLOCALIZED` or `UNCERTAIN`. Because sensitivity is poor, absence of localization must not contradict positive classifier evidence.
