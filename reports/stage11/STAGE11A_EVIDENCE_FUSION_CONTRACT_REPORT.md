# Stage 11A Evidence Fusion Contract

- Status: `FINALIZED_EVIDENCE_FUSION_CONTRACT`
- Shared fusion cohort required: `true`
- Cross-dataset record-level fusion permitted: `false`
- Training performed: `false`
- Locked test records accessed: `0`

The contract preserves model disagreement and requires unsupported localization to remain `UNLOCALIZED` or `UNCERTAIN`. Absence of localization cannot contradict positive classifier evidence. The Stage 10 anatomical evidence remains limited to image geometry and a thoracic-location proxy.

Stage 11B must validate shared identity, split, and label semantics before any record-level fusion is allowed.
