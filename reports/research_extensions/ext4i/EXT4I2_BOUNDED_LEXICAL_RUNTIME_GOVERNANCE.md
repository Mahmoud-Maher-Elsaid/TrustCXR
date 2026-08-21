# EXT-4I.2 bounded lexical runtime governance

The only remaining lexical field is `connector`, with the finite enum
`and | while`. Both values are semantically equivalent, and the existing
deterministic selector is sufficient. The readability benefit is marginal;
introducing a GPU LLM would add runtime variance, cost, and audit surface
without a demonstrated scientific benefit.

Decision: `EXT4I2_DECISION_FULLY_DETERMINISTIC_REALIZATION` and
`NO_LLM_NEEDED_FOR_REALIZATION`. The LLM remains available for separately
governed research tasks, but it has no role in the deterministic reporting
path. The smoke specification is preserved as a deterministic synthetic
acceptance case only; no model execution is authorized.
