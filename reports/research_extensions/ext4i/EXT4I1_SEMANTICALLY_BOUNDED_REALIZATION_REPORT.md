# EXT-4I.1 semantically bounded realization

EXT-4I.1 implements deterministic semantic atoms, proposition boundaries,
phrase skeletons, finite lexical choices, and a fail-closed semantic
validator V2. Model-authorized free-text fields are zero; the one lexical
choice field is finite and deterministic in this stage.

All 16 resolved H5 failures are rejected by the historical regression suite,
and all 16 deterministic replacements pass. The deterministic mutation suite
attempts 264 forbidden mutations with zero unexpected accepts. Positive
fixtures cover all six evidence states plus DEFER and contradiction paths.

No model was loaded, no generation occurred, and no unbiased benchmark or
final/locked partition was accessed.
