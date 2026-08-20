# EXT-4I.1 semantically bounded realization

EXT-4I.1 implements deterministic semantic atoms, proposition boundaries,
phrase skeletons, finite lexical choices, and a fail-closed semantic
validator V2. Model-authorized free-text fields are zero; the one lexical
choice field is finite and deterministic in this stage.

The prior static result was superseded by an authoritative local report of
1045 passed and 3 failed. The repair tightens token-boundary clinical checks,
exempts explicitly authorized topic wording, and adds explicit SUPPORTED
state-substitution rejection. The governed Windows interpreter was not
invocable in this session, so post-repair focused/full pytest and pip-check
results remain pending; final PASS is not claimed.

No model was loaded, no generation occurred, and no unbiased benchmark or
final/locked partition was accessed.

Repair provenance: the pre-repair static implementation commit was
`afb100a52e9db5ee32f0ef808a3288977435ecc5`. The authoritative local gate
reported 1045 passed and 3 failed. The bounded repair corrected token-boundary
clinical checks (including authorized topic exemptions) and added explicit
SUPPORTED state-substitution rejection. H5 and historical regression inputs
were not changed.

The current bounded repair also fixes the proven remaining defects:
`resolved` is matched as a word, so `unresolved conflict` remains valid;
WITHHELD explicitly rejects `normal`; and NOT_APPLICABLE explicitly rejects
`unavailable` and `not available`. Local focused and full-suite validation
remain required before restoring the PASS gate.
