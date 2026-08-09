# Stage 18 Full-Suite Fixture Audit

Five failures were stale synthetic fixtures. Stage 18C prospectively hardened the frozen
deterministic renderer to require numeric model scores, governed source-stage provenance, and
evidence-type-specific evidence codes. Two Stage 18B assertions still supplied the earlier
string score, while three Stage 18C assertions were blocked by generic synthetic provenance
before reaching their intended checks. Those fixtures now use the already-frozen validation
rules.

The sixth failure exposed validation-order drift: strict schema rejection occurred before the
more specific patient-identifier rejection. Both paths failed closed, but the safety test no
longer verified the explicit privacy control. Patient-field detection now precedes generic
schema rejection, restoring the stronger deterministic privacy classification. No contract,
schema, template, fingerprint, acceptance decision, or permitted capability changed. No test
was weakened, skipped, or relaxed.
