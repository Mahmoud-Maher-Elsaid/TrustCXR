# Stage 12D Input-Rejection Candidate Review

This bounded development-only review mines genuine local CheXpert records using source metadata, file resolution and decoding, image geometry, and the existing Stage 5 technical-quality proxy. It never synthesizes corruption, assigns rejection labels, approves candidates, or opens test-assigned images.

Corrupt input requires a genuine missing or undecodable source record. Unsupported format requires a real file outside the ingestion contract. Non-chest input requires trusted source evidence and is not inferred visually. Incomplete-anatomy and inadequate-quality candidates are objective screening candidates requiring human confirmation. Unsupported view requires positive trusted metadata for a known out-of-contract view; ambiguous metadata is insufficient.

Every class and split appears in the review CSV. When no defensible candidate exists, the row explicitly states `NO_DEFENSIBLE_EXAMPLE`. Contact-sheet titles say candidate only and do not represent annotations.
