# Stage 12D Remaining Candidate Discovery Plan

Two validation slots are evidence-approved: one strongest incomplete-anatomy example and the sole inadequate-quality example. Ten slots remain explicitly incomplete; the final manifest must not be promoted.

The next search is limited to existing governed CheXpert, NIH ChestXray14, and RSNA train/validation partitions. It begins with metadata and format inventories, then integrity checks, and decodes only development candidates with relevant evidence. It preserves each dataset's frozen patient split.

Genuine corruption requires an actual integrity, decode, or malformed-pixel failure. Unsupported format requires a real medical-image input outside the ingestion contract, not an archive or document. Non-chest requires trusted provenance. Incomplete anatomy excludes valid lateral views and requires material anatomical absence. Inadequate quality requires valid decoding and objective unusable quality. Unsupported view requires positive trusted metadata for a known out-of-contract view.

If existing unlocked data supplies no defensible example, the corresponding slot remains incomplete. No synthesis, deliberate corruption, visual-only labeling, test access, or training is permitted.
