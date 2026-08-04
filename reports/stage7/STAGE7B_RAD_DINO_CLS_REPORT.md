# TrustCXR Stage 7B RAD-DINO CLS Extraction

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_7C_PROBE_TRAINING`
- Model: `microsoft/rad-dino`
- Revision: `110cbc18d5133582e320b43d53bf5c44e410c936`
- Frozen encoder: `True`
- Total extracted images: `112120`
- Hidden size: `768`
- Stored embedding dtype: `float16`
- Total local artifact size: `179.65 MiB`
- Elapsed time: `111.92 minutes`
- Mean throughput: `16.70 images/second`
- Patient leakage violations: `0`

## Split coverage

- train: `77790` records across `19` shards
- validation: `8734` records across `3` shards
- test: `25596` records across `7` shards

## Artifact policy

- Full CLS embeddings are stored locally as sharded SafeTensors files.
- Labels and deterministic record indices are stored in each tensor shard.
- Image names, de-identified patient identifiers, split names, and labels are stored in paired JSONL metadata files.
- SHA-256 checksums are recorded for every tensor and metadata shard.
- Full patch-token embeddings are not stored because their estimated size exceeds 219 GiB.
- Local embedding artifacts are excluded from Git.

## Scientific disclosure

The public RAD-DINO model card states that NIH-CXR was included in RAD-DINO pretraining. This stage is therefore an in-domain frozen-representation extraction stage, not independent external validation.

## Next gate

Stage 7C may train deterministic linear and small MLP probes on the frozen CLS embeddings while preserving the same patient-safe train, validation, and untouched test splits.
