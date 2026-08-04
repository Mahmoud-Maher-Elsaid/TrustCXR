# Stage 7B: Frozen RAD-DINO CLS Embedding Extraction

Stage 7B extracts deterministic frozen CLS embeddings from the pinned RAD-DINO checkpoint for every NIH ChestXray14 image in the patient-safe Stage 6 split.

## Model contract

- Model: `microsoft/rad-dino`
- Revision: `110cbc18d5133582e320b43d53bf5c44e410c936`
- Input size: `518 x 518`
- Patch size: `14`
- Hidden size: `768`
- Model dtype: `float16`
- Trainable parameters: `0`
- Fast processor: disabled to preserve the validated preprocessing path

## Data contract

- Train: `77,790` images
- Validation: `8,734` images
- Test: `25,596` images
- Total: `112,120` images
- Patient leakage violations: `0`
- Label order: the same 14-label order used by Stage 6

## Artifact format

Embeddings are stored locally under `artifacts/stage7/rad_dino_cls` and excluded from Git.

Each shard contains:

- `embeddings`: FP16 tensor with shape `[N, 768]`
- `labels`: UINT8 tensor with shape `[N, 14]`
- `record_indices`: INT64 tensor with shape `[N]`
- Paired JSONL metadata with image name, de-identified patient ID, split, and labels
- SHA-256 checksums recorded in the manifest

The extraction is resumable at shard boundaries. Existing shards are skipped only after checksum validation.

## Storage policy

Full patch-token embeddings are not stored. Stage 7A estimated that storing them for all NIH images would require more than 219 GiB. Patch-token analysis remains bounded to future audit or segmentation-specific subsets.

## Scientific disclosure

RAD-DINO pretraining included NIH-CXR images according to its public model documentation. Stage 7 is therefore an in-domain frozen-representation experiment, not independent external validation.

## Completion gate

Stage 7B passes only when:

- all `112,120` records are extracted,
- all split counts match Stage 6,
- every shard checksum is valid,
- every embedding is finite and has hidden size `768`,
- patient leakage remains zero,
- the full project test suite and dependency check pass,
- reports are generated and the implementation is committed to `develop`.
