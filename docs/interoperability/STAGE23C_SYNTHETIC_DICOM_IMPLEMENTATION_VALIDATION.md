# Stage 23C Synthetic DICOM Implementation Validation

Stage 23C is prepared to create and decode 22 deterministic, synthetic, non-patient DICOM
fixtures under an ignored request-scoped cache directory. The generic layer accepts bytes only,
validates metadata before PixelData decoding, exposes only allowlisted non-identity metadata,
keeps raw/modality/display representations separate, and cleans all generated DICOM files.

The implemented scope is single-frame grayscale, Explicit or Implicit VR Little Endian, and
MONOCHROME1 or MONOCHROME2. Compressed syntax, multi-frame, real DICOM, patient metadata, UI
rendering, models, inference, GPU profiling, locked tests, and language models remain prohibited.

Stage 23D is the required acceptance and Stage 23 closure decision. It is not an optional
substage and does not authorize runtime expansion.
