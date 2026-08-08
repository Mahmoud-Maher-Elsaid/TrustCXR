# Stage 13B Study-Level Identity Resolution

Stage 13B resolves development study identity only by exact record-hash joins to governed CheXpert metadata and explicit `patient.../study.../` source-path segments. Study identifiers are hashed and retained only in an ignored local SQLite index.

Patient identity is never substituted for study identity, and this stage creates no view pairs. It reads no images, excludes locked splits, performs no training or inference, preserves `UNKNOWN`, keeps `OTHER` withheld, and leaves all frozen results unchanged.
