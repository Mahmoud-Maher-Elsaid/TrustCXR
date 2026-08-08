# Stage 12G Additional Development Evidence Search

Stage 12G searches a new deterministic tranche of governed CheXpert development images and a disjoint mapped tranche of RSNA development DICOM headers. NIH development metadata is inventoried without treating unavailable rejection labels as negatives.

The search targets only the nine withheld rejection slots. `OTHER` remains unresolved unless positive governed source metadata identifies a chest projection outside AP, PA, and LATERAL. Every candidate remains unapproved pending human adjudication and records its dataset, split, local path, stable group, SHA-256, objective evidence, proposed class, and rationale.

The stage prohibits synthetic data, deliberate corruption, locked-test access, inference, training, and modification of frozen results.
