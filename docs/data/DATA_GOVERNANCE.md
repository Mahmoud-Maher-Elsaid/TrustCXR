# Data Governance

## Local-only content

The following content must remain outside Git version control:

- Medical images
- Radiology reports
- Patient-level records
- Dataset archives
- Credentials
- Access tokens
- Model checkpoints
- Predictions containing patient-level information
- Derived patient-level embeddings
- Temporary preprocessing outputs

## Local dataset root

`<governed-data-root>\TrustCXR-Data`

## Repository-safe content

The repository may contain:

- Source code
- Configuration templates
- Synthetic test data
- Dataset manifests without patient information
- Aggregate statistics
- Documentation
- Reproducible preprocessing instructions
- Unit and integration tests

## Data sharing

Restricted or sensitive medical data must not be uploaded to GitHub or shared
with unauthorized users.
