# Contributing to TrustCXR

TrustCXR is a private research repository. The core release is frozen; `main` represents the stable frozen core and `develop` is retained for integration history. Do not modify frozen scientific conclusions, checkpoints, dataset splits, or accepted metrics through routine contributions.

## Future work

Post-release research must use a dedicated branch, such as `research-extension/explainability-grounded-llm`, and must preserve the frozen core contracts. Grad-CAM, pathology localization, LLM, and VLM work is not part of the core release.

## Data and artifacts

Do not commit datasets, patient data, medical images, real DICOM files, patient reports, credentials, secrets, local virtual environments, or checkpoints unless a future repository policy explicitly authorizes a governed artifact. Safe synthetic fixtures are preferred for tests.

## Evidence and validation

Scientific claims require tracked evidence. Do not casually recompute or alter frozen test metrics. Changes must pass the applicable Ruff, formatting, dependency, stage-coverage, and pytest checks before integration. Presentation changes must not silently modify model behavior or safety limitations.
