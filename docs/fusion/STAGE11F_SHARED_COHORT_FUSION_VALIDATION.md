# Stage 11F Shared-Cohort Fusion Validation

Stage 11F is a metadata-only integrity and policy validation of the repaired Stage 11 shared train/validation cohort. It checks schema, aggregate counts, unique mapped image identities, allowed splits, and zero patient split violations.

It performs no model training and no inference. It does not query or authorize locked-test records. It cannot change frozen Stage 9 or Stage 10 evaluations or reassign patients.

Successful validation permits only later preparation of train/validation record-level fusion. RSNA possible-pneumonia opacity remains partial support for NIH `Pneumonia`, never semantic equivalence. All Stage 10 and Stage 11A downstream evidence limitations remain mandatory.

## Environment prerequisite

The repository venv was created from Python 3.12.10 at `C:\Users\maher\AppData\Local\Programs\Python\Python312\python.exe`. That base executable is currently unavailable, so `.venv\Scripts\python.exe` cannot launch. Reinstall Python 3.12.10 at the recorded location or recreate `.venv` from a verified Python 3.12.10 installation and the repository lock files before relying on the full test suite or running Stage 11F.
