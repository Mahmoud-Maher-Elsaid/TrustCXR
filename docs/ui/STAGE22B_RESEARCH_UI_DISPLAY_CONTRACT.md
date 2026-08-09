# Stage 22B Research UI Display Contract

Stage 22B freezes a deterministic research UI display contract. It does not create HTML,
CSS, JavaScript, images, screenshots, a browser session, or a running service.

Every relevant screen must visibly state `RESEARCH USE ONLY`, `NOT A MEDICAL DIAGNOSIS`,
and `EXPERT REVIEW REQUIRED`. The frozen contract defines twelve layout regions, the exact
Stage 9 label order, predictive-only uncertainty wording, limited fusion semantics, all six
verifier states, DEFER-first decision wording, sanitized failures, safe provenance fields,
privacy, accessibility, and non-diagnostic visual semantics.

The prospective implementation remains static HTML, CSS, and vanilla JavaScript served by
the existing FastAPI/Starlette stack, with no Node, npm, CDN, cloud asset, analytics, or new
dependency. PNG and JPEG are the only prospective viewer formats. DICOM and all overlays
remain withheld.

Stage 22C may implement and validate this display contract using synthetic/non-patient
fixtures and synthetic PNG/JPEG images only. It may not display real images, load models,
run inference, process patients, access locked tests, or add language-model functionality.
