# Stage 22D Bounded Synthetic UI Runtime/Browser Validation

Stage 22D is prepared to exercise the four real UI routes through in-process ASGI and, when
available, an already-installed Chromium-family headless browser. Microsoft Edge and Google
Chrome executables were found locally; no Playwright, Selenium, Pyppeteer, browser, Node/npm,
or other dependency is added.

The manual launcher may start one loopback-only Uvicorn thread on port 42122 for bounded
headless DOM validation. It repeats the same render, checks required and prohibited wording,
renders the two frozen synthetic data images separately, then terminates the server and
deletes the ignored browser profile and screenshots. Browser networking is disabled and the
page CSP permits no external resources.

Only the synthetic non-patient PNG and JPEG fixtures are permitted. DICOM, Stage 8/10
overlays, real images, models, inference, GPU profiling, patient data, locked tests, and LLM
work remain prohibited. Stage 22E is a synthetic UI acceptance decision only and authorizes
none of those capabilities.
