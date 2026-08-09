# Stage 22C Synthetic Research UI Implementation

Stage 22C implements the frozen Stage 22B display contract with static HTML, CSS, and vanilla
JavaScript served by four fixed FastAPI routes. The existing `/v1` API remains unchanged.
No frontend framework, Node/npm toolchain, CDN, telemetry, external asset, or new dependency
is introduced.

The fixture bundle contains two tiny deterministic non-patient grayscale images: one PNG and
one JPEG, both 32 by 24 pixels. DICOM and Stage 8/10 overlays remain visibly withheld. All
dynamic fixture values use DOM `textContent`; unsafe HTML insertion, browser storage, cookies,
external URLs, arbitrary paths, and script execution are prohibited. A restrictive Content
Security Policy permits only local resources and embedded synthetic image data.

Stage 22C preparation starts no browser or server and displays no real image. Its manual
launcher performs bounded static asset, fixture, route, image-decode, privacy, accessibility,
and JavaScript safety validation only. Stage 22D may perform bounded synthetic UI runtime or
browser validation, but cannot use real images, models, inference, patients, or LLMs.
