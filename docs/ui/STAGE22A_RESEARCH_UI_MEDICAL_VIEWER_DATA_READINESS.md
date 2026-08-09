# Stage 22A Research UI and Medical Viewer Data Readiness

Stage 22A is a governance and readiness audit only. It does not implement or start a UI,
server, viewer, worker, model, inference path, or patient workflow.

The smallest defensible prospective UI is lightweight static HTML, CSS, and JavaScript
served locally by the already frozen FastAPI/Starlette stack. This avoids a new framework or
package and consumes only deterministic structured Stage 21 responses. No UI dependency is
required at this gate.

PNG and JPEG have established Pillow decoding paths in the repository. Tensor and NPZ
artifacts remain internal and are not browser display contracts. Although repository code
uses pydicom for governed dataset-specific metadata and RSNA pixel decoding, no generic
DICOM viewer, display transformation, de-identification, or serving contract is frozen;
DICOM viewing is therefore withheld.

Any future overlay must retain its evidence class visibly: Stage 8 is quality-filtered pseudo
lung/heart anatomy, Stage 10 is an image-geometry/thoracic-location proxy, and Stage 11 may
provide at most `PARTIALLY_SUPPORTED` evidence. Reliable positive lesion localization,
laterality inference, and negation from localization absence remain prohibited.
