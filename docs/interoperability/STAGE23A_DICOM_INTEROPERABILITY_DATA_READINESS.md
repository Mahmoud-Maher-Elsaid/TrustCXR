# Stage 23A DICOM Interoperability Data Readiness

Stage 23A is a code, dependency, privacy, identity, display, and transfer-syntax audit only.
It reads no DICOM object and does not create a viewer, decoder, fixture, upload control, or
patient artifact.

The repository has governed `pydicom==3.0.2` use for selected metadata and identity fields,
plus dataset-specific RSNA pixel decoding for localization workflows. This does not establish
a generic decoder or browser viewer. No generic SOP-class matrix, codec matrix, modality/VOI
display transform, de-identification contract, safe tag-display policy, multi-frame contract,
or compressed-transfer-syntax contract exists. DICOM therefore remains
`WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT`.

Stage 23B is limited to freezing the interoperability and synthetic-fixture protocol. It does
not authorize dependency installation, fixture creation/decoding, real DICOM display, patient
processing, inference, or language-model work.
