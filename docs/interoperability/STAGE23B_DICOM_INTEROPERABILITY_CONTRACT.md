# Stage 23B DICOM Interoperability Contract

Stage 23B freezes a prospective synthetic-only interoperability contract. The initial scope is
single-frame grayscale with Explicit or Implicit VR Little Endian and MONOCHROME1 or
MONOCHROME2. CR, DX, Secondary Capture, multi-frame, compressed transfer syntaxes, other
photometric interpretations, VOI LUT selection, generic viewing, and real data remain withheld.

Raw stored, modality-transformed, display-windowed, and normalized ML representations remain
distinct. MONOCHROME1 inversion applies only to display-windowed values; it never mutates raw
stored values. ML tensors are never viewer representations. Missing mandatory metadata,
invalid combinations, unsafe metadata, and unsupported capability fail closed without guessing.

Stage 23B creates and decodes no DICOM. If the contract passes, Stage 23C may create and decode
bounded deterministic synthetic fixtures only. It does not authorize UI rendering, real DICOM,
patient processing, model inference, optional codecs, or language-model work.
