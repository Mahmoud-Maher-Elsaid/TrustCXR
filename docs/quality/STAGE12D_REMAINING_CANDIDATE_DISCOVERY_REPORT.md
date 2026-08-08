# Stage 12D Remaining Candidate Discovery

The approved development-only discovery produced three unapproved CheXpert train candidates for `INCOMPLETE_ANATOMY`. Each local review row records its source, frozen split, local path, stable group identifier, file SHA-256, objective evidence, proposed class, and rationale. The contact sheet is supporting evidence only.

No candidate was approved automatically. Nine class/split slots remain explicitly incomplete because the governed search did not produce defensible genuine examples. The final discovery used only mapped train/validation records, accessed zero locked-test records, preserved patient splits, and performed no training.

The next gate is human adjudication of the three candidates. Extreme frontal geometry is a screening signal, not sufficient by itself: approval requires objective confirmation that required thoracic anatomy is materially cropped or absent.

An earlier superseded bounded RSNA attempt opened 5,000 DICOM headers before split matching. It read no pixels and its outputs were invalidated and quarantined, but some unmatched headers may have belonged to the locked split. The corrected final discovery filters records through the official mapping and frozen split index before file access; it accessed zero locked records. This incident is retained in the aggregate summary rather than hidden.
