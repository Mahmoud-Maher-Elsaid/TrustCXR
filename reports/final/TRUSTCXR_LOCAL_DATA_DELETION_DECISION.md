# TrustCXR local dataset deletion decision

Project status: `CLOSED`

Data retention mode: `ARCHIVED_RESEARCH_WITHOUT_LOCAL_DATASETS`

The user deliberately selected disk-space recovery over immediate local
retraining and dataset-level rerun capability. The planned deletion target is:

`F:\AI\TrustCXR\TrustCXR-Data`

Current measured size: `443,504,177,084 bytes` / `413.045 GiB`.

Deletion has **NOT** occurred at the time of this record.

No scientific result is being recomputed or altered. No model is loaded, no
inference or training is run, and no final or locked case content is accessed.

The recovery manifest records all ten local dataset directories, their roles,
known metadata, and explicit recovery uncertainty. All ten are classified
`SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED`, not verified recoverable. Deleting them
will preserve Git/GitHub source, committed reports, paper, closure evidence,
and the final merged Git bundle, but will remove immediate local retraining
and dataset-level reproduction capability.

One small non-sensitive COVID source note was preserved at
`reports/final/dataset_recovery_metadata/COVID_Radiography_README.md.txt`.
No raw images, PHI, patient identifiers, large archives, or label datasets
were copied.

This decision record authorizes documentation of a later deletion plan only;
it does not execute deletion.
