# TrustCXR dataset recovery manifest

Manifest: `TRUSTCXR_DATASET_RECOVERY_MANIFEST_V1`  
Project status: `CLOSED`  
Branch/HEAD: `main` / `199dfce15de1472206d3eacad4300d589ecf84ae`  
Retention mode: `ARCHIVED_RESEARCH_WITHOUT_LOCAL_DATASETS`

This manifest records the ten local dataset directories before planned
deletion. It does not recreate data, recompute results, open final/locked
cases, or guarantee that any source can be recovered. Where exact source,
version, license, checksum, or access procedure was not frozen in repository
evidence, recovery is explicitly marked `UNKNOWN_REQUIRES_MANUAL_RECOVERY`.

## Planned deletion decision

- Target: `F:\AI\TrustCXR\TrustCXR-Data`
- Current measured size: `443,504,177,084 bytes` / `413.045 GiB`
- Deletion occurred: **NO**
- User selected disk-space recovery over immediate local retraining and
  dataset-level reproduction.
- Scientific results are not recomputed or altered.
- Final cases accessed: `0`.
- Locked test accessed: `false`.

## Dataset inventory

| ID / local path | Size GiB | Role and status | Training | Evaluation | Recovery | Disposition |
|---|---:|---|---|---|---|---|
| `nih_chestxray14` / `01_NIH_ChestXray14` | 41.981 | Stage 6 classifier and frozen-core NIH evidence; CORE | YES | YES | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `vinbigdata` / `02_VinBigData` | 333.356 | Localization candidate; WITHHELD | NO | NO | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `indiana_reports` / `03_Indiana_Reports` | 13.211 | Report candidate; WITHHELD | NO | NO | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `nih_chexmask` / `04_NIH_CheXmask` | 2.042 | Pseudo-anatomy evidence; CORE-RELATED | YES | YES | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `crd_masks` / `05_CRD_Masks` | 2.375 | Anatomy-mask candidate; WITHHELD | NO | NO | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `rsna_pneumonia` / `06_RSNA_Pneumonia` | 3.685 | RSNA Lung Opacity bounding-box baseline | YES | YES | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `chexpert_small` / `07_CheXpert_Small` | 10.684 | Stage 5 view/quality and Stage 13 multiview evidence | YES | YES | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `siim_pneumothorax` / `08_SIIM_Pneumothorax` | 0.409 | Single-finding localization candidate; WITHHELD | NO | NO | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `tbx11k` / `09_TBX11K` | 3.791 | External tuberculosis candidate; WITHHELD | NO | NO | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |
| `covid_radiography` / `10_COVID_Radiography` | 1.511 | External COVID candidate; WITHHELD | NO | NO | UNKNOWN_REQUIRES_MANUAL_RECOVERY | SAFE_DELETE_IF_RAW_DATA_NOT_NEEDED |

All ten directories are untracked. Ignored status is not treated as proof
that deletion is scientifically harmless. The conditional disposition means
that deletion is acceptable only because the project is closed and the user
has accepted loss of immediate local dataset reruns. It does not mean that
the datasets are verified recoverable.

## Dataset-specific provenance and recovery notes

The authoritative repository evidence is `docs/release/FINAL_DATASET_USE_SUMMARY.md`,
`configs/data/dataset_catalog.json`,
`configs/data/concrete_adapter_registry.json`,
`configs/data/final_training_selection.json`, and
`reports/project_audit/data_inventory.csv`, supplemented by the stage reports
listed in the JSON manifest.

No exact source URL is frozen for NIH ChestXray14, VinBigData, Indiana,
CheXmask, CRD Masks, RSNA, CheXpert, SIIM, or TBX11K. Their catalog license
state is `REVIEW_REQUIRED`; exact versions and content checksums are not
recorded. The COVID source note does list multiple upstream URLs, which are
copied verbatim into the JSON manifest and the small metadata preservation
file, but it does not establish one exact aggregate version or license.

Important local metadata includes NIH split/label files, VinBigData and RSNA
label CSVs, Indiana projection/report CSVs, CheXpert train/valid CSVs, SIIM
train/submission CSVs, TBX11K `data.csv`, and COVID metadata spreadsheets.
These are documented as filenames only; no patient-level content was copied.

## What deletion preserves and removes

Deleting `TrustCXR-Data` does **not** delete:

- Git history or the synchronized GitHub repository;
- source code, configs, tests, paper, README, roadmap, and committed reports;
- scientific artifacts retained outside the dataset root;
- final closure evidence;
- `F:\AI\TrustCXR_FINAL_MERGED_MAIN_BACKUP.bundle`.

It **does** remove:

- the current local copies of all ten datasets;
- immediate local retraining;
- immediate dataset-level reproduction of published/project numbers;
- potentially difficult-to-recover source data where identity, version,
  licensing, or access details are incomplete.

This limitation is intentional and accepted for archival disk recovery; it is
not evidence that the datasets were reproducibly archived elsewhere.

## Small metadata preservation

One non-sensitive, genuinely useful recovery note was preserved:

`reports/final/dataset_recovery_metadata/COVID_Radiography_README.md.txt`

Source:
`TrustCXR-Data/10_COVID_Radiography/archive (1)/COVID-19_Radiography_Dataset/README.md.txt`

It records the aggregated dataset description, citations, and upstream URLs.
No images, patient data, PHI, large labels, or raw archives were copied. The
NIH and CRD PDF README files were observed but not copied because their source
details were not needed to establish the manifest and could not be safely
parsed in this environment.

## Recovery conclusion

The manifest is sufficient to explain the scientific role and local layout of
the deleted datasets, but not sufficient to promise exact re-download. The
correct recovery state for all ten datasets is therefore
`UNKNOWN_REQUIRES_MANUAL_RECOVERY`.
