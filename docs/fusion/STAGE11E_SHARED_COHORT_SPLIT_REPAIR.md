# Stage 11E Shared-Cohort Split Repair

Stage 11E builds a local, identity-bearing train/validation fusion cohort from the official RSNA-to-NIH mapping. It never changes the frozen Stage 9 or Stage 10 splits or evaluations.

The repair excludes every original NIH patient whose mapped train/validation records exhibit any Stage 9 versus Stage 10 split conflict. It does not reassign patients or images. This conservative exclusion resolves the 2,929 observed patient split violations while preserving the historical exposure contract of both component models.

Only `train` and `validation` rows are queried. The locked `test` split is neither queried nor authorized. The resulting SQLite database remains ignored because it contains identity mappings. Tracked output contains aggregate counts only.

The RSNA possible-pneumonia opacity annotation may only partially support NIH `Pneumonia`; it is not semantic equivalence. Localization absence remains `UNLOCALIZED_OR_UNCERTAIN`, not classifier contradiction. Stage 10 localization remains a research baseline with low small-lesion sensitivity and image-geometry/thoracic-location proxy evidence only.

Success opens only a train/validation shared-cohort fusion-validation gate. It does not authorize training, inference, test fusion, clinical localization, or external-validation claims.
