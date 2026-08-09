# Stage 16 Final Reliability Capability Freeze

Stage 16 is frozen as a research reliability capability under contract fingerprint `3bb34f4edd5d4ed1f27120e028efacf92eb7e0e5217dbc6a1d29c59521b9af9e`.

For `stage9_original`, validation-only calibration, descriptive predictive uncertainty, and selective prediction are accepted. Its validation-selected abstention threshold is frozen at `0.6917239984806322`.

For `stage13_frontal_only`, calibration is only partially accepted because NLL and Brier improved while ECE worsened. Descriptive predictive uncertainty is accepted, but selective prediction is not accepted because it did not reduce reliability-evaluation Brier risk. Its observed validation-selected threshold `0.6928039993196862` is frozen as evidence and must not be promoted to an accepted operational capability.

The global uncertainty claim is `PREDICTIVE_ONLY_NOT_EPISTEMIC`. OOD remains `WITHHELD_NO_GOVERNED_OOD_COHORT`. The two models use different cohorts and label contracts and are not ranked against each other. No additional fitting, tuning, inference, training, calibration, checkpoint modification, or locked-test access occurred during closure.
