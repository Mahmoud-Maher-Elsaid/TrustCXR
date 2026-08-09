# Stage 16B Reliability Contract

Stage 16B prospectively freezes calibration, predictive-uncertainty, and selective-prediction methods before fitting or threshold selection. Patients are deterministically partitioned within validation into disjoint calibration-fit, abstention-selection, and reliability-evaluation subsets.

Calibration uses one scalar temperature per frozen model, fitted by masked binary cross-entropy. Evaluation reports masked NLL, Brier score, and fixed 15-bin ECE per label and macro. Predictive uncertainty is limited to Bernoulli entropy and distance-from-half confidence; no epistemic claim is permitted without an independently valid method.

Selective prediction uses maximum label entropy per record, prespecified coverage levels, masked Brier risk, discrimination metrics, and patient-cluster confidence intervals. Any later abstention threshold is selected only on the dedicated validation selection subset. Locked test data cannot fit calibration or select thresholds.

OOD remains `WITHHELD_NO_GOVERNED_OOD_COHORT`. No external dataset or synthetic degradation is treated as OOD without a separate governed protocol.
