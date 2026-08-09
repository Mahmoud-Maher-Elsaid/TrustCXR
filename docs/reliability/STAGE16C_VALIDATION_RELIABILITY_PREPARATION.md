# Stage 16C Validation Reliability Preparation

Stage 16C reuses the hash-pinned Stage 9 validation prediction archive and performs only the minimum frozen Stage 13 frontal-only validation inference. It writes ignored local artifacts containing probabilities, targets, masks, deterministic partition codes, and salted hashes rather than patient identifiers.

Validation patients are assigned with salt `trustcxr-stage16-reliability-v1` to calibration-fit `[0.00,0.50)`, abstention-selection `[0.50,0.75)`, or reliability-evaluation `[0.75,1.00)`. No reliability metrics, calibration parameters, or abstention thresholds are produced. OOD remains withheld.
