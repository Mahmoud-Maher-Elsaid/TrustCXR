# TrustCXR Stage 8 Final Segmentation Model Card

## Model

- Architecture: U-Net
- Encoder: ResNet34
- Input: 256 x 256 three-channel chest radiograph
- Outputs: left lung, right lung, and heart masks
- Selected checkpoint: Stage 8C coverage continuation

## Final test performance

- Macro Dice: `0.971378`
- Macro IoU: `0.944425`

## Intended use

Research use inside the TrustCXR pipeline for anatomy-aware image processing, quality checks, and downstream experimental integration.

## Limitations

The targets are CheXmask pseudo-masks and are not manual clinical annotations. The model is not validated for independent clinical deployment, diagnosis, treatment decisions, or autonomous patient care.

## Reproducibility

- Checkpoint SHA256: `f97e6f94ea0ad5f33c98bdeffe3bf19f3d168c0bd41317e083fa08361cba00ba`
- Thresholds SHA256: `52dab314c34d2f079203a13eceb54bfb5fe3c8a88e2faea878746a76b49c0b42`
- Evaluation config SHA256: `009e4cd6ba8c13e4dab493c45838f23e30e66eabedb0d9597a728b6deaa2b825`
