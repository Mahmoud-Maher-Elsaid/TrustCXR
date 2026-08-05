# TrustCXR Stage 8B CheXmask U-Net Baseline

Stage 8B trains a bounded three-channel anatomy segmentation baseline on the
patient-safe NIH CheXmask index prepared in Stage 8A.

## Model

- U-Net decoder
- ImageNet-pretrained ResNet34 encoder
- Three output logits: left lung, right lung, and heart

## Bounded training

Each epoch uses a deterministic bounded subset of the training split. Different
subsets are visited across epochs. A fixed validation subset selects the best
checkpoint and per-organ thresholds. The untouched patient-safe test split is
evaluated once after model selection.

## Scientific limitation

CheXmask targets are quality-filtered pseudo-masks rather than manual clinical
ground truth. Reported Dice and IoU values measure agreement with those targets.
