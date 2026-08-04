# Stage 6: NIH ChestXray14 Multi-label Classification

Stage 6 trains an ImageNet-pretrained DenseNet-121 to predict the 14 NIH
ChestXray14 thoracic findings.

## Training policy

- Maximum epochs: 100
- Batch size: 96
- Input size: 224 x 224
- Classifier-head warm-up: 3 epochs
- Full discriminative fine-tuning after warm-up
- Bounded cyclic training windows
- Adaptive epoch controller targeting 180 to 240 seconds
- Validation Macro AUPRC early stopping with patience 10
- Best-checkpoint restoration
- Full validation threshold calibration
- Full untouched test evaluation
- Zero patient leakage tolerance

A bounded epoch is not a hidden fixed subset. The deterministic cyclic sampler
rotates through the complete patient-safe training split. The report records
total sample exposures and effective full-dataset passes.

## Overfitting controls

- Patient-safe split
- ImageNet initialization
- Dropout
- AdamW weight decay
- Mild medical-safe augmentation
- Label smoothing
- Per-class positive weighting
- Gradient clipping
- Exponential moving average weights
- ReduceLROnPlateau
- Early stopping
- Best model restoration
- Per-label threshold calibration

## Limitations

NIH ChestXray14 labels were produced with automated text mining and include
label noise. The resulting model is a research baseline, not a clinical system.
