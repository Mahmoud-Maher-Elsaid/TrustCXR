# TrustCXR Stage 8A Segmentation Readiness Report

## Result

- Status: `PASSED`
- Gate: `GO_FOR_STAGE_8B_UNET_BASELINE`
- Repository commit before Stage 8A: `a2e80aa`
- Selected dataset: `NIH CheXmask`
- Local index: `F:\AI\TrustCXR\artifacts\stage8\chexmask\chexmask_nih_index.sqlite`

## Correct dataset interpretation

CheXmask is distributed as a CSV containing image identifiers, quality scores,
Run-Length Encoded anatomical masks, and mask dimensions. Original images are
matched from the local NIH ChestXray14 dataset.

## Quality and pairing

- Source rows inspected: `112120`
- Minimum Dice RCA Mean: `0.7`
- Final unique records: `110795`
- Train records: `77782`
- Validation records: `15952`
- Test records: `17061`
- Train patients: `21500`
- Validation patients: `4486`
- Test patients: `4715`
- Patient leakage violations: `0`

## Decode validation

- Samples decoded: `192`
- Decode failures: `0`
- Image-mask shape mismatches: `0`
- Mean sample Dice RCA: `0.838501`
- Mean lung-union foreground fraction: `0.236669`
- Mean heart foreground fraction: `0.088414`

## NIH identity resolution

- Unique local NIH image names: `112120`
- NIH patient mappings: `112120`
- Patient metadata source: `F:\AI\TrustCXR\TrustCXR-Data\01_NIH_ChestXray14\archive\Data_Entry_2017.csv`

## Stage 8B baseline

The approved baseline is a three-output-channel U-Net with a ResNet34 encoder.
The target channels are left lung, right lung, and heart. Patient-safe splits
are locked before training, and any threshold selection must use validation
only.
