# System Architecture

TrustCXR uses a modular architecture.

## Core modules

1. Image quality and view assessment
2. Multi-label classification
3. Anatomical segmentation
4. Lesion segmentation
5. Object detection
6. Evidence fusion
7. Uncertainty estimation
8. Out-of-distribution detection
9. Structured report generation
10. Medical language model integration
11. Report verification
12. Accept, revise, or defer decision support

Each advanced component must have a smaller baseline implementation so that
failure of one model cannot block the full project.