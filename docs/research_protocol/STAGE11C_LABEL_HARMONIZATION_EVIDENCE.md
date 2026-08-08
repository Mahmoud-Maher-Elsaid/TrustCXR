# Stage 11C Label Harmonization Evidence

## Adjudicated relation

NIH `Pneumonia` and RSNA `Lung Opacity` are not equivalent labels. The approved relation is:

`RSNA_POSSIBLE_PNEUMONIA_OPACITY_MAY_PARTIALLY_SUPPORT_NIH_PNEUMONIA`

This relation permits only `PARTIALLY_SUPPORTED` evidence after image identity and split compatibility are independently proven. It cannot confirm pneumonia, and absence of an RSNA localization cannot contradict the classifier.

## Evidence

The RSNA challenge page states that the challenge localized pneumonia cases derived from NIH images and provides an official mapping from the RSNA image dataset to the original NIH dataset:

https://www.rsna.org/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

The peer-reviewed dataset paper explains that the bounding boxes represent pulmonary opacity that may indicate pneumonia in the appropriate clinical setting. It also explains that imaging findings such as consolidation and infiltration can have causes other than pneumonia and require clinical correlation:

George Shih et al., *Augmenting the National Institutes of Health Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia*, Radiology: Artificial Intelligence, DOI `10.1148/ryai.2019180041`.

## Identity status

The official mapping was downloaded from the RSNA challenge page and preserved under:

`F:\AI\TrustCXR\artifacts\stage11\identity\rsna_to_nih_official_mapping\`

The preserved filename is `pneumonia-challenge-dataset-mappings_2018.json`, and its SHA-256 is `803ce79e3bc9c66d3631738e91e62e1175730e98ad1415e8dc4d6292ba10bf27`. Stage 11C verified 30,000 complete and unique image-identity rows. The directory is local-only and remains ignored.

The mapping proves RSNA-to-original-NIH image identity. It does not by itself prove patient grouping or compatibility between project splits. Stage 11D must audit those relations using train and validation metadata only; locked test rows remain inaccessible.
