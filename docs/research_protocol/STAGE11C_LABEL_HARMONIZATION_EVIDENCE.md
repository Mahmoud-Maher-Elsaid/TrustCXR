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

## Identity status and manual action

The official RSNA-to-NIH mapping is not currently present locally. A shared image or patient identity contract therefore remains unproven. Download the official mapping from the RSNA challenge page and preserve its original filename under:

`F:\AI\TrustCXR\artifacts\stage11\identity\rsna_to_nih_official_mapping\`

This directory is local-only and must remain ignored. Downloading the mapping does not itself authorize fusion. Stage 11D must inspect its real schema, hashes, coverage, duplicates, patient consistency, and split compatibility without opening any locked test data.
