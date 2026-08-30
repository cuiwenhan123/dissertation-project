# Frozen Research Evidence

This directory contains the non-image evidence for the dissertation's main
500-image COCO val2017 study. It is included so that the aggregate values in
Chapters 4 and 5 and Appendix C can be checked without downloading model
weights or repeating approximately 5.37 hours of CPU inference.

## Contents

- `chapter4_main_study/conditions/`: 32 compressed prediction archives, one
  for each unique model/degradation/severity condition.
- `chapter4_main_study/results.json`: 36 reported curve rows. The clean row is
  repeated for each degradation family, so these represent 32 unique inference
  conditions.
- `chapter4_main_study/manifest.json`: the fixed 500-image selection, controls,
  model hashes and degradation schedule.
- `chapter4_main_study/analysis/`: aggregate metrics, transition analysis and
  the independent data audit.
- `chapter4_main_study/validation/`: paired bootstrap, leave-block-out and
  end-to-end validation records.
- `chapter4_main_study/CHECKSUMS.sha256`: integrity hashes for every evidence
  file in the directory.

The public manifest replaces the original machine-specific dataset path with
`${ROBUSTNESS_COCO_VAL2017_ROOT}`. Image identifiers, annotations, predictions,
experimental controls and numerical results are unchanged. No COCO image or
pretrained model weight is redistributed.

## Verification

From the repository root, run:

```bash
python scripts/verify_research_evidence.py
```

This checks the file hashes, the 32-condition design, the 500-image pairing,
the 1,978-object transition record, and the principal values reported in
Appendix C.

COCO images and annotations are external research inputs. Refer to the
[COCO website](https://cocodataset.org/) for the dataset terms and citations.
