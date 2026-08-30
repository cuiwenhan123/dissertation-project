# Independent Chapter 4 Data Audit

Overall status: **PASS**

## Coverage

- 32 condition files and 36 plotted curve rows
- 500 images per condition (16,000 inference tasks)
- 1,978 ground-truth objects across 80 classes
- 32 condition metrics independently recomputed
- Maximum stored-versus-recomputed metric difference: 0

## Data checks

- Invalid boxes: 0
- Size-label mismatches: 0
- Score-threshold violations: 0
- Prediction-limit violations: 0
- Object-transition accounting errors: 0

## Findings

- No numerical or structural inconsistencies were found.
- WARNING: Predictions are not score-sorted in 6495 image-condition rows; COCOeval sorts internally, so aggregate metrics are unaffected.
- WARNING: 19097 prediction boxes exceed an image boundary; maximum overflow is 29.2952 pixels. COCOeval clips overlap through intersection geometry.
