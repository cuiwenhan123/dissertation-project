# Research Protocol

## Research Question

How do common image degradations affect a transformer detector relative to a convolutional two-stage detector, and which object scales and failure modes are most vulnerable?

## Hypotheses

- H1: detection performance decreases monotonically as degradation severity increases.
- H2: DETR and Faster R-CNN exhibit different sensitivity profiles across motion blur, low illumination, and JPEG artefacts.
- H3: small-object AP and recall decline more rapidly than large-object performance.

## Experimental Design

- Fixed models: COCO-pretrained DETR ResNet-50 and Faster R-CNN ResNet-50 FPN.
- Dataset: COCO val2017 for the main held-out evaluation; COCO128 remains bundled for pilot evaluation. User-supplied COCO JSON and YOLO TXT datasets are also supported.
- Independent variables: model, degradation family, and severity level 0-5.
- Controlled variables: input dataset, labels, seed, score threshold 0.05, maximum 100 detections, model weights, and package versions.
- Conditions: 2 models x 3 degradations x 6 severity levels = 36 result rows.
- Clean inference at severity 0 is executed once per model and reused as the baseline for each degradation curve.
- Large-dataset selection: deterministic seeded class-aware sampling, with image IDs and selected class counts preserved in the result metadata.
- Study sizes: 128 images for the pilot, 500 for the primary dissertation study, 1,000 for an extended study, and up to all 4,952 labelled val2017 images for a full run.

## Degradation Parameters

| Severity | Motion blur kernel | Low-light exposure / gamma / noise sigma | JPEG quality |
|---:|---:|---:|---:|
| 0 | identity | identity | identity |
| 1 | 3 px, 12 deg | 0.78 / 1.20 / 2 | 85 |
| 2 | 7 px, 12 deg | 0.65 / 1.35 / 4 | 70 |
| 3 | 11 px, 12 deg | 0.52 / 1.50 / 6 | 50 |
| 4 | 15 px, 12 deg | 0.40 / 1.70 / 8 | 30 |
| 5 | 21 px, 12 deg | 0.30 / 1.90 / 10 | 15 |

Low-light noise is deterministic for a fixed study seed and image name.

## Measures

- Primary: COCO bbox mAP@[IoU .50:.95].
- Secondary: AP50, AP75, AR100, AP and AR for small/medium/large objects.
- Robustness: clean-to-corrupted absolute drop, clean retention, and normalized trapezoidal area under each severity curve.
- Failure modes: missed detection, false positive, classification error, and localisation error using one-to-one matching.

## Reproducibility Record

Every completed study stores:

- dataset name and SHA-256;
- both model SHA-256 values;
- Python and package versions;
- seed, score threshold, and maximum detections;
- annotation format and class-mapping source;
- source path, available/eligible/evaluated image counts, sampling method, sampling seed, and selected class counts;
- exact degradation parameters and condition-level runtime;
- all condition metrics and per-class results.

Records are saved in `runs/experiments.sqlite3` and can be exported from the Benchmark page.

## Validity Boundaries

- The synthetic Overview scenes demonstrate interaction and model loading; they are not used for dissertation claims.
- COCO128 supports framework validation and pilot analysis but has limited statistical power.
- The primary reported result should use a predeclared 500-image COCO val2017 subset. A 1,000-image sensitivity run is preferred when compute time permits.
- The same seed and image subset must be used for both models and every degradation condition; the test set must not be used for model selection or fine-tuning.
- Models remain frozen for the primary comparison. Any corruption-aware fine-tuning must be reported as a separate experiment with disjoint training and test data.
