# Final Validation Report

Validation evidence was consolidated on 21 August 2026 using Python 3.12 and the
local project weights. The automated suite was rerun on 21 August; the real-model
and browser checks below were retained from the completed 14 August validation run.

## Automated Checks

- Python compilation: passed for all backend modules and launchers.
- JavaScript syntax: passed for every module under `backend/static/js`.
- Local automated suite: 30 checks executed; 29 passed and one class-level HTTP check was skipped because socket binding is disabled in the restricted environment.
- Live HTTP boundary tests: configured to run where local socket binding is available, including GitHub Actions.
- COCO80 contiguous YOLO mapping: passed.
- Dataset YAML class override: passed.
- Deterministic motion blur, low-light, and JPEG configuration: passed.
- Official COCO AP perfect/missing/duplicate scenarios: passed.
- SQLite completed-study round trip: passed.
- Completed-study archive acceptance and rejection checks: passed.
- Object-transition archive loading, validation, and selection: passed.
- Full 36-condition study orchestration with a detector adapter: passed.
- Deterministic class-aware subset selection: passed.
- Lazy local-directory image loading and transient decode retry: passed.
- Frozen research evidence: file hashes and the principal Appendix C values passed.

## Real Model Smoke Checks

- DETR backend: `transformers-detr-resnet-50`; inference succeeded; no model errors.
- Faster R-CNN backend: `torchvision-fasterrcnn`; inference succeeded; no model errors.
- Local DETR and Faster R-CNN weights were loaded without network access.

## Real Study Smoke Check

- Dataset: local COCO val2017 directory.
- Imported dataset: 5,000 images, 36,781 annotations, 80 categories, and 4,952 labelled/eligible images.
- Evaluated images: 1.
- Real inference tasks: 32 of 32 completed.
- Stored result conditions: 36.
- Evaluator: `pycocotools COCOeval bbox`.
- Benchmark, Sweep, Curves, report generation, CSV/JSON export state, and SQLite restoration: passed.
- Result metadata recorded the verified import manifest, source path, available/eligible/evaluated counts, deterministic sampling seed, and selected class counts.

The one-image run validates the complete COCO val2017 execution path with both real models; it is not intended as the final dissertation result. Use the predeclared 500-image option for the main quantitative study.

## Browser Checks

- Benchmark restored the latest real SQLite study.
- Three degradation charts and six model polylines rendered.
- Research report showed evidence and enabled Markdown/PDF actions.
- Desktop and 390 px layouts had no horizontal overflow.
- Browser console warnings/errors: none.

## Frozen Evidence Checks

- 32 unique compressed condition archives are present.
- Every condition contains the same ordered set of 500 image identifiers.
- The stored design represents 16,000 image-model evaluations and 36 displayed curve rows.
- The object-transition archive contains 1,978 objects across six model/degradation combinations.
- Clean and severity-5 mAP endpoints match Appendix C at the reported precision.
- `research_evidence/chapter4_main_study/CHECKSUMS.sha256` verifies every public evidence file.

## Run Tests Again

```bash
cd dissertation-project
python -m unittest discover -s tests -v
python scripts/verify_research_evidence.py
```
