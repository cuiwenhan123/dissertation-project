# Supervisor Demonstration Guide

## Before the Meeting

1. Start `pycharm_run.py` in PyCharm using the project virtual environment.
2. Confirm the terminal shows `http://127.0.0.1:8877/` and no model errors.
3. Open that address in Safari.
4. Keep the completed 500-image COCO val2017 study saved in SQLite. Use a 1-image study only when demonstrating a new run live.

## Eight-Minute Walkthrough

1. **Overview**: explain that both local pretrained models execute real inference; run one JPEG severity-1 example.
2. **Dataset**: show the local COCO val2017 source, its 5,000 images, 4,952 eligible labelled images, and standard COCO80 class mapping.
3. **Benchmark**: show the 36-condition matrix, exact seed, progress control, mAP, worst degradation, and most robust model.
4. **Curves**: compare the six severity points for motion blur, low illumination, and JPEG artefacts; discuss robustness AUC and clean-to-severity-5 drop.
5. **Class Analysis**: identify classes or object scales with the weakest AP/recall.
6. **Failure Cases**: show missed, false-positive, classification, and localisation evidence.
7. **Report**: show the evidence-backed interpretation and export options.
8. **Methodology**: finish with fixed weights, official COCOeval, hashes, seed, and SQLite reproducibility.

## Short Explanation of the Contribution

"This project contributes a reproducible evaluation framework for comparing transformer and CNN object detectors under controlled image corruption. It combines real inference, official COCO metrics, size-aware and class-aware analysis, failure inspection, robustness curves, and exportable experiment records."

## Expected Questions

**Why no fine-tuning?**  
The primary research question measures inherent robustness of fixed pretrained detectors. Fine-tuning would introduce an additional training variable. Corruption-aware fine-tuning is reserved as a separate extension.

**Why COCO val2017?**  
It is a standard held-out object-detection benchmark with 80 categories and enough labelled images for stronger statistical evidence. COCO128 is retained only as a portable pilot dataset.

**Why use a 500-image subset?**  
It provides substantially broader class and scene coverage than the pilot while remaining practical for a 32-inference-task-per-image study on a laptop. Selection is deterministic and class-aware, so the exact subset is reproducible.

**Is the mAP standard?**  
Yes. The backend uses pycocotools COCOeval for bbox AP@[.50:.95], AP50, AP75, AR100, and object-size metrics.

**How is the experiment reproducible?**  
The system stores dataset and model hashes, package versions, random seed, thresholds, degradation parameters, condition runtime, and metrics in SQLite and JSON exports.
