# Detection Robustness Workbench

Detection Robustness Workbench is a local research application for evaluating object detectors under controlled image degradation. It compares COCO-pretrained DETR ResNet-50 and Faster R-CNN ResNet-50 FPN on the same labelled images, records the complete experimental configuration, and preserves the evidence needed to reconstruct reported metrics.

The project supports interactive inspection, uploaded COCO/YOLO datasets, background benchmark studies, COCO metrics, failure analysis, robustness curves, object-transition analysis, persistent study records, and CSV/JSON exports. Real-model inference is the default. The deterministic fallback is restricted to interface demonstration and must not be used as research evidence.

## Research Scope

- **Models:** DETR ResNet-50 and Faster R-CNN ResNet-50 FPN
- **Degradations:** directional motion blur, seeded low illumination, and JPEG compression
- **Severity:** clean input plus five ordered degradation levels
- **Metrics:** COCO AP@[.50:.95], AP50, AP75, AR100, and size-specific AP/AR
- **Diagnostics:** missed detections, false positives, classification errors, localisation errors, and object transitions
- **Reproducibility:** dataset and model hashes, package versions, seed, thresholds, degradation parameters, and persisted study records

## Architecture

```text
.
├── backend/
│   ├── application.py     application factory, logging, and server lifecycle
│   ├── routes.py          HTTP API and restricted static-file delivery
│   ├── settings.py        typed environment configuration
│   ├── models.py          DETR and Faster R-CNN adapters
│   ├── studies.py         background benchmark orchestration
│   ├── metrics.py         COCO evaluation and failure analysis
│   ├── annotations.py     COCO and YOLO annotation parsing
│   ├── images.py          deterministic image degradation
│   ├── study_storage.py   SQLite persistence
│   ├── transitions.py     object-level transition analysis
│   ├── research_data/     packaged frozen analysis summaries
│   └── static/
│       ├── views/         one HTML fragment per workbench page
│       ├── js/core/       API, file, formatting, and download services
│       ├── js/features/   self-contained research feature modules
│       ├── js/app.js      view loader and frontend bootstrap
│       └── css/           base, component, research, and responsive layers
├── research_evidence/     frozen predictions, metrics, validation, and hashes
├── scripts/               reproducibility and evidence-verification commands
├── tests/                 scientific-core and HTTP boundary tests
├── server.py              compatibility entry point
├── pyproject.toml         package metadata and tooling configuration
└── .env.example           runtime configuration template
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for component boundaries and the evidence flow.

## Requirements

- Python 3.12
- macOS, Linux, or Windows with sufficient memory for CPU inference
- Local COCO-pretrained model weights, or network access for first-time acquisition
- A labelled COCO JSON or YOLO TXT dataset for benchmark studies

The recorded dissertation experiment used the exact Torch, Torchvision, and Transformers versions pinned in `requirements.txt`.

## Installation

```bash
git clone https://github.com/cuiwenhan123/dissertation-project.git
cd dissertation-project

python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

## Configuration

Copy the environment template and edit the dataset path:

```bash
cp .env.example .env
```

Expected local weight layout:

```text
models/
├── detr-resnet-50/
│   ├── config.json
│   ├── preprocessor_config.json
│   └── model.safetensors
└── fasterrcnn-resnet50-fpn-coco.pth
```

Expected COCO val2017 layout:

```text
coco_val2017/
├── images/val2017/*.jpg
├── annotations/instances_val2017.json
└── dataset_manifest.json              # optional, recommended
```

Set `ROBUSTNESS_COCO_VAL2017_ROOT` to the absolute path of `coco_val2017`. Datasets and model weights are intentionally excluded from Git because of size and licensing constraints.

Weights may remain outside the repository. Set `ROBUSTNESS_DETR_MODEL_PATH`, `ROBUSTNESS_FASTER_RCNN_MODEL_PATH`, and `ROBUSTNESS_BUILTIN_DATASET_PATH` to their absolute locations instead of copying them into `models/` and `datasets/`.

To allow first-time model downloads instead of local-only loading:

```bash
export ROBUSTNESS_ALLOW_MODEL_DOWNLOAD=1
export TRANSFORMERS_OFFLINE=0
export HF_HUB_OFFLINE=0
```

Do not disable TLS verification when downloading model files.

## Running the Application

The launcher reads `.env` when present:

```bash
./run_local.sh
```

Alternatively:

```bash
python -m backend
# or
python server.py
```

Open [http://127.0.0.1:8877/](http://127.0.0.1:8877/). Change the host or port with `ROBUSTNESS_HOST` and `ROBUSTNESS_PORT`.

### PyCharm

1. Open the repository root as the project.
2. Select the Python 3.12 virtual environment.
3. Run `pycharm_run.py` with the repository root as the working directory.

## Tests

```bash
python -m compileall -q backend server.py pycharm_run.py sync_previous_study.py
python -m unittest discover -s tests -v
```

The test suite covers annotation mapping, deterministic degradation, COCO metric edge cases, class-aware selection, lazy dataset loading, study persistence and import, object transitions, API status, and static-file access control. GitHub Actions runs the same checks on every push and pull request.

To verify the public evidence against the main values in Appendix C:

```bash
python scripts/verify_research_evidence.py
```

## Benchmark Workflow

1. Confirm `/api/status` reports both local models and the intended dataset as available.
2. Inspect the dataset on the **Dataset** page.
3. Use **Overview** only for a quick interface and model-loading check.
4. Start the labelled study from **Benchmark** with a fixed image count and seed.
5. Keep the server running until every condition is complete.
6. Inspect **Curves**, **Class Analysis**, and **Failure Cases**.
7. Export CSV/JSON evidence and retain the generated study identifier.

Recommended scales are 1 image for an end-to-end smoke test, 16 for development, 128 for a pilot, and 500 for the main study. A full 5,000-image request evaluates all eligible labelled files and may require an overnight run on CPU hardware.

## API Summary

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/api/status` | Runtime, model, package, and dataset availability |
| `GET` | `/api/evaluate` | Single-scene evaluation |
| `GET` | `/api/compare` | Compare both detector adapters |
| `GET` | `/api/study/status?id=...` | Background-study progress |
| `GET` | `/api/study/latest` | Latest completed real study |
| `GET` | `/api/study/history` | Persisted study history |
| `POST` | `/api/study/start` | Start a labelled benchmark study |
| `POST` | `/api/study/cancel` | Request study cancellation |
| `POST` | `/api/inspect-zip` | Inspect an uploaded labelled archive |
| `POST` | `/api/upload-zip-evaluate` | Evaluate an uploaded labelled archive |

JSON request bodies are size-limited by `ROBUSTNESS_MAX_REQUEST_BYTES`. Static delivery is confined to packaged HTML, JavaScript, and CSS under `backend/static`; path traversal, runtime databases, source files, datasets, and model files are rejected by the web server. Automated contract tests verify that every frontend API reference exists in the backend route registry.

## Reproducibility Boundary

Research conclusions must come from a completed labelled real-model study. Synthetic scenes and fallback predictions are demonstrations only. The workbench does not fine-tune either detector and does not claim that the tested corruption functions reproduce every deployment condition. Results apply to the named model weights, preprocessing pipelines, dataset subset, and recorded configuration.

The repository includes the 32 compressed prediction archives used by the main study, the fixed 500-image manifest, aggregate condition metrics, object-transition records, paired bootstrap outputs, leave-block-out checks, and SHA-256 checksums under `research_evidence/chapter4_main_study/`. The workbench-ready archive remains under `backend/research_data/`. Run `python sync_previous_study.py` to reconstruct `runs/experiments.sqlite3`; the generated local database is intentionally not tracked.

## Repository Policy

Commit source code, tests, documentation, prediction evidence without source images, frozen analysis outputs, and acquisition instructions. Do not commit model weights, raw datasets, local databases, uploads, virtual environments, or secrets. Before publishing, review the licences for every external dataset, model, image, and copied figure.
