# Architecture

## Design Goals

The workbench separates experimental evidence from interface state. A browser request declares a configuration, the backend validates it, model adapters produce a shared prediction representation, and the evaluation layer calculates metrics without depending on the frontend.

## Component Boundaries

```text
Browser
  -> modular page views and feature controllers
  -> HTTP routes
  -> study orchestration
  -> dataset and annotation loaders
  -> deterministic degradation
  -> detector adapters
  -> COCO evaluation and failure analysis
  -> SQLite and JSON evidence
  -> API views and exports
```

### Application and HTTP Layer

`backend.application` owns logging and the `ThreadingHTTPServer` lifecycle. `backend.routes` parses HTTP inputs and translates them into calls to domain services. It does not perform model inference directly. Static access is allow-listed so a request cannot retrieve source files, weights, datasets, or the study database.

### Frontend Modules

`backend/static/index.html` contains only the application shell and navigation. `js/views.js` loads one HTML fragment for each page before `js/workbench.js` binds controls. Shared request, file, formatting and download behaviour lives under `js/core`; research-specific controllers live under `js/features`. CSS is divided into base, reusable component, research-page and responsive layers. This keeps page markup and feature logic independently reviewable without adding a Node build requirement.

Frontend route strings are checked against the backend GET and POST route registries in the automated suite. The same tests verify that every navigation target has one view fragment, every DOM identifier is unique across fragments, and every entry-point asset is package-accessible.

### Configuration

`backend.settings.Settings` is the typed boundary for environment configuration. `backend.config` derives stable project paths and experiment constants from one settings instance. Legacy `PROTOTYPE_*` variables remain readable for compatibility, but new deployments should use `ROBUSTNESS_*` variables.

### Model Adapters

`backend.models` keeps DETR-specific and Torchvision-specific preprocessing inside separate adapters. Both return the same `Box` domain model. Real-model failures are explicit and are never replaced silently in a research study.

### Experimental Services

`backend.studies` resolves the dataset, selects a deterministic class-aware subset, executes the model-condition matrix in a background thread, reports progress, and persists accepted results. Clean predictions are evaluated once per model and represented across all three degradation curves.

### Evaluation and Evidence

`backend.metrics` calculates official COCO metrics and direct failure counts from one shared sample structure. `backend.study_storage` stores study identity, status, configuration, results, and errors in SQLite. `backend.transitions` reads frozen object-level analysis records. Interface pages read completed evidence instead of recomputing reported values.

## Failure Handling

- Invalid route parameters are normalised or rejected before execution.
- Oversized or malformed JSON requests fail at the HTTP boundary.
- A real detector load or inference failure fails the condition.
- Cancellation is represented as a durable study state.
- Dataset and model identifiers are stored with the result.
- Runtime state is excluded from Git and cannot be served as a static asset.

## Extension Points

Add a detector by implementing the common `Box` output contract in `backend.models`. Add a degradation in `backend.images`, then declare its ordered parameters and include it in the study matrix. Add a metric in `backend.metrics` and persist it through the row construction in `backend.studies`. Each extension should include deterministic unit tests and one real-model smoke test.
