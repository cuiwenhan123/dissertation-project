import { createApiClient } from "./core/api.js";
import { downloadText } from "./core/download.js";
import { delay, isZipFile, readFileAsDataUrl } from "./core/files.js";
import {
  capitalise,
  compactName,
  csvCell,
  escapeHtml,
  failureSummary,
  formatMetric,
  prettifyBackend,
  prettifyDegradation,
  prettifyModel,
  setText,
} from "./core/format.js";
import {
  benchmarkSentence,
  buildReportMarkdown,
  classRiskNote,
  classRiskScore,
  failureNoteForRow,
  reportBullets,
} from "./features/reporting.js";
import { createTransitionFeature } from "./features/transitions.js";

const scene = document.querySelector("#scene");
const model = document.querySelector("#model");
const degradation = document.querySelector("#degradation");
const severity = document.querySelector("#severity");
const runButton = document.querySelector("#run");
const compareButton = document.querySelector("#compare");
const sweepButton = document.querySelector("#sweep");
const benchmarkButton = document.querySelector("#benchmark");
const cancelBenchmarkButton = document.querySelector("#cancelBenchmark");
const studyDatasetSource = document.querySelector("#studyDatasetSource");
const studyImageCount = document.querySelector("#studyImageCount");
const studySeed = document.querySelector("#studySeed");
const curvesRunButton = document.querySelector("#curvesRun");
const curvesJsonButton = document.querySelector("#curvesJson");
const exportCsvButton = document.querySelector("#exportCsv");
const exportJsonButton = document.querySelector("#exportJson");
const uploadRunButton = document.querySelector("#uploadRun");
const datasetCsvButton = document.querySelector("#datasetCsv");
const datasetJsonButton = document.querySelector("#datasetJson");
const failureJsonButton = document.querySelector("#failureJson");
const classJsonButton = document.querySelector("#classJson");
const refreshReportButton = document.querySelector("#refreshReport");
const exportReportButton = document.querySelector("#exportReport");
const printReportButton = document.querySelector("#printReport");
const refreshRunsButton = document.querySelector("#refreshRuns");
const uploadFile = document.querySelector("#uploadFile");
const uploadModel = document.querySelector("#uploadModel");
const uploadDegradation = document.querySelector("#uploadDegradation");
const uploadSeverity = document.querySelector("#uploadSeverity");
const uploadSeverityBadge = document.querySelector("#uploadSeverityBadge");
const runtimeDot = document.querySelector("#runtimeDot");
const runtimeLabel = document.querySelector("#runtimeLabel");
const severityBadge = document.querySelector("#severityBadge");
const { getJson, postJson, requestJson } = createApiClient((message) => {
  runtimeLabel.textContent = message;
});
const transitions = createTransitionFeature({
  getJson,
  onError: showError,
  onUpdated: renderReport,
});
const pages = Array.from(document.querySelectorAll(".page"));
const navTabs = Array.from(document.querySelectorAll(".navTab"));
let evaluateSeq = 0;
let history = [];
let benchmarkResult = null;
let uploadBatchResults = [];
let selectedUploadIndex = -1;
let reportMarkdownText = "";
let datasetPreview = null;
let savedRuns = [];
let studyPollSequence = 0;
const STORAGE_KEYS = {
  history: "robustnessWorkbench.history",
  benchmark: "robustnessWorkbench.benchmark",
  uploadBatch: "robustnessWorkbench.uploadBatch",
  activeStudy: "robustnessWorkbench.activeStudy",
};

async function init() {
  restoreLocalState();
  try {
    await refreshSavedRuns();
    const data = await getJson("/api/status", 15000);
    updateStatus(data);
    if (Array.isArray(data.scenes) && data.scenes.length) {
      const selected = scene.value || "street";
      scene.innerHTML = data.scenes.map((item) => `<option value="${item.id}">${item.name}</option>`).join("");
      scene.value = selected;
    }
  } catch (error) {
    showError(`Status unavailable: ${error.message}`);
  }
  primeControlledExperiment();
  await resumeStudyOrLoadLatest();
  await transitions.load(true);
}

function primeControlledExperiment() {
  setRunning(runButton, false, "Evaluate");
  severityBadge.textContent = `Severity ${severity.value}`;
  runtimeDot.classList.remove("running");
  if (!runtimeLabel.textContent || runtimeLabel.textContent === "Evaluating") {
    runtimeLabel.textContent = "Ready";
  }
  document.querySelector("#backend").textContent = "Not run yet";
  document.querySelector("#map").textContent = "-";
  document.querySelector("#ap50").textContent = "-";
  document.querySelector("#missed").textContent = "-";
  document.querySelector("#falsePositive").textContent = "-";
  document.querySelector("#analysisText").textContent = "Choose a scene, degradation, and model, then run the first evaluation.";
}

async function evaluate() {
  const seq = ++evaluateSeq;
  setRunning(runButton, true, "Running");
  const params = new URLSearchParams({
    scene: scene.value || "street",
    model: model.value,
    degradation: degradation.value,
    severity: severity.value,
  });
  try {
    const data = await getJson(`/api/evaluate?${params}`, 180000);
    if (seq !== evaluateSeq) return;
    document.querySelector("#clean").src = data.cleanImage;
    document.querySelector("#result").src = data.resultImage;
    document.querySelector("#backend").textContent = prettifyBackend(data.backend);
    document.querySelector("#map").textContent = data.metrics.map.toFixed(2);
    document.querySelector("#ap50").textContent = data.metrics.ap50.toFixed(2);
    document.querySelector("#missed").textContent = data.metrics.failures.missed;
    document.querySelector("#falsePositive").textContent = data.metrics.failures.falsePositive;
    document.querySelector("#analysisText").textContent = describeFailures(data);
    recordRun(data);
    updateStatus(data.runtime);
    refreshSavedRuns();
  } catch (error) {
    if (seq === evaluateSeq) showError(`Evaluation unavailable: ${error.message}`);
  } finally {
    if (seq === evaluateSeq) setRunning(runButton, false, "Evaluate");
  }
}

async function compareModels() {
  setRunning(compareButton, true, "Comparing");
  const params = new URLSearchParams({
    scene: scene.value || "street",
    model: model.value,
    degradation: degradation.value,
    severity: severity.value,
  });
  try {
    const data = await getJson(`/api/compare?${params}`, 240000);
    renderComparison(data);
    updateStatus(data.runtime);
  } catch (error) {
    showError(`Comparison unavailable: ${error.message}`);
  } finally {
    setRunning(compareButton, false, "Compare");
  }
}

async function runSweep() {
  setRunning(sweepButton, true, "Sweeping");
  const params = new URLSearchParams({
    scene: scene.value || "street",
    model: model.value,
    degradation: degradation.value,
    severity: severity.value,
  });
  try {
    const data = await getJson(`/api/sweep?${params}`, 30000);
    renderSweep(data);
    updateStatus(data.runtime);
  } catch (error) {
    showError(`Sweep unavailable: ${error.message}`);
  } finally {
    setRunning(sweepButton, false, "Sweep");
  }
}

async function runBenchmark() {
  const payload = {
    datasetSource: studyDatasetSource.value,
    maxImages: Number(studyImageCount.value),
    seed: Number(studySeed.value),
  };
  if (studyDatasetSource.value === "selected") {
    const archiveFile = Array.from(uploadFile.files || []).find(isZipFile);
    if (!archiveFile) {
      showError("Select a labelled ZIP on the Dataset page or choose Bundled COCO128.");
      return;
    }
    payload.archive = await readFileAsDataUrl(archiveFile);
    payload.datasetName = archiveFile.name;
  }
  setRunning(benchmarkButton, true, "Running study");
  cancelBenchmarkButton.disabled = false;
  try {
    const job = await postJson("/api/study/start", payload, 30000);
    localStorage.setItem(STORAGE_KEYS.activeStudy, job.id);
    await pollStudy(job.id);
  } catch (error) {
    showError(`Study unavailable: ${error.message}`);
    setRunning(benchmarkButton, false, "Start study");
    cancelBenchmarkButton.disabled = true;
  }
}

async function pollStudy(studyId) {
  const sequence = ++studyPollSequence;
  while (sequence === studyPollSequence) {
    const job = await getJson(`/api/study/status?id=${encodeURIComponent(studyId)}`, 15000);
    renderStudyStatus(job);
    if (job.status === "completed") {
      benchmarkResult = job.result;
      renderDashboard(benchmarkResult);
      exportCsvButton.disabled = false;
      exportJsonButton.disabled = false;
      localStorage.removeItem(STORAGE_KEYS.activeStudy);
      setRunning(benchmarkButton, false, "Start study");
      cancelBenchmarkButton.disabled = true;
      runtimeLabel.textContent = "Real study completed";
      runtimeDot.classList.add("ready");
      refreshSavedRuns();
      return;
    }
    if (["failed", "cancelled", "not-found"].includes(job.status)) {
      localStorage.removeItem(STORAGE_KEYS.activeStudy);
      setRunning(benchmarkButton, false, "Start study");
      cancelBenchmarkButton.disabled = true;
      if (job.status === "failed") throw new Error(job.error || "Study failed");
      runtimeLabel.textContent = job.status === "cancelled" ? "Study cancelled" : "Study not found";
      return;
    }
    await delay(1500);
  }
}

function renderStudyStatus(job) {
  const progress = Math.max(0, Math.min(1, Number(job.progress || 0)));
  document.querySelector("#studyStatus").textContent = String(job.status || "ready").replaceAll("-", " ");
  document.querySelector("#studyProgress").value = progress;
  document.querySelector("#studyProgressValue").textContent = `${Math.round(progress * 100)}%`;
  document.querySelector("#studyCurrent").textContent = job.current || "Waiting";
  document.querySelector("#studyTaskCount").textContent = `${job.completedTasks || 0} / ${job.totalTasks || 0} inference tasks`;
}

async function cancelBenchmark() {
  const studyId = localStorage.getItem(STORAGE_KEYS.activeStudy);
  if (!studyId) return;
  cancelBenchmarkButton.disabled = true;
  try {
    const job = await postJson("/api/study/cancel", { id: studyId }, 15000);
    renderStudyStatus(job);
  } catch (error) {
    showError(`Cancellation unavailable: ${error.message}`);
  }
}

async function resumeStudyOrLoadLatest() {
  const activeStudy = localStorage.getItem(STORAGE_KEYS.activeStudy);
  if (activeStudy) {
    setRunning(benchmarkButton, true, "Running study");
    cancelBenchmarkButton.disabled = false;
    pollStudy(activeStudy).catch((error) => {
      showError(`Study unavailable: ${error.message}`);
      setRunning(benchmarkButton, false, "Start study");
      cancelBenchmarkButton.disabled = true;
    });
    return;
  }
  try {
    const latest = await requestJson("/api/study/latest", 10000);
    benchmarkResult = latest;
    renderDashboard(latest);
    exportCsvButton.disabled = false;
    exportJsonButton.disabled = false;
    const imageCount = latest.config?.dataset?.evaluatedImages || 0;
    renderStudyStatus({
      status: "completed",
      progress: 1,
      current: `Synced saved study: ${latest.id}`,
      completedTasks: imageCount * 32,
      totalTasks: imageCount * 32,
    });
  } catch {
    // A first-time project has no completed study yet.
  } finally {
    saveLocalState();
  }
}

async function runUploadEvaluation() {
  const files = Array.from(uploadFile.files || []);
  if (!files.length) {
    document.querySelector("#uploadAnalysisText").textContent = "Select one or more image files before running batch evaluation.";
    return;
  }
  uploadBatchResults = [];
  selectedUploadIndex = -1;
  syncDatasetSettingsToControlled();
  renderUploadBatch();
  setRunning(uploadRunButton, true, "Running");
  try {
    for (const [index, file] of files.entries()) {
      document.querySelector("#uploadBatchProgress").textContent = `${index + 1} / ${files.length}: ${file.name}`;
      try {
        if (isZipFile(file)) {
          const archive = await readFileAsDataUrl(file);
          const data = await postJson("/api/upload-zip-evaluate", {
            archive,
            model: uploadModel.value,
            degradation: uploadDegradation.value,
            severity: uploadSeverity.value,
          }, 600000);
          if (data.error) throw new Error(data.error);
          uploadBatchResults.push(...data.rows);
          if (data.lastResult) {
            selectedUploadIndex = uploadBatchResults.length - 1;
            renderUploadResult(data.lastResult);
          }
          updateStatus(data.runtime);
          refreshSavedRuns();
          if (data.truncated) {
            document.querySelector("#uploadAnalysisText").textContent = "Zip archive contained more than 100 images; only the first 100 were evaluated.";
          }
        } else {
          const image = await readFileAsDataUrl(file);
          document.querySelector("#uploadOriginal").src = image;
          const data = await postJson("/api/upload-evaluate", {
            image,
            imageName: file.name,
            model: uploadModel.value,
            degradation: uploadDegradation.value,
            severity: uploadSeverity.value,
          }, 180000);
          if (data.error) throw new Error(data.error);
          uploadBatchResults.push(data.row || uploadRowFromResult(data, file.name));
          selectedUploadIndex = uploadBatchResults.length - 1;
          renderUploadResult(data);
          updateStatus(data.runtime);
          refreshSavedRuns();
        }
      } catch (itemError) {
        uploadBatchResults.push({
          imageName: file.name,
          model: uploadModel.value,
          degradation: uploadDegradation.value,
          severity: Number(uploadSeverity.value),
          backend: "failed",
          predictionCount: 0,
          meanConfidence: 0,
          small: 0,
          medium: 0,
          large: 0,
          labelAvailable: false,
          gtCount: 0,
          matched: 0,
          missed: 0,
          falsePositive: 0,
          ap50: null,
          meanIou: null,
          classMetrics: {},
          error: itemError.message,
        });
      }
      renderUploadBatch();
    }
    document.querySelector("#uploadBatchProgress").textContent = `Completed ${files.length} image${files.length === 1 ? "" : "s"}`;
    document.querySelector("#uploadAnalysisText").textContent = describeUploadBatch();
    recordDatasetBatch();
    renderLinkedDatasetEvidence();
    renderReport();
    saveLocalState();
  } catch (error) {
    showError(`Upload evaluation unavailable: ${error.message}`);
    document.querySelector("#uploadAnalysisText").textContent = error.message;
  } finally {
    setRunning(uploadRunButton, false, "Run batch");
  }
}

function showPage(pageName) {
  for (const page of pages) {
    page.classList.toggle("active", page.dataset.page === pageName);
  }
  for (const tab of navTabs) {
    tab.classList.toggle("active", tab.dataset.targetPage === pageName);
  }
  renderLinkedDatasetEvidence();
  if (pageName === "report") renderReport();
  if (pageName === "classAnalysis") renderClassAnalysis();
  if (pageName === "failures") renderFailureCases();
  if (pageName === "transitions" && !transitions.getResult()) transitions.load();
  if (pageName === "curves" && benchmarkResult) renderCurvesDashboard(benchmarkResult);
}

async function syncDatasetSettingsToControlled() {
  model.value = uploadModel.value;
  degradation.value = uploadDegradation.value;
  severity.value = uploadSeverity.value;
  severityBadge.textContent = `Severity ${severity.value}`;
  uploadSeverityBadge.textContent = `Severity ${uploadSeverity.value}`;
  runtimeLabel.textContent = "Dataset settings linked";
}

function updateStatus(data) {
  const loaded = (data.modelsLoaded || []).join(", ") || "none";
  const errors = Object.entries(data.modelErrors || {})
    .map(([name, message]) => `${name}: ${message}`)
    .join(" | ");
  runtimeLabel.textContent = errors ? "Needs attention" : loaded === "none" ? "Ready" : "Models loaded";
  runtimeDot.classList.toggle("ready", !errors);
  const cocoOption = studyDatasetSource?.querySelector('option[value="coco-val2017"]');
  if (cocoOption) {
    const available = Boolean(data.localDatasets?.cocoVal2017?.available);
    cocoOption.disabled = !available;
    cocoOption.textContent = available
      ? "Local COCO val2017 / 5,000 images"
      : "Local COCO val2017 / unavailable";
  }
}

function updateStudyEstimate() {
  const imageCount = Number(studyImageCount.value || 0);
  document.querySelector("#studyTaskCount").textContent = `Up to ${(imageCount * 32).toLocaleString()} inference tasks`;
}

function showError(message) {
  runtimeLabel.textContent = message;
  runtimeDot.classList.remove("ready");
  document.querySelector("#backend").textContent = "Unavailable";
}

function setRunning(button, isRunning, label) {
  button.disabled = isRunning;
  const labelTarget = button.querySelector("span:last-child") || button;
  labelTarget.textContent = label;
  severityBadge.textContent = `Severity ${severity.value}`;
  runtimeDot.classList.toggle("running", isRunning);
  if (isRunning) runtimeLabel.textContent = "Evaluating";
}

function describeFailures(data) {
  const { failures, sizeAP } = data.metrics;
  const parts = [];
  if (failures.missed > 0) parts.push(`${failures.missed} missed object${failures.missed === 1 ? "" : "s"}`);
  if (failures.falsePositive > 0) parts.push(`${failures.falsePositive} false positive${failures.falsePositive === 1 ? "" : "s"}`);
  if (failures.classification > 0) parts.push(`${failures.classification} class error${failures.classification === 1 ? "" : "s"}`);
  if (failures.localisation > 0) parts.push(`${failures.localisation} localisation error${failures.localisation === 1 ? "" : "s"}`);
  const weakestSize = Object.entries(sizeAP).sort((a, b) => a[1] - b[1])[0][0];
  const failureText = parts.length ? parts.join(", ") : "no major failure";
  return `${prettifyModel(data.model)} under ${prettifyDegradation(data.degradation)} severity ${data.severity}: ${failureText}. Weakest object-size band: ${weakestSize}.`;
}

function describeUpload(data) {
  const counts = data.summary.sizeCounts;
  return `${prettifyModel(data.model)} on uploaded image after ${prettifyDegradation(data.degradation)} severity ${data.severity}: ${data.summary.predictionCount} predictions, mean confidence ${data.summary.meanConfidence.toFixed(2)}. Size distribution: ${counts.small} small, ${counts.medium} medium, ${counts.large} large.`;
}

function renderUploadResult(data) {
  if (data.cleanImage) document.querySelector("#uploadOriginal").src = data.cleanImage;
  document.querySelector("#uploadResult").src = data.resultImage;
  document.querySelector("#uploadBackend").textContent = prettifyBackend(data.backend);
  document.querySelector("#uploadPredictionCount").textContent = data.summary.predictionCount;
  document.querySelector("#uploadMeanConfidence").textContent = data.summary.meanConfidence.toFixed(2);
  document.querySelector("#uploadSmallCount").textContent = data.summary.sizeCounts.small;
  document.querySelector("#uploadLargeCount").textContent = data.summary.sizeCounts.large;
  document.querySelector("#uploadAnalysisText").textContent = describeUpload(data);
}

function uploadRowFromResult(data, imageName) {
  return {
    imageName,
    model: data.model,
    degradation: data.degradation,
    severity: data.severity,
    backend: data.backend,
    predictionCount: data.summary.predictionCount,
    meanConfidence: data.summary.meanConfidence,
    small: data.summary.sizeCounts.small,
    medium: data.summary.sizeCounts.medium,
    large: data.summary.sizeCounts.large,
    annotationFormat: data.summary.groundTruthMetrics?.labelAvailable ? "provided" : "none",
    labelAvailable: data.summary.groundTruthMetrics?.labelAvailable || false,
    gtCount: data.summary.groundTruthMetrics?.gtCount || 0,
    matched: data.summary.groundTruthMetrics?.matched || 0,
    missed: data.summary.groundTruthMetrics?.missed || 0,
    falsePositive: data.summary.groundTruthMetrics?.falsePositive || 0,
    ap50: data.summary.groundTruthMetrics?.ap50 ?? null,
    meanIou: data.summary.groundTruthMetrics?.meanIou ?? null,
    classMetrics: data.summary.groundTruthMetrics?.classMetrics || {},
    cleanImage: data.cleanImage,
    resultImage: data.resultImage,
    error: "",
  };
}

function renderUploadSelection(index) {
  const row = uploadBatchResults[index];
  if (!row || row.error || !row.resultImage) return;
  selectedUploadIndex = index;
  document.querySelector("#uploadOriginal").src = row.cleanImage;
  document.querySelector("#uploadResult").src = row.resultImage;
  document.querySelector("#uploadBackend").textContent = prettifyBackend(row.backend);
  document.querySelector("#uploadPredictionCount").textContent = row.predictionCount;
  document.querySelector("#uploadMeanConfidence").textContent = row.meanConfidence.toFixed(2);
  document.querySelector("#uploadSmallCount").textContent = row.small;
  document.querySelector("#uploadLargeCount").textContent = row.large;
  const annotationText = row.annotationFormat && row.annotationFormat !== "none" ? ` Annotation ${row.annotationFormat}.` : "";
  const gtText = row.labelAvailable
    ? ` GT ${row.gtCount}, matched ${row.matched}, missed ${row.missed}, AP50 ${formatMetric(row.ap50)}.${annotationText}`
    : " No ground-truth label was available for this image.";
  document.querySelector("#uploadAnalysisText").textContent = `${row.imageName}: ${prettifyModel(row.model)} after ${prettifyDegradation(row.degradation)} severity ${row.severity}, ${row.predictionCount} predictions, mean confidence ${row.meanConfidence.toFixed(2)}.${gtText}`;
  renderUploadBatch();
}

function renderUploadBatch() {
  const body = document.querySelector("#uploadBatchBody");
  const count = uploadBatchResults.length;
  document.querySelector("#uploadBatchCount").textContent = `${count} image${count === 1 ? "" : "s"}`;
  datasetCsvButton.disabled = count === 0;
  datasetJsonButton.disabled = count === 0;
  if (!count) {
    body.innerHTML = `<tr><td colspan="16">No uploaded-image experiment recorded yet.</td></tr>`;
    document.querySelector("#datasetMeanDetections").textContent = "-";
    document.querySelector("#datasetMeanConfidence").textContent = "-";
    document.querySelector("#datasetWeakestImage").textContent = "-";
    document.querySelector("#datasetWeakestDetail").textContent = "Lowest confidence or no predictions.";
    document.querySelector("#datasetLabelCoverage").textContent = "-";
    document.querySelector("#datasetLabelDetail").textContent = "YOLO label files enable real AP and IoU scoring.";
    document.querySelector("#datasetLabelAp50").textContent = "-";
    document.querySelector("#datasetLabelAp50Detail").textContent = "Average AP50 for images with ground-truth labels.";
    document.querySelector("#datasetBackendSummary").textContent = "-";
    renderLinkedDatasetEvidence();
    renderClassAnalysis();
    return;
  }
  const meanDetections = uploadBatchResults.reduce((sum, row) => sum + row.predictionCount, 0) / count;
  const meanConfidence = uploadBatchResults.reduce((sum, row) => sum + row.meanConfidence, 0) / count;
  const weakest = uploadBatchResults
    .slice()
    .sort((a, b) => a.meanConfidence - b.meanConfidence || a.predictionCount - b.predictionCount)[0];
  const last = uploadBatchResults[count - 1];
  document.querySelector("#datasetMeanDetections").textContent = meanDetections.toFixed(1);
  document.querySelector("#datasetMeanConfidence").textContent = meanConfidence.toFixed(2);
  document.querySelector("#datasetWeakestImage").textContent = compactName(weakest.imageName);
  document.querySelector("#datasetWeakestDetail").textContent = weakest.error
    ? `Failed: ${weakest.error}`
    : `${weakest.predictionCount} predictions, mean confidence ${weakest.meanConfidence.toFixed(2)}.`;
  const labelled = uploadBatchResults.filter((row) => row.labelAvailable);
  const meanLabelAp50 = labelled.length ? labelled.reduce((sum, row) => sum + Number(row.ap50 || 0), 0) / labelled.length : null;
  const totalGt = labelled.reduce((sum, row) => sum + Number(row.gtCount || 0), 0);
  const totalMissed = labelled.reduce((sum, row) => sum + Number(row.missed || 0), 0);
  document.querySelector("#datasetLabelCoverage").textContent = `${labelled.length}/${count}`;
  document.querySelector("#datasetLabelDetail").textContent = labelled.length
    ? `${totalGt} GT objects, ${totalMissed} missed across labelled images.`
    : "Add YOLO .txt labels with matching image filenames inside the zip to enable AP/IoU metrics.";
  document.querySelector("#datasetLabelAp50").textContent = formatMetric(meanLabelAp50);
  document.querySelector("#datasetLabelAp50Detail").textContent = labelled.length
    ? `Computed from ${labelled.length} labelled image${labelled.length === 1 ? "" : "s"}.`
    : "Confidence-only evidence until labels are provided.";
  document.querySelector("#datasetBackendSummary").textContent = prettifyBackend(last.backend);
  body.innerHTML = uploadBatchResults.map((row, index) => `
    <tr class="${index === selectedUploadIndex ? "selectedRow" : ""}">
      <td>${index + 1}</td>
      <td title="${escapeHtml(row.imageName)}">${escapeHtml(compactName(row.imageName))}</td>
      <td>${prettifyModel(row.model)}</td>
      <td>${prettifyDegradation(row.degradation)}</td>
      <td>${row.severity}</td>
      <td>${escapeHtml(row.annotationFormat || "none")}</td>
      <td>${row.labelAvailable ? row.gtCount : "-"}</td>
      <td>${row.labelAvailable ? row.matched : "-"}</td>
      <td>${row.labelAvailable ? row.missed : "-"}</td>
      <td>${row.labelAvailable ? row.falsePositive : "-"}</td>
      <td>${formatMetric(row.ap50)}</td>
      <td>${formatMetric(row.meanIou)}</td>
      <td>${row.predictionCount}</td>
      <td>${row.meanConfidence.toFixed(2)}</td>
      <td title="${escapeHtml(row.error || prettifyBackend(row.backend))}">${row.error ? "Failed" : prettifyBackend(row.backend)}</td>
      <td><button class="miniButton" type="button" data-upload-view="${index}" ${row.error || !row.resultImage ? "disabled" : ""}>View</button></td>
    </tr>
  `).join("");
  renderLinkedDatasetEvidence();
  renderClassAnalysis();
  renderFailureCases();
}

function buildClassAnalysis() {
  const aggregate = {};
  const sourceRows = uploadBatchResults.length
    ? uploadBatchResults
    : (benchmarkResult?.rows || []).filter((row) => row.severity === 5);
  for (const row of sourceRows) {
    if (row.error) continue;
    const metrics = row.classMetrics || {};
    if (Object.keys(metrics).length) {
      for (const [label, values] of Object.entries(metrics)) {
        const item = aggregate[label] ||= { label, gt: 0, matched: 0, missed: 0, falsePositive: 0, iouTotal: 0, iouCount: 0, ap50Total: 0, ap50Count: 0 };
        item.gt += Number(values.gt || 0);
        item.matched += Number(values.matched || 0);
        item.missed += Number(values.missed || 0);
        item.falsePositive += Number(values.falsePositive || 0);
        if (values.iouCount) {
          item.iouTotal += Number(values.iouTotal || 0);
          item.iouCount += Number(values.iouCount || 0);
        } else if (values.meanIou != null && values.matched) {
          item.iouTotal += Number(values.meanIou) * Number(values.matched);
          item.iouCount += Number(values.matched);
        }
        if (values.ap50 != null) {
          item.ap50Total += Number(values.ap50);
          item.ap50Count += 1;
        }
      }
    } else if (row.labelAvailable) {
      const label = "labelled objects";
      const item = aggregate[label] ||= { label, gt: 0, matched: 0, missed: 0, falsePositive: 0, iouTotal: 0, iouCount: 0, ap50Total: 0, ap50Count: 0 };
      item.gt += Number(row.gtCount || 0);
      item.matched += Number(row.matched || 0);
      item.missed += Number(row.missed || 0);
      item.falsePositive += Number(row.falsePositive || 0);
      if (row.meanIou != null && row.matched) {
        item.iouTotal += Number(row.meanIou) * Number(row.matched);
        item.iouCount += Number(row.matched);
      }
    }
  }
  return Object.values(aggregate)
    .map((item) => ({
      ...item,
      ap50: item.ap50Count ? item.ap50Total / item.ap50Count : item.gt ? item.matched / item.gt : null,
      meanIou: item.iouCount ? item.iouTotal / item.iouCount : null,
      risk: classRiskScore(item),
    }))
    .sort((a, b) => b.risk - a.risk || b.missed - a.missed || b.falsePositive - a.falsePositive || a.label.localeCompare(b.label));
}

function renderClassAnalysis() {
  const rows = buildClassAnalysis();
  const body = document.querySelector("#classAnalysisBody");
  if (!body) return;
  const labelledRows = uploadBatchResults.filter((row) => row.labelAvailable).length;
  const usingStudy = !uploadBatchResults.length && Boolean(benchmarkResult?.rows?.length);
  classJsonButton.disabled = rows.length === 0;
  document.querySelector("#classStatus").textContent = rows.length
    ? usingStudy ? "Severity-5 study evidence" : "Class evidence available"
    : "Awaiting labelled Dataset batch";
  document.querySelector("#classLabelCount").textContent = rows.length || "-";
  document.querySelector("#classTableCount").textContent = `${rows.length} row${rows.length === 1 ? "" : "s"}`;
  document.querySelector("#classDatasetLink").textContent = usingStudy
    ? `${benchmarkResult.config.dataset.evaluatedImages} images / 6 S5 conditions`
    : uploadBatchResults.length ? `${uploadBatchResults.length} images / ${labelledRows} labelled` : "-";
  const gtRows = rows.filter((row) => row.gt > 0);
  const meanClassAp50 = gtRows.length ? gtRows.reduce((sum, row) => sum + Number(row.ap50 || 0), 0) / gtRows.length : null;
  document.querySelector("#classMeanAp50").textContent = formatMetric(meanClassAp50);
  const weakest = gtRows[0] || rows[0];
  document.querySelector("#classWorstClass").textContent = weakest ? weakest.label : "-";
  document.querySelector("#classWorstDetail").textContent = weakest ? classRiskNote(weakest) : "Run a labelled zip batch to populate class-level errors.";
  body.innerHTML = rows.length
    ? rows.map((row) => `
      <tr>
        <td>${escapeHtml(row.label)}</td>
        <td>${row.gt || "-"}</td>
        <td>${row.matched || "-"}</td>
        <td>${row.missed || "-"}</td>
        <td>${row.falsePositive || "-"}</td>
        <td>${formatMetric(row.ap50)}</td>
        <td>${formatMetric(row.meanIou)}</td>
        <td>${escapeHtml(classRiskNote(row))}</td>
      </tr>
    `).join("")
    : `<tr><td colspan="8">No class evidence yet. Upload a labelled zip dataset first.</td></tr>`;
}

function exportClassJson() {
  const rows = buildClassAnalysis();
  if (!rows.length) return;
  downloadText("robustness-class-analysis.json", "application/json", JSON.stringify({
    generatedAt: new Date().toISOString(),
    datasetPreview,
    rows: rows.map((row) => ({ ...row, note: classRiskNote(row) })),
  }, null, 2));
}

async function refreshSavedRuns() {
  try {
    const data = await getJson("/api/runs", 15000);
    savedRuns = Array.isArray(data.runs) ? data.runs.slice().reverse() : [];
    renderSavedRuns();
  } catch {
    savedRuns = [];
    renderSavedRuns();
  }
}

function renderSavedRuns() {
  const body = document.querySelector("#savedRunBody");
  const count = document.querySelector("#savedRunCount");
  if (!body || !count) return;
  count.textContent = `${savedRuns.length} saved`;
  if (!savedRuns.length) {
    body.innerHTML = `<tr><td colspan="5">No backend-saved experiments yet.</td></tr>`;
    return;
  }
  body.innerHTML = savedRuns.slice(0, 30).map((run, index) => {
    const payload = run.payload || {};
    const summary = payload.summary || payload.metrics || {};
    const condition = savedRunCondition(run);
    const evidence = run.kind === "dataset-batch"
      ? `${summary.imageCount || 0} images, ${summary.labelledCount || 0} labelled, AP50 ${formatMetric(summary.meanAp50)}`
      : run.kind === "benchmark"
      ? `${payload.rowCount || summary.runCount || 0} controlled rows`
      : `mAP ${formatMetric(summary.map)}, AP50 ${formatMetric(summary.ap50)}`;
    return `
      <tr>
        <td>${index + 1}</td>
        <td>${escapeHtml(run.kind)}</td>
        <td>${escapeHtml(run.createdAt || "-")}</td>
        <td>${escapeHtml(condition)}</td>
        <td>${escapeHtml(evidence)}</td>
      </tr>
    `;
  }).join("");
}

function savedRunCondition(run) {
  const payload = run.payload || {};
  if (run.kind === "benchmark") return "full controlled grid";
  const sceneName = payload.sceneName || payload.scene || "uploaded dataset";
  const modelName = payload.model ? prettifyModel(payload.model) : payload.summary?.bestModel ? prettifyModel(payload.summary.bestModel) : "-";
  const degradationName = payload.degradation ? prettifyDegradation(payload.degradation) : "-";
  const severityValue = payload.severity ?? "-";
  return `${sceneName}, ${modelName}, ${degradationName}, severity ${severityValue}`;
}

function renderDatasetPreview(preview) {
  datasetPreview = preview;
  if (!preview) {
    document.querySelector("#datasetPreviewStatus").textContent = "Awaiting files";
    document.querySelector("#previewImageCount").textContent = "-";
    document.querySelector("#previewLabelCount").textContent = "-";
    document.querySelector("#previewCoverage").textContent = "-";
    document.querySelector("#previewCoverageDetail").textContent = "Preview labels before running expensive inference.";
    document.querySelector("#previewClassSummary").textContent = "-";
    document.querySelector("#previewSampleList").textContent = "Select images or a zip archive to inspect dataset composition.";
    return;
  }
  const classEntries = Object.entries(preview.classCounts || {});
  const sampleNames = preview.sampleImages || [];
  document.querySelector("#datasetPreviewStatus").textContent = preview.source || "Ready";
  document.querySelector("#previewImageCount").textContent = preview.imageCount;
  document.querySelector("#previewLabelCount").textContent = `${preview.matchedLabelCount || 0}/${preview.imageCount || 0}`;
  document.querySelector("#previewCoverage").textContent = `${Math.round((preview.labelCoverage || 0) * 100)}%`;
  document.querySelector("#previewCoverageDetail").textContent = preview.truncatedOnEvaluation
    ? "Archive has more than 100 images; batch evaluation will use the first 100."
    : preview.unreadableImageCount
    ? `${preview.unreadableImageCount} unreadable image${preview.unreadableImageCount === 1 ? "" : "s"} detected.`
    : preview.annotationFormat && preview.annotationFormat !== "none"
    ? `${preview.annotationFormat} annotations found and ready for evaluation.`
    : "Dataset structure is ready for evaluation.";
  const formatLabel = preview.annotationFormat && preview.annotationFormat !== "none" ? `${preview.annotationFormat}: ` : "";
  document.querySelector("#previewClassSummary").textContent = classEntries.length
    ? `${formatLabel}${classEntries.slice(0, 2).map(([label, count]) => `${label} ${count}`).join(", ")}`
    : preview.annotationFormat && preview.annotationFormat !== "none" ? preview.annotationFormat : "-";
  document.querySelector("#previewSampleList").innerHTML = sampleNames.length
    ? sampleNames.map((name) => `<span class="sampleChip" title="${escapeHtml(name)}">${escapeHtml(compactName(name))}</span>`).join("")
    : "No supported image files found.";
}

async function inspectSelectedDataset(files) {
  if (!files.length) {
    renderDatasetPreview(null);
    return;
  }
  const zipFiles = files.filter(isZipFile);
  if (zipFiles.length === 1 && files.length === 1) {
    renderDatasetPreview({ source: "Inspecting archive", imageCount: "...", matchedLabelCount: 0, labelCoverage: 0, classCounts: {}, sampleImages: [] });
    try {
      const archive = await readFileAsDataUrl(zipFiles[0]);
      const data = await postJson("/api/inspect-zip", { archive }, 120000);
      if (data.error) throw new Error(data.error);
      renderDatasetPreview({ ...data, source: "Archive preview" });
      updateStatus(data.runtime);
    } catch (error) {
      renderDatasetPreview({ source: "Preview failed", imageCount: 0, matchedLabelCount: 0, labelCoverage: 0, classCounts: {}, sampleImages: [error.message] });
    }
    return;
  }
  const imageFiles = files.filter((file) => !isZipFile(file));
  renderDatasetPreview({
    source: "Image selection",
    imageCount: imageFiles.length,
    matchedLabelCount: 0,
    labelCoverage: 0,
    classCounts: {},
    sampleImages: imageFiles.slice(0, 8).map((file) => file.name),
    unreadableImageCount: 0,
    truncatedOnEvaluation: imageFiles.length > 100,
  });
}

function describeUploadBatch() {
  if (!uploadBatchResults.length) return "No uploaded-image experiment recorded yet.";
  const meanConfidence = uploadBatchResults.reduce((sum, row) => sum + row.meanConfidence, 0) / uploadBatchResults.length;
  const totalPredictions = uploadBatchResults.reduce((sum, row) => sum + row.predictionCount, 0);
  const failedCount = uploadBatchResults.filter((row) => row.error).length;
  const zeroDetectionCount = uploadBatchResults.filter((row) => !row.error && row.predictionCount === 0).length;
  const weakest = uploadBatchResults.slice().sort((a, b) => a.meanConfidence - b.meanConfidence)[0];
  const failureNote = failedCount
    ? `${failedCount} image${failedCount === 1 ? "" : "s"} failed and were kept in the table.`
    : zeroDetectionCount
    ? `${zeroDetectionCount} image${zeroDetectionCount === 1 ? "" : "s"} produced no predictions.`
    : `Weakest image: ${weakest.imageName} with mean confidence ${weakest.meanConfidence.toFixed(2)}.`;
  return `Batch completed: ${uploadBatchResults.length} image${uploadBatchResults.length === 1 ? "" : "s"}, ${totalPredictions} total predictions, mean confidence ${meanConfidence.toFixed(2)}. ${failureNote}`;
}

function recordRun(data) {
  history.unshift({
    scene: data.sceneName || data.scene,
    model: prettifyModel(data.model),
    degradation: prettifyDegradation(data.degradation),
    severity: data.severity,
    map: data.metrics.map,
    ap50: data.metrics.ap50,
    missed: data.metrics.failures.missed,
    falsePositive: data.metrics.failures.falsePositive,
  });
  if (history.length > 12) history.pop();
  renderHistory();
  saveLocalState();
}

function recordDatasetBatch() {
  if (!uploadBatchResults.length) return;
  const completed = uploadBatchResults.filter((row) => !row.error);
  const failed = uploadBatchResults.length - completed.length;
  const totalPredictions = uploadBatchResults.reduce((sum, row) => sum + row.predictionCount, 0);
  const meanConfidence = uploadBatchResults.reduce((sum, row) => sum + row.meanConfidence, 0) / uploadBatchResults.length;
  history.unshift({
    scene: `Uploaded dataset (${uploadBatchResults.length} images)`,
    model: prettifyModel(uploadModel.value),
    degradation: prettifyDegradation(uploadDegradation.value),
    severity: Number(uploadSeverity.value),
    map: null,
    ap50: meanConfidence,
    missed: failed,
    falsePositive: totalPredictions,
    datasetBatch: true,
  });
  if (history.length > 12) history.pop();
  renderHistory();
}

function renderHistory() {
  const body = document.querySelector("#historyBody");
  document.querySelector("#historyCount").textContent = `${history.length} run${history.length === 1 ? "" : "s"}`;
  if (!history.length) {
    body.innerHTML = `<tr><td colspan="9">No experiment recorded yet.</td></tr>`;
    return;
  }
  body.innerHTML = history.map((row, index) => `
    <tr>
      <td>${history.length - index}</td>
      <td>${row.scene}</td>
      <td>${row.model}</td>
      <td>${row.degradation}</td>
      <td>${row.severity}</td>
      <td>${row.map == null ? "-" : row.map.toFixed(2)}</td>
      <td>${row.datasetBatch ? `conf ${row.ap50.toFixed(2)}` : row.ap50.toFixed(2)}</td>
      <td>${row.missed}</td>
      <td>${row.falsePositive}</td>
    </tr>
  `).join("");
}

function renderLinkedDatasetEvidence() {
  const summary = datasetEvidenceSummary();
  setText("#comparisonDatasetCondition", summary.condition);
  setText("#comparisonDatasetEvidence", summary.evidence);
  setText("#comparisonDatasetDetail", summary.detail);
  setText("#comparisonDatasetWeakest", summary.weakestName);
  setText("#comparisonDatasetWeakestDetail", summary.weakestDetail);
  setText("#comparisonDatasetLink", summary.linkLabel);
  setText("#benchmarkDatasetCondition", summary.condition);
  setText("#benchmarkDatasetEvidence", summary.evidence);
  setText("#benchmarkDatasetDetail", summary.detail);
  setText("#benchmarkDatasetConfidence", summary.confidence);
  setText("#benchmarkReportLink", summary.reportStatus);
  if (benchmarkResult?.config?.dataset) {
    const dataset = benchmarkResult.config.dataset;
    setText("#benchmarkDatasetCondition", dataset.name);
    setText("#benchmarkDatasetEvidence", `${dataset.evaluatedImages} labelled images`);
    setText("#benchmarkDatasetDetail", `${dataset.annotationFormat}; ${dataset.classMappingSource}.`);
    setText("#benchmarkDatasetConfidence", "COCOeval");
    setText("#benchmarkReportLink", "Reproducible record saved");
  }
}

function datasetEvidenceSummary() {
  if (!uploadBatchResults.length) {
    return {
      condition: `${prettifyModel(uploadModel.value)}, ${prettifyDegradation(uploadDegradation.value)}, severity ${uploadSeverity.value}`,
      evidence: "-",
      detail: "Run Dataset evaluation to connect real-image evidence.",
      weakestName: "-",
      weakestDetail: "No uploaded-image evidence yet.",
      confidence: "-",
      linkLabel: "Controls linked",
      reportStatus: "Awaiting Dataset",
    };
  }
  const completed = uploadBatchResults.filter((row) => !row.error);
  const failed = uploadBatchResults.length - completed.length;
  const totalPredictions = uploadBatchResults.reduce((sum, row) => sum + row.predictionCount, 0);
  const meanConfidence = uploadBatchResults.reduce((sum, row) => sum + row.meanConfidence, 0) / uploadBatchResults.length;
  const weakest = uploadBatchResults
    .slice()
    .sort((a, b) => Number(Boolean(b.error)) - Number(Boolean(a.error)) || a.meanConfidence - b.meanConfidence || a.predictionCount - b.predictionCount)[0];
  const dominantDegradation = uploadBatchResults[0]?.degradation || uploadDegradation.value;
  const severityValue = uploadBatchResults[0]?.severity ?? uploadSeverity.value;
  return {
    condition: `${prettifyModel(uploadBatchResults[0]?.model || uploadModel.value)}, ${prettifyDegradation(dominantDegradation)}, severity ${severityValue}`,
    evidence: `${uploadBatchResults.length} images`,
    detail: `${completed.length} completed, ${failed} failed, ${totalPredictions} total predictions.`,
    weakestName: compactName(weakest.imageName),
    weakestDetail: failureNoteForRow(weakest),
    confidence: meanConfidence.toFixed(2),
    linkLabel: "Dataset-driven",
    reportStatus: benchmarkResult ? "Report-ready" : "Needs Benchmark",
  };
}

function buildFailureCases() {
  const uploadCases = uploadBatchResults
    .map((row, index) => ({ ...classifyFailureRow(row), row, index }))
    .filter((item) => item.risk > 0)
    .sort((a, b) => b.risk - a.risk || a.row.meanConfidence - b.row.meanConfidence || b.row.missed - a.row.missed);
  if (uploadCases.length || uploadBatchResults.length) return uploadCases;
  return (benchmarkResult?.failureExamples || []).map((example, index) => ({
    mode: example.degradedPredictionCount === 0 ? "prediction collapse" : "severe retention loss",
    risk: example.degradedPredictionCount === 0 ? 5 : 4,
    note: `${prettifyModel(example.model)} retained ${example.degradedPredictionCount} of ${example.cleanPredictionCount} predictions with ${example.groundTruthCount} annotated objects.`,
    index,
    studyArchive: true,
    row: {
      imageName: example.image,
      model: example.model,
      degradation: example.degradation,
      severity: example.severity,
      gtCount: example.groundTruthCount,
      missed: null,
      falsePositive: null,
      predictionCount: example.degradedPredictionCount,
      meanConfidence: 0,
      error: "",
      resultImage: null,
    },
  }));
}

function classifyFailureRow(row) {
  if (row.error) return { mode: "runtime failure", risk: 5, note: `Processing failed: ${row.error}` };
  if (row.labelAvailable && row.missed > 0) return { mode: "missed detection", risk: 4 + Math.min(2, row.missed), note: `${row.missed} missed GT object${row.missed === 1 ? "" : "s"}; AP50 ${formatMetric(row.ap50)}.` };
  if (row.labelAvailable && row.falsePositive > 0) return { mode: "false positive", risk: 3 + Math.min(2, row.falsePositive), note: `${row.falsePositive} unmatched prediction${row.falsePositive === 1 ? "" : "s"}; mean IoU ${formatMetric(row.meanIou)}.` };
  if (row.predictionCount === 0) return { mode: "zero detection", risk: 4, note: "No predictions remained after degradation." };
  if (row.meanConfidence < 0.35) return { mode: "very low confidence", risk: 3, note: `Mean confidence ${row.meanConfidence.toFixed(2)}.` };
  if (row.meanConfidence < 0.5) return { mode: "low confidence", risk: 2, note: `Mean confidence ${row.meanConfidence.toFixed(2)}.` };
  if (row.small > row.large && row.meanConfidence < 0.6) return { mode: "small-object sensitivity", risk: 1, note: `${row.small} small-object prediction${row.small === 1 ? "" : "s"} with moderate confidence.` };
  return { mode: "usable", risk: 0, note: "No major failure flag." };
}

function renderFailureCases() {
  const cases = buildFailureCases();
  const body = document.querySelector("#failureCaseBody");
  if (!body) return;
  const usingStudy = !uploadBatchResults.length && cases.some((item) => item.studyArchive);
  failureJsonButton.disabled = cases.length === 0;
  document.querySelector("#failureStatus").textContent = usingStudy ? "Archived study cases" : uploadBatchResults.length ? "Dataset evidence available" : "Awaiting Dataset batch";
  document.querySelector("#failureCaseCount").textContent = cases.length || "-";
  document.querySelector("#failureTableCount").textContent = `${cases.length} row${cases.length === 1 ? "" : "s"}`;
  document.querySelector("#failureDatasetLink").textContent = usingStudy
    ? `${benchmarkResult.config.dataset.evaluatedImages} study images`
    : uploadBatchResults.length ? `${uploadBatchResults.length} images` : "-";
  const highRisk = cases.filter((item) => item.risk >= 4).length;
  document.querySelector("#failureHighRisk").textContent = highRisk || "-";
  const modeCounts = cases.reduce((acc, item) => {
    acc[item.mode] = (acc[item.mode] || 0) + 1;
    return acc;
  }, {});
  const dominant = Object.entries(modeCounts).sort((a, b) => b[1] - a[1])[0];
  document.querySelector("#failureDominantMode").textContent = dominant ? dominant[0] : "-";
  document.querySelector("#failureDominantDetail").textContent = dominant
    ? `${dominant[1]} ${usingStudy ? "archived case" : "sample"}${dominant[1] === 1 ? "" : "s"} in this mode.`
    : "Run Dataset evaluation to classify failures.";
  body.innerHTML = cases.length
    ? cases.map((item, rank) => `
      <tr>
        <td>${rank + 1}</td>
        <td title="${escapeHtml(item.row.imageName)}">${escapeHtml(compactName(item.row.imageName))}</td>
        <td><span class="modePill">${escapeHtml(item.mode)}</span></td>
        <td>${item.risk}</td>
        <td>${item.row.gtCount ?? "-"}</td>
        <td>${item.row.missed ?? "-"}</td>
        <td>${item.row.falsePositive ?? "-"}</td>
        <td>${item.row.predictionCount}</td>
        <td>${item.studyArchive ? "-" : item.row.meanConfidence.toFixed(2)}</td>
        <td>${escapeHtml(item.note)}</td>
        <td><button class="miniButton" type="button" data-failure-view="${item.index}" ${item.row.error || !item.row.resultImage ? "disabled" : ""}>View</button></td>
      </tr>
    `).join("")
    : `<tr><td colspan="11">No flagged failure cases yet. Run Dataset evaluation or use a harder degradation setting.</td></tr>`;
}

function exportFailureJson() {
  const cases = buildFailureCases();
  if (!cases.length) return;
  downloadText("robustness-failure-cases.json", "application/json", JSON.stringify({
    generatedAt: new Date().toISOString(),
    datasetPreview,
    cases: cases.map((item, rank) => ({
      rank: rank + 1,
      mode: item.mode,
      risk: item.risk,
      note: item.note,
      ...stripUploadPreview(item.row),
    })),
  }, null, 2));
}

function renderComparison(data) {
  const transformer = data.transformer.metrics;
  const cnn = data.cnn.metrics;
  const gap = transformer.map - cnn.map;
  document.querySelector("#compareSummary").textContent = `${prettifyDegradation(data.degradation)}, severity ${data.severity}`;
  document.querySelector("#transformerMap").textContent = transformer.map.toFixed(2);
  document.querySelector("#cnnMap").textContent = cnn.map.toFixed(2);
  document.querySelector("#robustnessGap").textContent = `${gap >= 0 ? "+" : ""}${gap.toFixed(2)}`;
  document.querySelector("#transformerFailures").textContent = failureSummary(transformer.failures);
  document.querySelector("#cnnFailures").textContent = failureSummary(cnn.failures);
  document.querySelector("#gapInterpretation").textContent = gap >= 0
    ? "Transformer is stronger under this condition."
    : "CNN baseline is stronger under this condition.";
}

function renderSweep(data) {
  document.querySelector("#sweepSummary").textContent = `${prettifyDegradation(data.degradation)} curve`;
  const width = 560;
  const height = 220;
  const pad = 28;
  const points = (modelKey) => data.rows.map((row, index) => {
    const x = pad + index * ((width - pad * 2) / 5);
    const y = height - pad - row[modelKey].map * (height - pad * 2);
    return `${x},${y}`;
  }).join(" ");
  const tickRows = data.rows.map((row, index) => {
    const x = pad + index * ((width - pad * 2) / 5);
    return `<text x="${x}" y="${height - 7}" text-anchor="middle">${row.severity}</text>`;
  }).join("");
  document.querySelector("#sweepChart").innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="mAP robustness curve">
      <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
      <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />
      <polyline class="line transformerLine" points="${points("transformer")}" />
      <polyline class="line cnnLine" points="${points("cnn")}" />
      ${tickRows}
      <text x="${pad}" y="16">mAP</text>
      <text x="${width - pad}" y="${height - 7}" text-anchor="end">severity</text>
    </svg>
    <div class="legend"><span class="legendTransformer"></span> Transformer <span class="legendCnn"></span> CNN baseline</div>
  `;
}

async function runCurves() {
  setRunning(curvesRunButton, true, "Building");
  try {
    const data = benchmarkResult || await getJson("/api/benchmark", 45000);
    benchmarkResult = data;
    renderDashboard(data);
    renderCurvesDashboard(data);
    curvesJsonButton.disabled = false;
    exportCsvButton.disabled = false;
    exportJsonButton.disabled = false;
    saveLocalState();
  } catch (error) {
    showError(`Curves unavailable: ${error.message}`);
  } finally {
    setRunning(curvesRunButton, false, "Build curves");
  }
}

function renderCurvesDashboard(data) {
  if (!data || !Array.isArray(data.rows) || !data.rows.length) return;
  const grouped = buildCurveGroups(data.rows);
  const summary = summariseCurves(data, grouped);
  document.querySelector("#curvesStatus").textContent = `${Object.keys(grouped).length} degradation curves`;
  document.querySelector("#curvesBestModel").textContent = prettifyModel(data.summary.bestModel);
  document.querySelector("#curvesBestModelDetail").textContent = `Mean mAP: Transformer ${data.summary.averageMap.transformer.toFixed(2)}, CNN ${data.summary.averageMap.cnn.toFixed(2)}.`;
  document.querySelector("#curvesWorstDrop").textContent = `${summary.worstDrop.value.toFixed(2)}`;
  document.querySelector("#curvesWorstDropDetail").textContent = `${prettifyModel(summary.worstDrop.model)}, ${prettifyDegradation(summary.worstDrop.degradation)} from severity 0 to 5.`;
  document.querySelector("#curvesMeanGap").textContent = `${summary.meanGap >= 0 ? "+" : ""}${summary.meanGap.toFixed(2)}`;
  document.querySelector("#curvesRunCount").textContent = data.summary.runCount;
  document.querySelector("#curvesChartGrid").innerHTML = Object.entries(grouped)
    .map(([degradationName, rows]) => renderCurveCard(degradationName, rows))
    .join("");
  document.querySelector("#curvesInterpretationStatus").textContent = "Evidence draft";
  document.querySelector("#curvesNotes").innerHTML = summary.notes.map((item) => `<p>${escapeHtml(item)}</p>`).join("");
  curvesJsonButton.disabled = false;
}

function buildCurveGroups(rows) {
  const groups = {};
  for (const row of rows) {
    const key = row.degradation;
    groups[key] ||= {};
    groups[key][row.severity] ||= { severity: row.severity, transformer: [], cnn: [] };
    groups[key][row.severity][row.model].push(row.map);
  }
  return Object.fromEntries(Object.entries(groups).map(([degradationName, severityRows]) => [
    degradationName,
    Object.values(severityRows).sort((a, b) => a.severity - b.severity).map((row) => ({
      severity: row.severity,
      transformer: average(row.transformer),
      cnn: average(row.cnn),
    })),
  ]));
}

function summariseCurves(data, grouped) {
  const drops = [];
  for (const [degradationName, rows] of Object.entries(grouped)) {
    const start = rows.find((row) => row.severity === 0) || rows[0];
    const end = rows.find((row) => row.severity === 5) || rows[rows.length - 1];
    drops.push({ degradation: degradationName, model: "transformer", value: Math.max(0, start.transformer - end.transformer) });
    drops.push({ degradation: degradationName, model: "cnn", value: Math.max(0, start.cnn - end.cnn) });
  }
  const worstDrop = drops.sort((a, b) => b.value - a.value)[0] || { degradation: "blur", model: "transformer", value: 0 };
  const meanGap = data.summary.averageMap.transformer - data.summary.averageMap.cnn;
  const notes = [
    `${prettifyModel(data.summary.bestModel)} has the stronger average controlled benchmark score across all scenes and degradations.`,
    `${prettifyDegradation(worstDrop.degradation)} creates the steepest observed severity response for the ${prettifyModel(worstDrop.model)} detector.`,
    `The mean mAP gap is ${meanGap >= 0 ? "+" : ""}${meanGap.toFixed(2)}, which can be discussed as the model-family robustness difference.`,
  ];
  if (uploadBatchResults.length) {
    notes.push(`Dataset evidence is linked: ${uploadBatchResults.length} uploaded image${uploadBatchResults.length === 1 ? "" : "s"} can be used beside these controlled curves.`);
  }
  return { worstDrop, meanGap, notes };
}

function renderCurveCard(degradationName, rows) {
  const width = 620;
  const height = 260;
  const pad = 34;
  const xFor = (severity) => pad + Number(severity) * ((width - pad * 2) / 5);
  const yFor = (value) => height - pad - Math.max(0, Math.min(1, value)) * (height - pad * 2);
  const lineFor = (key) => rows.map((row) => `${xFor(row.severity)},${yFor(row[key])}`).join(" ");
  const markers = rows.map((row) => `
    <circle class="point transformerPoint" cx="${xFor(row.severity)}" cy="${yFor(row.transformer)}" r="4" />
    <circle class="point cnnPoint" cx="${xFor(row.severity)}" cy="${yFor(row.cnn)}" r="4" />
  `).join("");
  const ticks = rows.map((row) => `<text x="${xFor(row.severity)}" y="${height - 8}" text-anchor="middle">${row.severity}</text>`).join("");
  const last = rows[rows.length - 1];
  return `
    <article class="curveCard">
      <div class="sectionHeading compactHeading">
        <span>${prettifyDegradation(degradationName)}</span>
        <strong>T ${last.transformer.toFixed(2)} / C ${last.cnn.toFixed(2)}</strong>
      </div>
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="${prettifyDegradation(degradationName)} robustness curve">
        <line x1="${pad}" y1="${height - pad}" x2="${width - pad}" y2="${height - pad}" />
        <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${height - pad}" />
        <polyline class="line transformerLine" points="${lineFor("transformer")}" />
        <polyline class="line cnnLine" points="${lineFor("cnn")}" />
        ${markers}
        ${ticks}
        <text x="${pad}" y="18">mAP</text>
        <text x="${width - pad}" y="${height - 8}" text-anchor="end">severity</text>
      </svg>
      <div class="legend"><span class="legendTransformer"></span> Transformer <span class="legendCnn"></span> CNN baseline</div>
    </article>
  `;
}

function average(values) {
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : 0;
}

function renderDashboard(data) {
  const { summary } = data;
  const transformerMap = summary.averageMap.transformer;
  const cnnMap = summary.averageMap.cnn;
  const worst = summary.worstCase;
  const imageCount = data.config?.dataset?.evaluatedImages;
  document.querySelector("#dashboardSummary").textContent = imageCount ? `${summary.runCount} conditions / ${imageCount} images` : `${summary.runCount} real conditions`;
  document.querySelector("#benchRunCount").textContent = summary.runCount;
  document.querySelector("#bestModel").textContent = prettifyModel(summary.bestModel);
  document.querySelector("#bestModelDetail").textContent = `Mean mAP: Transformer ${transformerMap.toFixed(2)}, CNN ${cnnMap.toFixed(2)}.`;
  document.querySelector("#worstDegradation").textContent = prettifyDegradation(summary.worstDegradation);
  document.querySelector("#worstDegradationDetail").textContent = `Measured mAP drop: ${summary.degradationImpact[summary.worstDegradation].toFixed(2)} from severity 0 to 5.`;
  document.querySelector("#worstCaseMap").textContent = worst.map.toFixed(2);
  document.querySelector("#worstCaseDetail").textContent = `${worst.sceneName}, ${prettifyModel(worst.model)}, ${prettifyDegradation(worst.degradation)}, severity ${worst.severity}.`;
  renderLinkedDatasetEvidence();
  renderReport();
  renderCurvesDashboard(data);
  saveLocalState();
}

function renderReport() {
  const evidence = buildReportEvidence();
  document.querySelector("#reportStatus").textContent = evidence.status;
  document.querySelector("#reportImageCount").textContent = evidence.imageCount || evidence.studyImageCount || "-";
  document.querySelector("#reportDatasetDetail").textContent = evidence.datasetDetail;
  document.querySelector("#reportMeanConfidence").textContent = evidence.imageCount ? evidence.meanConfidence.toFixed(2) : "-";
  document.querySelector("#reportWeakestSample").textContent = evidence.weakest
    ? compactName(evidence.weakest.imageName)
    : evidence.weakestCondition ? `${prettifyDegradation(evidence.weakestCondition.degradation)} S${evidence.weakestCondition.severity}` : "-";
  document.querySelector("#reportWeakestDetail").textContent = evidence.weakest
    ? failureNoteForRow(evidence.weakest)
    : evidence.weakestCondition
      ? `${prettifyModel(evidence.weakestCondition.model)} mAP ${evidence.weakestCondition.map.toFixed(3)} in the synchronized study.`
      : "Lowest-confidence or failed image.";
  document.querySelector("#reportBestModel").textContent = evidence.benchmark ? prettifyModel(evidence.benchmark.summary.bestModel) : "-";
  document.querySelector("#reportBenchmarkDetail").textContent = evidence.benchmark
    ? benchmarkSentence(evidence.benchmark)
    : "Run Benchmark to include controlled robustness evidence.";
  const reportWorstCount = evidence.worstRows.length || evidence.studyFailureCases.length;
  document.querySelector("#reportWorstCount").textContent = `${reportWorstCount} listed`;
  document.querySelector("#reportWorstBody").innerHTML = evidence.worstRows.length
    ? evidence.worstRows.map((row, index) => `
      <tr>
        <td>${index + 1}</td>
        <td title="${escapeHtml(row.imageName)}">${escapeHtml(compactName(row.imageName))}</td>
        <td>${row.predictionCount}</td>
        <td>${row.meanConfidence.toFixed(2)}</td>
        <td>${escapeHtml(failureNoteForRow(row))}</td>
      </tr>
    `).join("")
    : evidence.studyFailureCases.length
      ? evidence.studyFailureCases.map((item, index) => `
        <tr>
          <td>${index + 1}</td>
          <td>${escapeHtml(item.row.imageName)}</td>
          <td>${item.row.predictionCount}</td>
          <td>-</td>
          <td>${escapeHtml(item.note)}</td>
        </tr>
      `).join("")
    : `<tr><td colspan="5">No batch evidence yet.</td></tr>`;
  const bullets = reportBullets(evidence);
  document.querySelector("#reportInterpretationStatus").textContent = evidence.imageCount || evidence.benchmark ? "Evidence draft" : "Draft";
  document.querySelector("#reportBullets").innerHTML = bullets.map((item) => `<p>${escapeHtml(item)}</p>`).join("");
  reportMarkdownText = buildReportMarkdown(evidence, bullets);
  document.querySelector("#reportMarkdown").textContent = reportMarkdownText;
  const reportUnavailable = !evidence.imageCount && !evidence.benchmark;
  exportReportButton.disabled = reportUnavailable;
  printReportButton.disabled = reportUnavailable;
}

function buildReportEvidence() {
  const completedRows = uploadBatchResults.filter((row) => !row.error);
  const failedRows = uploadBatchResults.filter((row) => row.error);
  const labelledRows = uploadBatchResults.filter((row) => row.labelAvailable);
  const imageCount = uploadBatchResults.length;
  const totalPredictions = uploadBatchResults.reduce((sum, row) => sum + row.predictionCount, 0);
  const meanConfidence = imageCount
    ? uploadBatchResults.reduce((sum, row) => sum + row.meanConfidence, 0) / imageCount
    : 0;
  const weakest = uploadBatchResults
    .slice()
    .sort((a, b) => Number(Boolean(b.error)) - Number(Boolean(a.error)) || a.meanConfidence - b.meanConfidence || a.predictionCount - b.predictionCount)[0];
  const worstRows = uploadBatchResults
    .slice()
    .sort((a, b) => Number(Boolean(b.error)) - Number(Boolean(a.error)) || a.meanConfidence - b.meanConfidence || a.predictionCount - b.predictionCount)
    .slice(0, 3);
  const degradationCounts = uploadBatchResults.reduce((acc, row) => {
    acc[row.degradation] = (acc[row.degradation] || 0) + 1;
    return acc;
  }, {});
  const dominantDegradation = Object.entries(degradationCounts).sort((a, b) => b[1] - a[1])[0]?.[0] || uploadDegradation.value;
  const severityValues = uploadBatchResults.map((row) => Number(row.severity)).filter((value) => Number.isFinite(value));
  const meanSeverity = severityValues.length ? severityValues.reduce((sum, value) => sum + value, 0) / severityValues.length : Number(uploadSeverity.value);
  const studyImageCount = Number(benchmarkResult?.config?.dataset?.evaluatedImages || 0);
  const weakestCondition = benchmarkResult?.summary?.worstCase || null;
  const studyFailureCases = imageCount ? [] : buildFailureCases().filter((item) => item.studyArchive).slice(0, 3);
  return {
    imageCount,
    studyImageCount,
    studyTaskCount: studyImageCount * 32,
    completedCount: completedRows.length,
    failedCount: failedRows.length,
    labelledCount: labelledRows.length,
    totalGt: labelledRows.reduce((sum, row) => sum + row.gtCount, 0),
    totalMatched: labelledRows.reduce((sum, row) => sum + row.matched, 0),
    totalMissed: labelledRows.reduce((sum, row) => sum + row.missed, 0),
    totalFalsePositive: labelledRows.reduce((sum, row) => sum + row.falsePositive, 0),
    meanAp50: labelledRows.length ? labelledRows.reduce((sum, row) => sum + (row.ap50 || 0), 0) / labelledRows.length : null,
    meanIou: labelledRows.length ? labelledRows.reduce((sum, row) => sum + (row.meanIou || 0), 0) / labelledRows.length : null,
    totalPredictions,
    meanConfidence,
    weakest,
    weakestCondition,
    worstRows,
    uploadFailureCases: imageCount ? buildFailureCases() : [],
    studyFailureCases,
    dominantDegradation,
    meanSeverity,
    classRows: buildClassAnalysis(),
    benchmark: benchmarkResult,
    transition: transitions.getResult()?.analysis || null,
    status: imageCount || benchmarkResult ? "Evidence available" : "Awaiting evidence",
    datasetDetail: imageCount
      ? `${completedRows.length} completed, ${failedRows.length} failed, ${totalPredictions} total predictions.`
      : studyImageCount
        ? `${studyImageCount} labelled ${benchmarkResult.config.dataset.name} images; ${benchmarkResult.summary.runCount} reported conditions.`
        : "Run Dataset batch evaluation to populate this section.",
  };
}

function restoreLocalState() {
  try {
    const savedHistory = JSON.parse(localStorage.getItem(STORAGE_KEYS.history) || "[]");
    if (Array.isArray(savedHistory)) history = savedHistory.slice(0, 12);
    renderHistory();
    const savedBenchmark = JSON.parse(localStorage.getItem(STORAGE_KEYS.benchmark) || "null");
    if (savedBenchmark && Array.isArray(savedBenchmark.rows) && savedBenchmark.summary && savedBenchmark.config?.evaluator) {
      benchmarkResult = savedBenchmark;
      renderCurvesDashboard(benchmarkResult);
      renderDashboard(savedBenchmark);
      exportCsvButton.disabled = false;
      exportJsonButton.disabled = false;
    }
    const savedUploadBatch = JSON.parse(localStorage.getItem(STORAGE_KEYS.uploadBatch) || "[]");
    if (Array.isArray(savedUploadBatch)) {
      uploadBatchResults = savedUploadBatch;
      renderUploadBatch();
      if (uploadBatchResults.length) {
        document.querySelector("#uploadBatchProgress").textContent = `Restored ${uploadBatchResults.length} previous result${uploadBatchResults.length === 1 ? "" : "s"}`;
        document.querySelector("#uploadAnalysisText").textContent = describeUploadBatch();
      }
    }
    renderReport();
  } catch {
    history = [];
    benchmarkResult = null;
    uploadBatchResults = [];
    renderReport();
  }
}

function saveLocalState() {
  try {
    localStorage.setItem(STORAGE_KEYS.history, JSON.stringify(history));
    if (benchmarkResult) {
      localStorage.setItem(STORAGE_KEYS.benchmark, JSON.stringify(benchmarkResult));
    }
    localStorage.setItem(STORAGE_KEYS.uploadBatch, JSON.stringify(uploadBatchResults.map(stripUploadPreview)));
  } catch {
    // Local storage is optional; the workbench still functions without it.
  }
}

function exportBenchmarkCsv() {
  if (!benchmarkResult) return;
  const headers = ["model", "degradation", "severity", "imageCount", "map", "ap50", "ap75", "ar100", "smallAP", "mediumAP", "largeAP", "smallAR", "mediumAR", "largeAR", "cleanMap", "absoluteDrop", "retention", "inferenceSeconds", "missed", "falsePositive", "classification", "localisation"];
  const lines = [headers.join(",")].concat(benchmarkResult.rows.map((row) => headers.map((key) => csvCell(row[key])).join(",")));
  downloadText("robustness-benchmark.csv", "text/csv", lines.join("\n"));
}

function exportBenchmarkJson() {
  if (!benchmarkResult) return;
  downloadText("robustness-benchmark-report.json", "application/json", JSON.stringify(benchmarkResult, null, 2));
}

function exportCurvesJson() {
  if (!benchmarkResult) return;
  const grouped = buildCurveGroups(benchmarkResult.rows);
  downloadText("robustness-curves.json", "application/json", JSON.stringify({
    generatedAt: new Date().toISOString(),
    curves: grouped,
    summary: summariseCurves(benchmarkResult, grouped),
  }, null, 2));
}

function exportDatasetCsv() {
  if (!uploadBatchResults.length) return;
  const headers = ["imageName", "model", "degradation", "severity", "annotationFormat", "gtCount", "matched", "missed", "falsePositive", "ap50", "meanIou", "predictionCount", "meanConfidence", "small", "medium", "large", "backend", "error"];
  const lines = [headers.join(",")].concat(uploadBatchResults.map((row) => headers.map((key) => csvCell(row[key])).join(",")));
  downloadText("uploaded-dataset-evaluation.csv", "text/csv", lines.join("\n"));
}

function exportDatasetJson() {
  if (!uploadBatchResults.length) return;
  downloadText("uploaded-dataset-evaluation.json", "application/json", JSON.stringify({
    generatedAt: new Date().toISOString(),
    model: uploadModel.value,
    degradation: uploadDegradation.value,
    severity: Number(uploadSeverity.value),
    rows: uploadBatchResults.map(stripUploadPreview),
    summary: {
      imageCount: uploadBatchResults.length,
      totalPredictions: uploadBatchResults.reduce((sum, row) => sum + row.predictionCount, 0),
      meanConfidence: uploadBatchResults.reduce((sum, row) => sum + row.meanConfidence, 0) / uploadBatchResults.length,
    },
  }, null, 2));
}

function exportReportMarkdown() {
  renderReport();
  if (!reportMarkdownText) return;
  downloadText("robustness-research-report.md", "text/markdown", reportMarkdownText);
}

function printReport() {
  renderReport();
  document.body.classList.add("printMode");
  window.print();
  window.setTimeout(() => document.body.classList.remove("printMode"), 700);
}

function stripUploadPreview(row) {
  const { cleanImage, resultImage, ...rest } = row;
  return rest;
}

runButton.addEventListener("click", evaluate);
compareButton.addEventListener("click", compareModels);
sweepButton.addEventListener("click", runSweep);
benchmarkButton.addEventListener("click", runBenchmark);
cancelBenchmarkButton.addEventListener("click", cancelBenchmark);
curvesRunButton.addEventListener("click", runCurves);
curvesJsonButton.addEventListener("click", exportCurvesJson);
exportCsvButton.addEventListener("click", exportBenchmarkCsv);
exportJsonButton.addEventListener("click", exportBenchmarkJson);
datasetCsvButton.addEventListener("click", exportDatasetCsv);
datasetJsonButton.addEventListener("click", exportDatasetJson);
failureJsonButton.addEventListener("click", exportFailureJson);
classJsonButton.addEventListener("click", exportClassJson);
refreshReportButton.addEventListener("click", renderReport);
exportReportButton.addEventListener("click", exportReportMarkdown);
printReportButton.addEventListener("click", printReport);
refreshRunsButton?.addEventListener("click", refreshSavedRuns);
uploadRunButton.addEventListener("click", runUploadEvaluation);
document.querySelector("#uploadBatchBody").addEventListener("click", (event) => {
  const button = event.target.closest("[data-upload-view]");
  if (!button) return;
  renderUploadSelection(Number(button.dataset.uploadView));
});
document.querySelector("#failureCaseBody")?.addEventListener("click", (event) => {
  const button = event.target.closest("[data-failure-view]");
  if (!button) return;
  showPage("dataset");
  renderUploadSelection(Number(button.dataset.failureView));
});
uploadFile.addEventListener("change", async () => {
  if (uploadFile.files && uploadFile.files[0]) {
    if (!isZipFile(uploadFile.files[0])) {
      document.querySelector("#uploadOriginal").src = await readFileAsDataUrl(uploadFile.files[0]);
    } else {
      document.querySelector("#uploadOriginal").removeAttribute("src");
    }
    const selectedFiles = Array.from(uploadFile.files);
    await inspectSelectedDataset(selectedFiles);
    const count = selectedFiles.length;
    const zipCount = selectedFiles.filter(isZipFile).length;
    if (zipCount) studyDatasetSource.value = "selected";
    document.querySelector("#uploadBatchProgress").textContent = `${count} file${count === 1 ? "" : "s"} selected`;
    document.querySelector("#uploadAnalysisText").textContent = zipCount
      ? "Archive preview ready. Run batch evaluation to extract images and apply detector inference."
      : "Images loaded. Run batch evaluation to apply degradation and detector inference.";
  }
});
studyImageCount.addEventListener("change", updateStudyEstimate);
studyDatasetSource.addEventListener("change", updateStudyEstimate);
uploadSeverity.addEventListener("input", () => {
  uploadSeverityBadge.textContent = `Severity ${uploadSeverity.value}`;
  syncDatasetSettingsToControlled();
  renderLinkedDatasetEvidence();
});
for (const input of [uploadModel, uploadDegradation]) {
  input.addEventListener("change", () => {
    syncDatasetSettingsToControlled();
    renderLinkedDatasetEvidence();
    renderReport();
  });
}
for (const tab of navTabs) {
  tab.addEventListener("click", () => showPage(tab.dataset.targetPage));
}
severity.addEventListener("input", () => {
  severityBadge.textContent = `Severity ${severity.value}`;
});
for (const input of [scene, model, degradation]) {
  input.addEventListener("change", () => {
    runtimeLabel.textContent = "Ready";
  });
}

updateStudyEstimate();
init();
