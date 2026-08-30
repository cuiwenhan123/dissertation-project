import { downloadText } from "../core/download.js";
import {
  capitalise,
  compactName,
  escapeHtml,
  formatPercent,
  formatSeverity,
  prettifyDegradation,
  prettifyModel,
  setText,
} from "../core/format.js";


export function createTransitionFeature({ getJson, onError, onUpdated }) {
  const modelInput = document.querySelector("#transitionModel");
  const degradationInput = document.querySelector("#transitionDegradation");
  const stepInput = document.querySelector("#transitionStep");
  const exportButton = document.querySelector("#transitionJson");
  let result = null;
  let sequence = 0;

  async function load(silent = false) {
    const requestSequence = ++sequence;
    setText("#transitionStatus", "Loading frozen study");
    exportButton.disabled = true;
    const params = new URLSearchParams({
      model: modelInput.value,
      degradation: degradationInput.value,
    });
    try {
      const data = await getJson(`/api/transitions?${params}`, 30000);
      if (requestSequence !== sequence) return;
      result = data;
      render(data);
      exportButton.disabled = false;
      onUpdated?.(data);
    } catch (error) {
      if (requestSequence !== sequence) return;
      result = null;
      setText("#transitionStatus", "Analysis unavailable");
      if (!silent) onError?.(`Transition analysis unavailable: ${error.message}`);
    }
  }

  function render(data) {
    const analysis = data.analysis;
    const cleanCorrect = Number(analysis.cleanCorrectCount || 0);
    const stable = Number(analysis.firstFailure?.neverCount || 0);
    const medianSeverity = analysis.firstFailure?.medianSeverityAmongFailures;
    setText("#transitionStatus", `${data.studyId} / ${prettifyModel(analysis.model)} / ${prettifyDegradation(analysis.degradation)}`);
    setText("#transitionObjectCount", Number(analysis.objectCount || data.objectCount).toLocaleString());
    setText("#transitionCleanCorrect", cleanCorrect.toLocaleString());
    setText("#transitionMedianFailure", medianSeverity == null ? "No failures" : `Severity ${formatSeverity(medianSeverity)}`);
    setText("#transitionStableRate", formatPercent(stable / Math.max(1, cleanCorrect)));
    setText("#transitionFailureTotal", `${analysis.firstFailure.failedCount.toLocaleString()} first failures`);
    setText("#transitionExampleCount", `${analysis.examples.length} ranked trajectories`);
    renderSurvival(analysis);
    renderFirstFailureBars(analysis);
    renderMatrix(analysis);
    renderSizes(analysis);
    renderExamples(analysis);
  }

  function renderSurvival(analysis) {
    const rows = analysis.cleanToSeverity;
    const width = 720;
    const height = 300;
    const left = 52;
    const right = 22;
    const top = 24;
    const bottom = 42;
    const xFor = (severityValue) => left + Number(severityValue) * ((width - left - right) / 5);
    const yFor = (value) => top + (1 - Number(value)) * (height - top - bottom);
    const points = (key) => rows.map((row) => `${xFor(row.severity)},${yFor(row[key])}`).join(" ");
    const grid = [0, 0.25, 0.5, 0.75, 1].map((value) => `
      <line x1="${left}" y1="${yFor(value)}" x2="${width - right}" y2="${yFor(value)}" />
      <text x="${left - 10}" y="${yFor(value) + 4}" text-anchor="end">${Math.round(value * 100)}%</text>
    `).join("");
    const ticks = rows.map((row) => `<text x="${xFor(row.severity)}" y="${height - 12}" text-anchor="middle">S${row.severity}</text>`).join("");
    const markers = rows.map((row) => `
      <circle class="retentionPoint" cx="${xFor(row.severity)}" cy="${yFor(row.correctRetention)}" r="4" />
      <circle class="survivalPoint" cx="${xFor(row.severity)}" cy="${yFor(row.uninterruptedSurvival)}" r="4" />
    `).join("");
    document.querySelector("#transitionSurvivalChart").innerHTML = `
      <svg viewBox="0 0 ${width} ${height}" role="img" aria-label="Object-level detection survival from severity zero to five">
        ${grid}
        <polyline class="transitionLine retentionLine" points="${points("correctRetention")}" />
        <polyline class="transitionLine survivalLine" points="${points("uninterruptedSurvival")}" />
        ${markers}
        ${ticks}
      </svg>
      <div class="transitionLegend">
        <span><i class="retentionSwatch"></i>Correct at severity</span>
        <span><i class="survivalSwatch"></i>Never failed up to severity</span>
      </div>
    `;
    const last = rows[rows.length - 1];
    setText("#transitionSurvivalLabel", `S5 correct ${formatPercent(last.correctRetention)} / uninterrupted ${formatPercent(last.uninterruptedSurvival)}`);
  }

  function renderFirstFailureBars(analysis) {
    const counts = analysis.firstFailure.counts;
    const denominator = Math.max(1, analysis.cleanCorrectCount);
    const entries = [
      ...[1, 2, 3, 4, 5].map((level) => ({ key: String(level), label: `Severity ${level}` })),
      { key: "never", label: "Never failed" },
    ];
    const maximum = Math.max(...entries.map((entry) => Number(counts[entry.key] || 0)), 1);
    document.querySelector("#transitionFailureBars").innerHTML = entries.map((entry) => {
      const count = Number(counts[entry.key] || 0);
      return `
        <div class="failureBarRow ${entry.key === "never" ? "stableBar" : ""}">
          <span>${entry.label}</span>
          <div class="failureBarTrack"><i style="width:${(count / maximum * 100).toFixed(1)}%"></i></div>
          <strong>${count.toLocaleString()} <small>${formatPercent(count / denominator)}</small></strong>
        </div>
      `;
    }).join("");
  }

  function renderMatrix(analysis) {
    const step = analysis.steps[Number(stepInput.value)] || analysis.steps[0];
    const statuses = ["correct", "localisation", "classification", "missed"];
    const body = document.querySelector("#transitionMatrixBody");
    body.innerHTML = statuses.map((source) => {
      const row = step.transitions.filter((item) => item.from === source);
      return `
        <tr>
          <th>${statusLabel(source)}</th>
          ${statuses.map((target) => {
            const item = row.find((value) => value.to === target) || { count: 0, rate: 0 };
            const colour = target === "correct" ? "23,107,99" : source === "correct" ? "170,82,63" : "56,95,136";
            const alpha = 0.04 + Number(item.rate || 0) * 0.42;
            return `<td style="background:rgba(${colour},${alpha.toFixed(3)})"><strong>${Number(item.count).toLocaleString()}</strong><span>${formatPercent(item.rate)}</span></td>`;
          }).join("")}
        </tr>
      `;
    }).join("");
  }

  function renderSizes(analysis) {
    document.querySelector("#transitionSizeBody").innerHTML = analysis.bySize.map((row) => {
      const earlyFailure = Number(row.firstFailure["1"] || 0) + Number(row.firstFailure["2"] || 0);
      return `
        <tr>
          <td><strong>${escapeHtml(capitalise(row.size))}</strong></td>
          <td>${Number(row.objects).toLocaleString()}</td>
          <td>${Number(row.cleanCorrect).toLocaleString()}</td>
          <td>${earlyFailure.toLocaleString()} <span class="tableSubvalue">${formatPercent(earlyFailure / Math.max(1, row.cleanCorrect))}</span></td>
          <td>${Number(row.firstFailure.never).toLocaleString()} <span class="tableSubvalue">${formatPercent(row.firstFailure.never / Math.max(1, row.cleanCorrect))}</span></td>
          <td>${formatPercent(row.severity5Retention)}</td>
        </tr>
      `;
    }).join("");
  }

  function renderExamples(analysis) {
    const examples = analysis.examples.slice(0, 10);
    document.querySelector("#transitionExamplesBody").innerHTML = examples.length
      ? examples.map((item) => `
        <tr>
          <td title="${escapeHtml(item.imageName)}">${escapeHtml(compactName(item.imageName))}</td>
          <td>${escapeHtml(item.label)}</td>
          <td>${escapeHtml(capitalise(item.size))}</td>
          <td>S${item.firstFailureSeverity}</td>
          ${item.states.map((stateValue) => `<td>${statePill(stateValue)}</td>`).join("")}
          <td>${item.recovered ? "Yes" : "No"}</td>
        </tr>
      `).join("")
      : `<tr><td colspan="11">No clean-correct object failed under this condition.</td></tr>`;
  }

  function statePill(stateValue) {
    const statusValue = stateValue.status || "missed";
    const details = [
      statusLabel(statusValue),
      stateValue.predictedLabel ? `predicted ${stateValue.predictedLabel}` : "",
      stateValue.iou ? `IoU ${Number(stateValue.iou).toFixed(2)}` : "",
      stateValue.score != null ? `score ${Number(stateValue.score).toFixed(2)}` : "",
    ].filter(Boolean).join(", ");
    return `<span class="statePill state-${statusValue}" title="${escapeHtml(details)}">${statusShort(statusValue)}</span>`;
  }

  function statusLabel(value) {
    return {
      correct: "Correct",
      localisation: "Localisation",
      classification: "Classification",
      missed: "Missed",
    }[value] || value;
  }

  function statusShort(value) {
    return {
      correct: "OK",
      localisation: "LOC",
      classification: "CLS",
      missed: "MISS",
    }[value] || value;
  }

  function exportJson() {
    if (!result) return;
    const selection = result.selection;
    downloadText(
      `object-transitions-${selection.model}-${selection.degradation}.json`,
      "application/json",
      JSON.stringify(result, null, 2),
    );
  }

  modelInput.addEventListener("change", () => load());
  degradationInput.addEventListener("change", () => load());
  stepInput.addEventListener("change", () => {
    if (result) renderMatrix(result.analysis);
  });
  exportButton.addEventListener("click", exportJson);

  return {
    getResult: () => result,
    load,
  };
}
