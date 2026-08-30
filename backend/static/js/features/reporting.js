import {
  formatMetric,
  formatPercent,
  formatSeverity,
  prettifyDegradation,
  prettifyModel,
} from "../core/format.js";


export function classRiskScore(item) {
  const missRate = item.gt ? item.missed / item.gt : 0;
  const falsePositivePenalty = Math.min(1, item.falsePositive / Math.max(1, item.gt || item.matched || 1));
  const iouPenalty = item.meanIou == null ? 0.15 : Math.max(0, 0.65 - item.meanIou);
  return missRate * 5 + falsePositivePenalty * 2 + iouPenalty;
}


export function classRiskNote(item) {
  if (!item.gt && item.falsePositive) return "Only unmatched predictions; inspect as false-positive class.";
  if (item.gt && item.matched === 0) return "No GT object matched at AP50 threshold.";
  if (item.gt && item.missed / item.gt >= 0.5) return "High missed-detection rate under this condition.";
  if (item.falsePositive > item.matched) return "False positives exceed matched detections.";
  if (item.meanIou != null && item.meanIou < 0.55) return "Localisation is weak even when detections match.";
  return "Class behaviour is currently stable.";
}


export function failureNoteForRow(row) {
  if (row.error) return `Failed: ${row.error}`;
  if (row.labelAvailable && row.missed > 0) {
    return `${row.missed} missed GT object${row.missed === 1 ? "" : "s"}, AP50 ${formatMetric(row.ap50)}.`;
  }
  if (row.predictionCount === 0) return "No predictions after degradation.";
  if (row.meanConfidence < 0.35) return "Very low confidence.";
  if (row.meanConfidence < 0.5) return "Low confidence.";
  return "Usable prediction confidence.";
}


export function benchmarkSentence(data) {
  const { summary } = data;
  const gap = summary.averageMap.transformer - summary.averageMap.cnn;
  const winner = prettifyModel(summary.bestModel);
  return `${winner} is the stronger model in the real dataset study; average mAP gap is ${gap >= 0 ? "+" : ""}${gap.toFixed(2)}, and the largest degradation impact is ${prettifyDegradation(summary.worstDegradation)}.`;
}


function degradationInterpretation(degradationName, meanSeverity) {
  const severityText = meanSeverity >= 4 ? "high" : meanSeverity >= 2 ? "moderate" : "low";
  if (degradationName === "blur") {
    return `Motion blur at ${severityText} severity is expected to affect localisation and small-object boundaries first.`;
  }
  if (degradationName === "lowlight") {
    return `Low illumination at ${severityText} severity is expected to reduce confidence and increase missed detections.`;
  }
  return `JPEG artefacts at ${severityText} severity are expected to disturb texture cues and can create false positives around compressed edges.`;
}


export function reportBullets(evidence) {
  const bullets = [];
  if (evidence.imageCount) {
    bullets.push(`The uploaded dataset batch contains ${evidence.imageCount} images with ${evidence.totalPredictions} total predictions and mean confidence ${evidence.meanConfidence.toFixed(2)}.`);
    if (evidence.labelledCount) {
      bullets.push(`${evidence.labelledCount} images include ground-truth labels: ${evidence.totalGt} GT objects, ${evidence.totalMatched} matched detections, ${evidence.totalMissed} missed objects, AP50 ${formatMetric(evidence.meanAp50)}, mean IoU ${formatMetric(evidence.meanIou)}.`);
    }
    if (evidence.failedCount) {
      bullets.push(`${evidence.failedCount} image${evidence.failedCount === 1 ? "" : "s"} failed during processing and remains in the evidence record.`);
    }
    bullets.push(
      evidence.meanConfidence < 0.45
        ? "The batch has low mean confidence under the selected degradation setting."
        : "The batch retains enough prediction confidence for qualitative failure inspection.",
    );
    if (evidence.uploadFailureCases.length) {
      const topFailure = evidence.uploadFailureCases[0];
      bullets.push(`Failure-case mining flagged ${evidence.uploadFailureCases.length} sample${evidence.uploadFailureCases.length === 1 ? "" : "s"}; the highest-risk mode is ${topFailure.mode} on ${topFailure.row.imageName}.`);
    }
    if (evidence.classRows.length) {
      const weakestClass = evidence.classRows.find((row) => row.gt > 0) || evidence.classRows[0];
      bullets.push(`Class-level aggregation identifies ${weakestClass.label} as the weakest class: ${classRiskNote(weakestClass)}`);
    }
    bullets.push(degradationInterpretation(evidence.dominantDegradation, evidence.meanSeverity));
  } else if (evidence.studyImageCount) {
    bullets.push(`The synchronized main study contains ${evidence.studyImageCount} labelled images, ${evidence.benchmark.summary.runCount} reported conditions, and ${evidence.studyTaskCount.toLocaleString()} inference tasks.`);
    if (evidence.studyFailureCases.length) {
      bullets.push(`${evidence.studyFailureCases.length} archived failure examples are available for qualitative inspection.`);
    }
  } else {
    bullets.push("Dataset evidence is not available yet; run a batch using real images or the COCO128 zip to generate image-level findings.");
  }
  if (evidence.benchmark) {
    bullets.push(benchmarkSentence(evidence.benchmark));
    bullets.push(`The lowest real-image condition is ${prettifyModel(evidence.benchmark.summary.worstCase.model)} under ${prettifyDegradation(evidence.benchmark.summary.worstCase.degradation)} severity ${evidence.benchmark.summary.worstCase.severity}.`);
  } else {
    bullets.push("Real dataset study evidence is not available yet; complete a Benchmark study before using this as a final dissertation result.");
  }
  if (evidence.transition) {
    const stableRate = evidence.transition.firstFailure.neverCount / Math.max(1, evidence.transition.cleanCorrectCount);
    bullets.push(`Object-level tracking followed ${evidence.transition.objectCount.toLocaleString()} annotated objects under ${prettifyDegradation(evidence.transition.degradation)}; ${formatPercent(stableRate)} of the clean-correct cohort remained correct through severity 5.`);
  }
  return bullets;
}


export function buildReportMarkdown(evidence, bullets) {
  const lines = [
    "# Detection Robustness Report",
    "",
    `Generated: ${new Date().toLocaleString()}`,
    "",
    "## Dataset Batch Summary",
    evidence.imageCount
      ? `- Images evaluated: ${evidence.imageCount}`
      : evidence.studyImageCount
        ? `- Synchronized study images: ${evidence.studyImageCount}`
        : "- Images evaluated: no dataset batch evidence yet",
  ];
  if (evidence.imageCount) {
    lines.push(`- Completed: ${evidence.completedCount}`);
    lines.push(`- Failed: ${evidence.failedCount}`);
    lines.push(`- Total predictions: ${evidence.totalPredictions}`);
    lines.push(`- Mean confidence: ${evidence.meanConfidence.toFixed(2)}`);
    if (evidence.labelledCount) {
      lines.push(`- Labelled images: ${evidence.labelledCount}`);
      lines.push(`- Ground-truth objects: ${evidence.totalGt}`);
      lines.push(`- Matched detections: ${evidence.totalMatched}`);
      lines.push(`- Missed objects: ${evidence.totalMissed}`);
      lines.push(`- False positives: ${evidence.totalFalsePositive}`);
      lines.push(`- Mean AP50: ${formatMetric(evidence.meanAp50)}`);
      lines.push(`- Mean IoU: ${formatMetric(evidence.meanIou)}`);
    }
    lines.push(`- Weakest sample: ${evidence.weakest.imageName} (${failureNoteForRow(evidence.weakest)})`);
  }
  lines.push("", "## Class-Level Analysis");
  if (evidence.classRows.length) {
    for (const row of evidence.classRows.slice(0, 8)) {
      lines.push(`- ${row.label}: GT ${row.gt}, matched ${row.matched}, missed ${row.missed}, false positives ${row.falsePositive}, AP50 ${formatMetric(row.ap50)}, mean IoU ${formatMetric(row.meanIou)}. ${classRiskNote(row)}`);
    }
  } else {
    lines.push("- No class-level evidence available yet; upload a labelled YOLO or COCO zip batch.");
  }
  lines.push("", "## Object-Level Failure Transitions");
  if (evidence.transition) {
    const transition = evidence.transition;
    const stableRate = transition.firstFailure.neverCount / Math.max(1, transition.cleanCorrectCount);
    lines.push(`- Model: ${prettifyModel(transition.model)}`);
    lines.push(`- Degradation: ${prettifyDegradation(transition.degradation)}`);
    lines.push(`- Objects tracked: ${transition.objectCount}`);
    lines.push(`- Clean-correct cohort: ${transition.cleanCorrectCount}`);
    lines.push(`- Median first failure among failed objects: severity ${formatSeverity(transition.firstFailure.medianSeverityAmongFailures)}`);
    lines.push(`- Stable through severity 5: ${transition.firstFailure.neverCount} (${formatPercent(stableRate)})`);
  } else {
    lines.push("- Object-level transition evidence is unavailable.");
  }
  lines.push("", "## Worst-Case Samples");
  if (evidence.worstRows.length) {
    for (const [index, row] of evidence.worstRows.entries()) {
      lines.push(`${index + 1}. ${row.imageName}: ${row.predictionCount} predictions, mean confidence ${row.meanConfidence.toFixed(2)}. ${failureNoteForRow(row)}`);
    }
  } else if (evidence.studyFailureCases.length) {
    for (const [index, item] of evidence.studyFailureCases.entries()) {
      lines.push(`${index + 1}. ${item.row.imageName}: ${item.note}`);
    }
  } else {
    lines.push("- No worst-case samples available yet.");
  }
  lines.push("", "## Interpretation");
  for (const item of bullets) lines.push(`- ${item}`);
  lines.push("", "## Evidence Status");
  lines.push(`- Dataset batch: ${evidence.imageCount ? "available" : "missing"}`);
  lines.push(`- Synchronized main study: ${evidence.studyImageCount ? "available" : "missing"}`);
  lines.push(`- Controlled benchmark: ${evidence.benchmark ? "available" : "missing"}`);
  lines.push(`- Object transitions: ${evidence.transition ? "available" : "missing"}`);
  return lines.join("\n");
}
