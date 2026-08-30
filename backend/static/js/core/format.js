export function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}


export function formatSeverity(value) {
  return Number.isInteger(Number(value)) ? String(value) : Number(value).toFixed(1);
}


export function capitalise(value) {
  const text = String(value || "");
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : text;
}


export function formatMetric(value) {
  return value == null || Number.isNaN(Number(value)) ? "-" : Number(value).toFixed(2);
}


export function csvCell(value) {
  const text = String(value ?? "");
  return text.includes(",") || text.includes('"') ? `"${text.replaceAll('"', '""')}"` : text;
}


export function compactName(value) {
  const text = String(value || "");
  return text.length > 28 ? `${text.slice(0, 12)}...${text.slice(-12)}` : text;
}


export function escapeHtml(value) {
  return String(value || "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}


export function setText(selector, value) {
  const target = document.querySelector(selector);
  if (target) target.textContent = value;
}


export function failureSummary(failures) {
  return `${failures.missed} missed, ${failures.falsePositive} false positive, ${failures.classification} class errors`;
}


export function prettifyModel(value) {
  return value === "cnn" ? "CNN baseline" : "Transformer";
}


export function prettifyDegradation(value) {
  return {
    blur: "motion blur",
    lowlight: "low illumination",
    jpeg: "JPEG artefacts",
  }[value] || value;
}


export function prettifyBackend(value) {
  return {
    "torchvision-fasterrcnn": "Faster R-CNN / torchvision",
    "transformers-detr-resnet-50": "DETR ResNet-50 / transformers",
    "fallback-detector": "Demonstration fallback",
    "demonstration-sweep-model": "Controlled sweep estimate",
  }[value] || value;
}
