export const VIEW_NAMES = [
  "overview",
  "dataset",
  "classAnalysis",
  "failures",
  "transitions",
  "comparison",
  "curves",
  "benchmark",
  "report",
  "log",
  "methodology",
];


export async function loadViews(host) {
  const responses = await Promise.all(
    VIEW_NAMES.map(async (name) => {
      const response = await fetch(`/views/${name}.html`, { cache: "no-store" });
      if (!response.ok) {
        throw new Error(`Could not load ${name} view (${response.status})`);
      }
      return response.text();
    }),
  );
  host.innerHTML = responses.join("\n");
}
