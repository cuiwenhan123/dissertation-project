import { loadViews } from "./views.js";


async function bootstrap() {
  const host = document.querySelector("#pageHost");
  if (!host) throw new Error("Page host is missing");
  await loadViews(host);
  await import("./workbench.js");
}


bootstrap().catch((error) => {
  const host = document.querySelector("#pageHost");
  if (host) {
    host.innerHTML = `<section class="page active"><p class="emptyState">${error.message}</p></section>`;
  }
  console.error(error);
});
