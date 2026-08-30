import { delay } from "./files.js";


function parseResponse(request, resolve, reject) {
  try {
    const data = JSON.parse(request.responseText);
    if (request.status < 200 || request.status >= 300) {
      reject(new Error(data.error || `${request.status} ${request.statusText}`));
      return;
    }
    resolve(data);
  } catch {
    reject(new Error("Invalid JSON response"));
  }
}


function requestJson(url, timeoutMs) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("GET", `${url}${url.includes("?") ? "&" : "?"}_=${Date.now()}`, true);
    request.timeout = timeoutMs;
    request.setRequestHeader("Accept", "application/json");
    request.onload = () => parseResponse(request, resolve, reject);
    request.onerror = () => reject(new Error("Local server request failed"));
    request.ontimeout = () => reject(new Error("Local server timed out"));
    request.send();
  });
}


function requestJsonPost(url, payload, timeoutMs) {
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open("POST", url, true);
    request.timeout = timeoutMs;
    request.setRequestHeader("Accept", "application/json");
    request.setRequestHeader("Content-Type", "application/json");
    request.onload = () => parseResponse(request, resolve, reject);
    request.onerror = () => reject(new Error("Local server request failed"));
    request.ontimeout = () => reject(new Error("Local server timed out"));
    request.send(JSON.stringify(payload));
  });
}


async function withRetry(operation, onRetry) {
  let lastError;
  for (let attempt = 1; attempt <= 3; attempt += 1) {
    try {
      return await operation();
    } catch (error) {
      lastError = error;
      if (attempt < 3) {
        onRetry?.(`Retrying local request (${attempt + 1}/3)`);
        await delay(700 * attempt);
      }
    }
  }
  throw lastError;
}


export function createApiClient(onRetry) {
  return {
    getJson: (url, timeoutMs) => withRetry(() => requestJson(url, timeoutMs), onRetry),
    postJson: (url, payload, timeoutMs) => (
      withRetry(() => requestJsonPost(url, payload, timeoutMs), onRetry)
    ),
    requestJson,
  };
}
