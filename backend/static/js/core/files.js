export function readFileAsDataUrl(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(new Error("Could not read selected file"));
    reader.readAsDataURL(file);
  });
}


export function isZipFile(file) {
  const name = (file.name || "").toLowerCase();
  return name.endsWith(".zip")
    || file.type === "application/zip"
    || file.type === "application/x-zip-compressed";
}


export function delay(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}
