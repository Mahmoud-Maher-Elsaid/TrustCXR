"use strict";

const LABELS = Object.freeze([
  "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
  "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
  "Fibrosis", "Pleural_Thickening", "Hernia",
]);
const GOVERNED_TARGET_LAYER = "features.norm5";

const byId = (id) => document.getElementById(id);
const selectedLabel = byId("attribution-label");
const imageInput = byId("image-input");
const generate = byId("generate-attribution");
let selectedFile = null;
let previewUrl = null;

LABELS.forEach((label) => {
  const option = document.createElement("option");
  option.value = label;
  option.textContent = label.replaceAll("_", " ");
  selectedLabel.append(option);
});

function clearOutput(message) {
  byId("visuals").classList.add("is-hidden");
  byId("metadata").replaceChildren();
  byId("status").textContent = message;
  ["original", "heatmap", "overlay"].forEach((id) => byId(id).removeAttribute("src"));
}

function renderMetadata(data) {
  const metadata = byId("metadata");
  metadata.replaceChildren();
  const rows = [
    ["Selected class", data.label.replaceAll("_", " ")],
    ["Model score", Number(data.model_score).toFixed(4)],
    ["Method", "Grad-CAM"],
    ["Model", data.model],
    ["Target layer", data.target_layer],
    ["Raw attribution", `${data.raw_attribution_dimensions[0]} × ${data.raw_attribution_dimensions[1]}`],
    ["Display map", `${data.display_dimensions[0]} × ${data.display_dimensions[1]}`],
  ];
  rows.forEach(([label, value]) => {
    const term = document.createElement("dt");
    term.textContent = label;
    const description = document.createElement("dd");
    description.textContent = value;
    metadata.append(term, description);
  });
}

async function requestAttribution() {
  if (!selectedFile) return;
  generate.disabled = true;
  clearOutput("Generating one bounded research attribution…");
  try {
    const response = await fetch("/research/explainability/gradcam", {
      method: "POST",
      headers: { "Content-Type": selectedFile.type, "X-TrustCXR-Attribution-Label": selectedLabel.value },
      body: selectedFile,
      credentials: "omit",
      cache: "no-store",
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.reason_code === "ATTRIBUTION_UNAVAILABLE"
        ? "No non-degenerate Grad-CAM attribution was available for this class/input pair. This does not imply absence of pathology."
        : "The research attribution could not be generated.");
    }
    if (result.target_layer !== GOVERNED_TARGET_LAYER) {
      throw new Error("The research attribution returned an ungoverned target layer.");
    }
    byId("original").src = previewUrl;
    byId("heatmap").src = `data:image/png;base64,${result.heatmap_png_base64}`;
    byId("overlay").src = `data:image/png;base64,${result.overlay_png_base64}`;
    byId("visuals").classList.remove("is-hidden");
    byId("status").textContent = "Attribution generated for the selected model class.";
    renderMetadata(result);
  } catch (error) {
    clearOutput(error.message || "The research attribution could not be generated.");
  } finally {
    generate.disabled = !selectedFile;
  }
}

imageInput.addEventListener("change", () => {
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  selectedFile = imageInput.files[0] || null;
  previewUrl = selectedFile ? URL.createObjectURL(selectedFile) : null;
  generate.disabled = !selectedFile;
  clearOutput(selectedFile ? "Ready. Select a model finding and generate one attribution." : "Choose a local image to begin.");
});
generate.addEventListener("click", requestAttribution);
window.addEventListener("beforeunload", () => { if (previewUrl) URL.revokeObjectURL(previewUrl); });
