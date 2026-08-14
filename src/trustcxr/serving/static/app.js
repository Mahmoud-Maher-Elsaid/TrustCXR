"use strict";

const byId = (id) => document.getElementById(id);

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addDataRow(container, label, value) {
  const row = element("div", "data-row");
  row.append(element("span", "", label), element("strong", "", String(value)));
  container.append(row);
}

const REVIEW_PRESENTATION = Object.freeze({
  DEFER: "Expert Review Required",
  REVISE_DETERMINISTICALLY: "Draft Requires Revision",
  ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW: "Ready for Expert Review",
});

const VERIFIER_PRESENTATION = Object.freeze({
  VERIFIED: "Verified",
  PARTIALLY_VERIFIED: "Partially verified",
  UNVERIFIED: "Not verified",
  CONTRADICTED: "Contradicted",
  NOT_APPLICABLE: "Not applicable",
  WITHHELD_INSUFFICIENT_EVIDENCE: "Evidence unavailable",
});

const ATTRIBUTION_LABELS = Object.freeze([
  "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
  "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema",
  "Fibrosis", "Pleural_Thickening", "Hernia",
]);

function displayLabel(label) {
  return label.replaceAll("_", " ");
}

function uncertaintyPresentation(value) {
  if (!Number.isFinite(value)) return { label: "Unavailable", value: null };
  // Presentation-only thirds of the maximum Bernoulli entropy (ln 2).
  const fraction = Math.max(0, Math.min(1, value / Math.log(2)));
  if (fraction < 1 / 3) return { label: "Low", value };
  if (fraction < 2 / 3) return { label: "Moderate", value };
  return { label: "High", value };
}

function uncertaintyExplanation(category) {
  const explanations = {
    High: "Predictive uncertainty is high. Model outputs should be interpreted cautiously and require expert review.",
    Moderate: "Predictive uncertainty is moderate. Model outputs contain meaningful uncertainty and require expert interpretation.",
    Low: "Predictive uncertainty is relatively low for this model, but expert review is still required.",
    Unavailable: "Predictive uncertainty is unavailable for this review. Expert review is required.",
  };
  return explanations[category];
}

function qualityPresentation(quality) {
  if (!quality) return "Quality warning";
  return quality.status === "PASS" ? "Acceptable" : "Quality warning";
}

function buildPresentation(data) {
  const classifierScores = data.classifier_scores.map((item, index) => ({
    label: item.label,
    score: Number(item.score),
    sourceIndex: index,
  }));
  const sortedScores = classifierScores.toSorted(
    (left, right) => right.score - left.score || left.sourceIndex - right.sourceIndex,
  );
  const rawDecision = data.decisions.actual || data.decisions.precedence[0];
  const verificationCounts = {};
  data.verifier_statuses.forEach((item) => {
    verificationCounts[item.status] = (verificationCounts[item.status] || 0) + 1;
  });
  return {
    view: data.view.selected,
    quality: qualityPresentation(data.technical_quality),
    uncertainty: uncertaintyPresentation(data.reliability.predictive_uncertainty),
    rawDecision,
    reviewStatus: REVIEW_PRESENTATION[rawDecision] || "Expert Review Required",
    classifierScores: sortedScores,
    topSignals: sortedScores.slice(0, 3),
    verificationCounts,
  };
}

function renderViewer(image, format) {
  const viewer = byId("synthetic-viewer");
  viewer.src = image.data_url;
  viewer.alt = "Synthetic non-patient grayscale validation fixture";
  viewer.width = image.width;
  viewer.height = image.height;
  byId("image-dimensions").textContent = `${image.width} × ${image.height} px`;
  byId("viewer-format").textContent = format;
}

function renderScores(presentation) {
  const top = byId("top-signals-content");
  top.replaceChildren();
  presentation.topSignals.forEach((item) => {
    const signal = element("div", "top-signal");
    signal.append(element("strong", "", displayLabel(item.label)), element("span", "", item.score.toFixed(3)));
    top.append(signal);
  });

  const scores = byId("classifier-content");
  scores.replaceChildren();
  presentation.classifierScores.forEach((item) => {
    const row = element("div", "score-row");
    const track = element("div", "score-track");
    track.setAttribute("aria-label", `${displayLabel(item.label)} model score ${item.score.toFixed(4)}`);
    const fill = element("div", "score-fill");
    fill.style.setProperty("--score", `${Math.max(0, Math.min(1, item.score)) * 100}%`);
    track.append(fill);
    row.append(
      element("span", "score-label", displayLabel(item.label)),
      track,
      element("span", "score-value", item.score.toFixed(4)),
    );
    scores.append(row);
  });
}

function renderInterpretation(presentation) {
  const primary = presentation.topSignals[0];
  byId("interpretation-primary-label").textContent = displayLabel(primary.label);
  byId("interpretation-primary-score").textContent = primary.score.toFixed(3);
  const secondary = byId("interpretation-secondary");
  secondary.replaceChildren();
  presentation.topSignals.slice(1).forEach((item) => {
    const row = element("div", "secondary-signal");
    row.append(element("strong", "", displayLabel(item.label)), element("span", "", item.score.toFixed(3)));
    secondary.append(row);
  });
}

function renderSummary(presentation) {
  const summary = byId("summary-content");
  summary.replaceChildren();
  const primary = presentation.topSignals[0];
  const second = presentation.topSignals[1];
  const third = presentation.topSignals[2];
  const qualitySentence = presentation.quality === "Acceptable"
    ? `The uploaded chest X-ray was identified as a ${presentation.view} view and passed the technical-quality check for research analysis.`
    : `The uploaded chest X-ray was identified as a ${presentation.view} view and received a technical-quality warning for research analysis.`;
  const signalSentence = `Among the model outputs, ${displayLabel(primary.label)} produced the highest signal (${primary.score.toFixed(3)}). ${displayLabel(second.label)} (${second.score.toFixed(3)}) and ${displayLabel(third.label)} (${third.score.toFixed(3)}) were the next highest model signals.`;
  [
    qualitySentence,
    signalSentence,
    uncertaintyExplanation(presentation.uncertainty.label),
    "Reliable lesion-localization evidence is not available for this image, so the system cannot identify where a suspected finding is located.",
  ].forEach((text) => summary.append(element("p", "explanation-paragraph", text)));
  const disposition = element("p", "explanation-disposition");
  disposition.append(element("strong", "", "Overall disposition: "), document.createTextNode(`${presentation.reviewStatus}.`));
  summary.append(disposition, element("p", "summary-disclaimer", "These outputs are research model signals and do not constitute a medical diagnosis."));
}

function renderVerification(data, presentation) {
  const summary = byId("verification-summary");
  summary.replaceChildren();
  const statuses = Object.entries(presentation.verificationCounts);
  const allUnavailable = statuses.length === 1 && statuses[0][0] === "WITHHELD_INSUFFICIENT_EVIDENCE";
  if (allUnavailable) {
    summary.append(element("p", "verification-limited", "Evidence verification is limited for this local image review."));
    return;
  }
  const grid = element("div", "verification-counts");
  statuses.forEach(([status, count]) => {
    const item = element("div", "verification-count");
    item.append(element("strong", "", String(count)), element("span", "", VERIFIER_PRESENTATION[status] || status));
    grid.append(item);
  });
  summary.append(grid);
}

function renderTechnicalDetails(data) {
  const reliability = byId("reliability-content");
  reliability.replaceChildren();
  addDataRow(reliability, "Internal status", "PREDICTIVE UNCERTAINTY");
  addDataRow(reliability, "Numeric value", Number.isFinite(data.reliability.predictive_uncertainty) ? data.reliability.predictive_uncertainty.toFixed(6) : "WITHHELD");
  addDataRow(reliability, "Calibration", data.reliability.calibration_label);
  addDataRow(reliability, "OOD capability", data.reliability.ood);
  addDataRow(reliability, "Stage 13 selective prediction", data.reliability.stage13_selective_prediction);
  addDataRow(reliability, "Presentation rule", "Low < 1/3, Moderate < 2/3, High ≥ 2/3 of ln(2)");

  const fusion = byId("fusion-content");
  fusion.replaceChildren();
  addDataRow(fusion, "Status", data.fusion.status);
  addDataRow(fusion, "Evidence", data.fusion.label);
  addDataRow(fusion, "Boundary", "No reliable localization, laterality, or absence-based contradiction");
  fusion.append(
    element("p", "visually-hidden", "Stage 8 WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION"),
    element("p", "visually-hidden", "Stage 10 WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION"),
  );

  const decision = byId("decision-content");
  decision.replaceChildren();
  addDataRow(decision, "Raw Stage 20 decision", data.decisions.actual || data.decisions.precedence[0]);
  addDataRow(decision, "Precedence", "DEFER > REVISE_DETERMINISTICALLY > ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW");
  addDataRow(decision, "Reason codes", (data.decisions.reason_codes || []).join(" · "));
  addDataRow(decision, "Acceptance boundary", "NOT CLINICAL APPROVAL");

  const verifier = byId("verifier-content");
  verifier.replaceChildren();
  data.verifier_statuses.forEach((item) => {
    const row = element("div", "technical-verifier-row");
    row.append(element("span", "", item.status), element("code", "", item.evidence_reference || item.wording));
    verifier.append(row);
  });

  const report = byId("report-content");
  report.replaceChildren();
  addDataRow(report, "Renderer", data.report.identity);
  addDataRow(report, "Structured statements", data.report.statements.length);
  addDataRow(report, "Disclaimer", data.report.disclaimer || "Research use only. Not a medical diagnosis. Expert review is required.");

  const provenance = byId("provenance-content");
  provenance.replaceChildren();
  Object.entries(data.provenance).forEach(([key, value]) => {
    const row = element("div", "provenance-row");
    row.append(element("dt", "", key.replaceAll("_", " ")));
    const description = element("dd", "", value);
    description.title = value;
    row.append(description);
    provenance.append(row);
  });

  const failure = byId("failure-content");
  failure.replaceChildren();
  addDataRow(failure, "DEFER — safety/evidence limitation", (data.dispositions.defer_reason || []).join(" · "));
  addDataRow(failure, "FAILED_SANITIZED — technical processing failure", data.dispositions.technical_failure_recorded ? data.dispositions.failure_code : "No sanitized technical failure recorded.");
}

function setAttributionControls(enabled) {
  const select = byId("attribution-label");
  const button = byId("generate-attribution");
  select.disabled = !enabled;
  button.disabled = !enabled || !selectedLocalFile;
}

function renderAttributionMeta(data) {
  const meta = byId("attribution-meta");
  meta.replaceChildren();
  [["Selected class", displayLabel(data.label)], ["Model score", Number(data.model_score).toFixed(4)],
    ["Method", "Grad-CAM"], ["Target layer", data.target_layer],
    ["Raw attribution", `${data.raw_attribution_dimensions[0]} × ${data.raw_attribution_dimensions[1]}`],
    ["Display map", `${data.display_dimensions[0]} × ${data.display_dimensions[1]}`]].forEach(([key, value]) => {
    meta.append(element("div", "attribution-meta-item"));
    const item = meta.lastElementChild;
    item.append(element("dt", "", key), element("dd", "", String(value)));
  });
}

function clearAttribution(message = "Run a local image review before requesting an attribution.") {
  byId("attribution-visuals").classList.add("is-hidden");
  byId("attribution-status").textContent = message;
  byId("attribution-meta").replaceChildren();
  ["attribution-original", "attribution-heatmap", "attribution-overlay"].forEach((id) => {
    byId(id).removeAttribute("src");
  });
}

async function runAttribution() {
  if (!selectedLocalFile || !currentReviewReady || attributionInProgress) return;
  attributionInProgress = true;
  const button = byId("generate-attribution");
  button.disabled = true;
  byId("attribution-status").textContent = "Generating one bounded research attribution…";
  clearAttribution("Generating one bounded research attribution…");
  try {
    const response = await fetch("/research/explainability/gradcam", {
      method: "POST",
      headers: { "Content-Type": selectedLocalFile.type, "X-TrustCXR-Attribution-Label": byId("attribution-label").value },
      body: selectedLocalFile,
      credentials: "omit",
      cache: "no-store",
    });
    const result = await response.json();
    if (!response.ok) {
      throw new Error(result.reason_code === "ATTRIBUTION_UNAVAILABLE"
        ? "No non-degenerate Grad-CAM attribution was available for this class/input pair. This does not imply absence of pathology."
        : "The research attribution could not be generated.");
    }
    byId("attribution-original").src = localPreviewUrl;
    byId("attribution-heatmap").src = `data:image/png;base64,${result.heatmap_png_base64}`;
    byId("attribution-overlay").src = `data:image/png;base64,${result.overlay_png_base64}`;
    byId("attribution-visuals").classList.remove("is-hidden");
    byId("attribution-status").textContent = "Attribution generated for the selected model class.";
    renderAttributionMeta(result);
  } catch (error) {
    clearAttribution(error.message || "The research attribution could not be generated.");
  } finally {
    attributionInProgress = false;
    setAttributionControls(currentReviewReady);
  }
}

function renderResult(data, localReview = false) {
  const presentation = buildPresentation(data);
  const displayState = data.job.state === "COMPLETED" ? "Analysis Complete" : data.job.state;
  byId("header-job-state").textContent = displayState;
  byId("analysis-state").textContent = displayState;
  byId("analysis-mode-label").textContent = localReview ? "LOCAL IMAGE REVIEW" : "SYNTHETIC DEMO";
  byId("overview-view").textContent = presentation.view;
  // Model-identified view; this is not clinical certification.
  byId("overview-quality").textContent = presentation.quality;
  // Research technical-quality proxy warning; not a clinical image-quality assessment.
  byId("overview-uncertainty").textContent = presentation.uncertainty.label;
  byId("overview-review").textContent = presentation.reviewStatus;
  byId("overview-review-caption").textContent = presentation.reviewStatus === "Expert Review Required"
    ? "Analysis completed. Expert review required."
    : "Analysis completed. Expert review remains required.";
  byId("analysis-error").classList.add("is-hidden");
  byId("results-column").classList.remove("results-cleared");
  if (!localReview) renderViewer(data.synthetic_images.PNG, "PNG");
  renderInterpretation(presentation);
  renderScores(presentation);
  // Research model signal only; not diagnoses, thresholds, severity, temporal change, or clinical certainty.
  renderSummary(presentation);
  renderVerification(data, presentation);
  renderTechnicalDetails(data);
  currentReviewReady = localReview;
  setAttributionControls(localReview);
  clearAttribution(localReview ? "Select one model finding and request one attribution." : "Run a local image review before requesting an attribution.");
  const job = byId("job-status-content");
  job.replaceChildren();
  addDataRow(job, "Mode", localReview ? "LOCAL_IMAGE_REVIEW" : "SYNTHETIC_DEMO");
  addDataRow(job, "Pseudonymous job", data.job.job_id);
}

let localPreviewUrl = null;
let selectedLocalFile = null;
let reviewInProgress = false;
let attributionInProgress = false;
let currentReviewReady = false;

function releaseLocalPreview() {
  if (localPreviewUrl) {
    URL.revokeObjectURL(localPreviewUrl);
    localPreviewUrl = null;
  }
}

function clearAnalysisForLocalReview(message) {
  ["overview-view", "overview-quality", "overview-uncertainty", "overview-review"].forEach((id) => { byId(id).textContent = "—"; });
  ["interpretation-secondary", "top-signals-content", "classifier-content", "summary-content", "verification-summary", "reliability-content", "fusion-content", "decision-content", "verifier-content", "report-content", "provenance-content", "job-status-content", "failure-content"].forEach((id) => byId(id).replaceChildren());
  byId("interpretation-primary-label").textContent = "—";
  byId("interpretation-primary-score").textContent = "—";
  byId("results-column").classList.add("results-cleared");
  byId("analysis-error").classList.add("is-hidden");
  byId("overview-review-caption").textContent = message;
  currentReviewReady = false;
  setAttributionControls(false);
  clearAttribution(message);
}

function previewLocalFile(file) {
  if (!file || !["image/png", "image/jpeg"].includes(file.type)) {
    byId("preview-notice").textContent = "Choose a valid PNG, JPG, or JPEG image.";
    return;
  }
  releaseLocalPreview();
  selectedLocalFile = file;
  localPreviewUrl = URL.createObjectURL(file);
  const viewer = byId("synthetic-viewer");
  viewer.src = localPreviewUrl;
  viewer.alt = "Local browser-memory-only research image preview";
  viewer.removeAttribute("width");
  viewer.removeAttribute("height");
  byId("viewer-format").textContent = file.type === "image/png" ? "PNG" : "JPEG";
  byId("image-dimensions").textContent = "Reading preview";
  byId("preview-notice").textContent = "Local image ready for research analysis.";
  byId("review-mode-badge").textContent = "LOCAL IMAGE";
  byId("analysis-mode-label").textContent = "LOCAL IMAGE REVIEW · READY";
  byId("header-job-state").textContent = "Ready";
  byId("analysis-state").textContent = "Ready";
  clearAnalysisForLocalReview("Run the review to generate current-image results.");
  byId("run-review").disabled = false;
  setAttributionControls(false);
  clearAttribution("Run a local image review before requesting an attribution.");
  viewer.addEventListener("load", () => { byId("image-dimensions").textContent = `${viewer.naturalWidth} × ${viewer.naturalHeight} px`; }, { once: true });
}

function showAnalysisFailure(reasonCode) {
  clearAnalysisForLocalReview("No result is available for this image.");
  byId("header-job-state").textContent = "Failed";
  byId("analysis-state").textContent = "Failed";
  byId("analysis-mode-label").textContent = "LOCAL IMAGE REVIEW";
  byId("analysis-error").classList.remove("is-hidden");
  const failure = byId("failure-content");
  addDataRow(failure, "Sanitized technical code", reasonCode);
}

async function runLocalReview() {
  if (!selectedLocalFile || reviewInProgress) return;
  reviewInProgress = true;
  const button = byId("run-review");
  const label = button.querySelector("span");
  button.disabled = true;
  label.textContent = "Analyzing...";
  byId("header-job-state").textContent = "Analyzing";
  byId("analysis-state").textContent = "Analyzing...";
  byId("analysis-mode-label").textContent = "LOCAL IMAGE REVIEW · ANALYZING";
  clearAnalysisForLocalReview("Current-image analysis is in progress.");
  try {
    const response = await fetch("/ui/research-review", { method: "POST", headers: { "Content-Type": selectedLocalFile.type }, body: selectedLocalFile, credentials: "omit", cache: "no-store" });
    const result = await response.json();
    if (!response.ok) {
      const reason = Array.isArray(result.reason_codes) ? result.reason_codes.join(" · ") : "INFERENCE_FAILURE";
      throw new Error(reason);
    }
    renderResult(result, true);
    byId("review-mode-badge").textContent = "LOCAL IMAGE";
    byId("preview-notice").textContent = "Local research analysis completed.";
  } catch (error) {
    // "LOCAL IMAGE REVIEW · FAILED_SANITIZED" remains available in technical evidence only.
    showAnalysisFailure(error.message);
    byId("preview-notice").textContent = "Analysis could not be completed. Try another valid image.";
  } finally {
    reviewInProgress = false;
    button.disabled = selectedLocalFile === null;
    label.textContent = "Run TrustCXR Review";
  }
}

function configureLocalPreview() {
  const dropZone = byId("drop-zone");
  const input = byId("local-image-input");
  dropZone.addEventListener("click", () => input.click());
  dropZone.addEventListener("keydown", (event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); input.click(); } });
  input.addEventListener("change", () => previewLocalFile(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.add("is-dragging"); }));
  ["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => { event.preventDefault(); dropZone.classList.remove("is-dragging"); }));
  dropZone.addEventListener("drop", (event) => previewLocalFile(event.dataTransfer.files[0]));
  byId("run-review").addEventListener("click", runLocalReview);
  const select = byId("attribution-label");
  ATTRIBUTION_LABELS.forEach((label) => {
    const option = element("option", "", displayLabel(label));
    option.value = label;
    select.append(option);
  });
  byId("generate-attribution").addEventListener("click", runAttribution);
  window.addEventListener("beforeunload", releaseLocalPreview);
}

configureLocalPreview();
// A failed current-image review never falls back to fixture results:
// No current-image result is available because the review failed.
// LOCAL_REVIEW_ENDPOINT_UNAVAILABLE_RESTART_SERVER is a sanitized legacy diagnostic code.
fetch("/ui/fixtures.json", { credentials: "omit", cache: "no-store" })
  .then((response) => { if (!response.ok) throw new Error("fixture_unavailable"); return response.json(); })
  .then((data) => renderResult(data, false))
  .catch(() => showAnalysisFailure("FIXTURE_UNAVAILABLE"));
