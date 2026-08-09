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
  row.append(element("span", "", label), element("strong", "", value));
  container.append(row);
}

function statusClass(status) {
  const classes = {
    VERIFIED: "status-verified",
    PARTIALLY_VERIFIED: "status-partial",
    PARTIALLY_SUPPORTED: "status-partial",
    UNVERIFIED: "status-unverified",
    CONTRADICTED: "status-contradicted",
    NOT_APPLICABLE: "status-na",
    WITHHELD_INSUFFICIENT_EVIDENCE: "status-withheld",
  };
  return classes[status] || "status-unverified";
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

function renderFixture(data, preservePreview = false) {
  const jobState = data.job.state.toLowerCase();
  byId("header-job-state").textContent = jobState;
  byId("analysis-state").textContent = jobState;
  byId("kpi-view").textContent = data.view.selected;
  byId("kpi-decision").textContent = data.decisions.actual || data.decisions.precedence[0];

  const job = byId("job-status-content");
  job.replaceChildren();
  addDataRow(job, "Pseudonymous job", data.job.job_id);
  addDataRow(job, "Workflow", "Synthetic fixture review only");
  if (!preservePreview) renderViewer(data.synthetic_images.PNG, "PNG");

  const view = byId("view-quality-content");
  view.replaceChildren();
  addDataRow(view, "Model-identified view", `${data.view.selected} — not clinical certification`);
  const qualityText = data.technical_quality
    ? `${data.technical_quality.status} · model score ${Number(data.technical_quality.score).toFixed(6)} · not a clinical image-quality assessment.`
    : "Research technical-quality proxy warning; not a clinical image-quality assessment.";
  addDataRow(view, "Technical quality", qualityText);
  addDataRow(view, "Governed states", data.view.states.join(" · "));

  const scores = byId("classifier-content");
  scores.replaceChildren();
  data.classifier_scores.forEach((item) => {
    const row = element("div", "score-row");
    const label = element("span", "score-label", item.label);
    const track = element("div", "score-track");
    track.setAttribute("aria-label", `${item.label} research model score ${item.score.toFixed(4)}`);
    const fill = element("div", "score-fill");
    fill.style.setProperty("--score", `${Math.max(0, Math.min(1, item.score)) * 100}%`);
    track.append(fill);
    row.append(label, track, element("span", "score-value", item.score.toFixed(4)));
    scores.append(row);
  });
  const scoreQualifier = element("p", "visually-hidden", "Research model signal only; not diagnoses, thresholds, severity, temporal change, or clinical certainty");
  scores.append(scoreQualifier);

  const reliability = byId("reliability-content");
  reliability.replaceChildren();
  const reliabilityCallout = element("div", "reliability-callout");
  reliabilityCallout.append(
    element("strong", "", "PREDICTIVE UNCERTAINTY"),
    element("p", "", "Predictive uncertainty only. Not epistemic uncertainty."),
  );
  const gauge = element("div", "reliability-gauge");
  const gaugeTrack = element("div", "gauge-track");
  gaugeTrack.setAttribute("aria-label", "Predictive uncertainty evidence indicator; no clinical certainty meaning");
  const gaugeFill = element("div", "gauge-fill");
  if (Number.isFinite(data.reliability.predictive_uncertainty)) {
    const uncertaintyPercent = Math.min(100, Math.max(0, data.reliability.predictive_uncertainty / Math.log(2) * 100));
    gaugeFill.style.width = `${uncertaintyPercent}%`;
    gaugeTrack.setAttribute("aria-label", `Predictive uncertainty ${data.reliability.predictive_uncertainty.toFixed(6)}`);
  }
  gaugeTrack.append(gaugeFill);
  gauge.append(gaugeTrack);
  reliability.append(reliabilityCallout, gauge);
  if (Number.isFinite(data.reliability.predictive_uncertainty)) {
    addDataRow(reliability, "Maximum Bernoulli predictive entropy", data.reliability.predictive_uncertainty.toFixed(6));
  }
  addDataRow(reliability, "Calibration", data.reliability.calibration_label);
  addDataRow(reliability, "OOD capability", data.reliability.ood);
  addDataRow(reliability, "Stage 13 selective prediction", data.reliability.stage13_selective_prediction);

  const fusion = byId("fusion-content");
  fusion.replaceChildren();
  const fusionCallout = element("div", "fusion-callout");
  const fusionChip = element("span", `status-chip ${statusClass(data.fusion.status)}`, data.fusion.status);
  fusionCallout.append(
    fusionChip,
    element("strong", "", "Partial governed research evidence only."),
    element("p", "", data.fusion.label),
  );
  fusion.append(fusionCallout);
  addDataRow(fusion, "Identity", "Exact governed identity required");
  addDataRow(fusion, "Localization boundary", "No reliable lesion localization, finding laterality, or contradiction from localization absence.");
  const stage8 = element("p", "visually-hidden", "Stage 8 WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION");
  const stage10 = element("p", "visually-hidden", "Stage 10 WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION");
  fusion.append(stage8, stage10);

  const verifier = byId("verifier-content");
  verifier.replaceChildren();
  data.verifier_statuses.forEach((item) => {
    const card = element("article", "verifier-card");
    card.append(
      element("span", `status-chip ${statusClass(item.status)}`, item.status),
      element("p", "", item.wording),
    );
    verifier.append(card);
  });

  const report = byId("report-content");
  report.replaceChildren();
  report.append(element("div", "report-identity", data.report.identity));
  const reportCopy = element("div", "report-copy");
  reportCopy.append(element("p", "", "Deterministic research draft generated from governed structured evidence for expert review."));
  data.report.statements.forEach((statement) => {
    reportCopy.append(element("p", "", typeof statement === "string" ? statement : statement.text));
  });
  report.append(
    reportCopy,
    element("p", "report-disclaimer", data.report.disclaimer || "Research use only. Not a medical diagnosis. Expert review is required."),
  );

  const decision = byId("decision-content");
  decision.replaceChildren();
  const precedence = element("p", "precedence");
  precedence.append(element("strong", "", "DEFER"), document.createTextNode(" > REVISE_DETERMINISTICALLY > ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW"));
  const stack = element("div", "decision-stack");
  [
    ["is-defer", "DEFER", data.decisions.defer],
    ["is-revise", "REVISE_DETERMINISTICALLY", data.decisions.revise],
    ["is-accept", "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW", data.decisions.accept],
  ].forEach(([className, title, description]) => {
    const card = element("article", `decision-card ${className}`);
    card.append(element("strong", "", title), element("p", "", description));
    stack.append(card);
  });
  decision.append(precedence, stack, element("p", "contract-note", "Acceptance is not clinical approval. DEFER remains highest precedence."));

  const provenance = byId("provenance-content");
  provenance.replaceChildren();
  Object.entries(data.provenance).forEach(([key, value]) => {
    const item = element("div", "provenance-item");
    const term = element("dt", "", key.replaceAll("_", " "));
    const description = element("dd", "", value);
    description.title = value;
    item.append(term, description);
    provenance.append(item);
  });

  const failure = byId("failure-content");
  failure.replaceChildren();
  const failureGrid = element("div", "failure-grid");
  const deferBox = element("article", "failure-box");
  deferBox.append(
    element("strong", "status-defer", "DEFER — safety/evidence limitation"),
    element("span", "", Array.isArray(data.dispositions.defer_reason) ? data.dispositions.defer_reason.join(" · ") : data.dispositions.defer_reason),
  );
  const technicalBox = element("article", "failure-box");
  const failureRecorded = data.dispositions.technical_failure_recorded === true;
  technicalBox.append(
    element("strong", "status-failed", "FAILED_SANITIZED — technical processing failure"),
    element("span", "", failureRecorded ? data.dispositions.failure_code : "No sanitized technical failure recorded."),
  );
  failureGrid.append(deferBox, technicalBox);
  failure.append(failureGrid);
}

let fixtureData = null;
let localPreviewUrl = null;
let selectedLocalFile = null;
let reviewInProgress = false;

function releaseLocalPreview() {
  if (localPreviewUrl) {
    URL.revokeObjectURL(localPreviewUrl);
    localPreviewUrl = null;
  }
}

function previewLocalFile(file) {
  if (!file || !["image/png", "image/jpeg"].includes(file.type)) {
    byId("preview-notice").textContent = "Only PNG, JPG, or JPEG files can be previewed locally. DICOM remains withheld.";
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
  byId("image-dimensions").textContent = "Reading local preview";
  byId("preview-notice").textContent = "Local browser-memory preview only. This image is not uploaded, stored, or analyzed.";
  byId("review-mode-badge").textContent = "LOCAL IMAGE REVIEW";
  byId("run-review").disabled = false;
  viewer.addEventListener("load", () => {
    byId("image-dimensions").textContent = `${viewer.naturalWidth} × ${viewer.naturalHeight} px`;
  }, { once: true });
}

function configureLocalPreview() {
  const dropZone = byId("drop-zone");
  const input = byId("local-image-input");
  dropZone.addEventListener("click", () => input.click());
  dropZone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => previewLocalFile(input.files[0]));
  ["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
    event.preventDefault();
    dropZone.classList.remove("is-dragging");
  }));
  dropZone.addEventListener("drop", (event) => previewLocalFile(event.dataTransfer.files[0]));
  byId("run-review").addEventListener("click", runLocalReview);
  window.addEventListener("beforeunload", releaseLocalPreview);
}

async function runLocalReview() {
  if (!selectedLocalFile || reviewInProgress) {
    byId("preview-notice").textContent = "Choose a local PNG or JPEG image before running a review.";
    return;
  }
  reviewInProgress = true;
  const button = byId("run-review");
  button.disabled = true;
  const originalLabel = button.lastChild.textContent;
  button.lastChild.textContent = "Analyzing...";
  byId("header-job-state").textContent = "analyzing";
  byId("analysis-state").textContent = "Analyzing...";
  byId("preview-notice").textContent = "Running the frozen TrustCXR research pipeline locally. No external service is used.";
  try {
    const response = await fetch("/ui/research-review", {
      method: "POST",
      headers: { "Content-Type": selectedLocalFile.type },
      body: selectedLocalFile,
      credentials: "omit",
      cache: "no-store",
    });
    const result = await response.json();
    if (!response.ok) {
      const reason = Array.isArray(result.reason_codes) ? result.reason_codes.join(" · ") : "INFERENCE_FAILURE";
      throw new Error(reason);
    }
    renderFixture(result, true);
    byId("review-mode-badge").textContent = "LOCAL IMAGE REVIEW";
    byId("preview-notice").textContent = "Local image review completed from the selected image. Research use only.";
  } catch (error) {
    byId("header-job-state").textContent = "failed";
    byId("analysis-state").textContent = "FAILED_SANITIZED";
    byId("preview-notice").textContent = `Local review failed safely: ${error.message}`;
  } finally {
    reviewInProgress = false;
    button.disabled = selectedLocalFile === null;
    button.lastChild.textContent = originalLabel;
  }
}

configureLocalPreview();
fetch("/ui/fixtures.json", { credentials: "omit", cache: "no-store" })
  .then((response) => {
    if (!response.ok) throw new Error("fixture_unavailable");
    return response.json();
  })
  .then((data) => {
    fixtureData = data;
    renderFixture(data);
  })
  .catch(() => {
    byId("header-job-state").textContent = "unavailable";
    byId("analysis-state").textContent = "FAILED_SANITIZED";
    byId("preview-notice").textContent = "Synthetic fixture unavailable. No processing was attempted.";
  });
