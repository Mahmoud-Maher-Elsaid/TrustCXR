"use strict";

const byId = (id) => document.getElementById(id);
const addText = (parent, tag, text, className = "") => {
  const node = document.createElement(tag);
  node.textContent = String(text);
  if (className) node.className = className;
  parent.appendChild(node);
  return node;
};

const content = (panelId) => document.querySelector(`#${panelId} [data-content]`);

const renderScores = (parent, scores) => {
  const qualifier = addText(parent, "p", "Research model scores; not diagnoses, thresholds, severity, temporal change, or clinical certainty.");
  qualifier.className = "status";
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  ["Finding label", "Model score", "Qualifier"].forEach((label) => addText(headRow, "th", label));
  head.appendChild(headRow);
  table.appendChild(head);
  const body = document.createElement("tbody");
  scores.forEach((item) => {
    const row = document.createElement("tr");
    addText(row, "td", item.label);
    addText(row, "td", item.score.toFixed(4));
    addText(row, "td", "Research model signal only");
    body.appendChild(row);
  });
  table.appendChild(body);
  parent.appendChild(table);
};

const renderFixture = (fixture, imageFormat = "PNG") => {
  addText(content("job-status-panel"), "p", `Job: ${fixture.job.job_id}`);
  addText(content("job-status-panel"), "p", `State: ${fixture.job.state}`, "status");

  const image = byId("synthetic-image");
  const selectedImage = fixture.synthetic_images[imageFormat];
  image.src = selectedImage.data_url;
  image.alt = selectedImage.alt;
  byId("image-dimensions").textContent = `${selectedImage.width} × ${selectedImage.height} pixels — synthetic non-patient ${imageFormat}`;
  const viewer = content("medical-image-viewer-panel");
  addText(viewer, "p", "DICOM: WITHHELD_NO_GOVERNED_GENERIC_VIEWER_CONTRACT", "status status-withheld");
  addText(viewer, "p", "Stage 8 overlay: WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION", "status status-withheld");
  addText(viewer, "p", "Stage 10 overlay: WITHHELD_PENDING_EXPLICIT_UI_OVERLAY_AUTHORIZATION", "status status-withheld");

  const viewPanel = content("view-quality-panel");
  addText(viewPanel, "p", `Model-identified view: ${fixture.view.selected}; not clinical certification.`);
  addText(viewPanel, "p", "Research technical-quality proxy warning; not a clinical image-quality assessment.");
  addText(viewPanel, "p", `Synthetic view states validated: ${fixture.view.states.join(", ")}`);

  renderScores(content("classifier-score-panel"), fixture.classifier_scores);
  const reliability = content("reliability-panel");
  addText(reliability, "p", "PREDICTIVE UNCERTAINTY", "status");
  addText(reliability, "p", fixture.reliability.calibration_label);
  addText(reliability, "p", `OOD: ${fixture.reliability.ood}; Stage 13 selective prediction: ${fixture.reliability.stage13_selective_prediction}`);

  addText(content("fusion-panel"), "p", fixture.fusion.label, "status status-partial");
  addText(content("fusion-panel"), "p", "Exact governed identity required; no reliable localization, laterality, or contradiction from localization absence.");

  const report = content("report-panel");
  addText(report, "p", "Research use only. Not a medical diagnosis. Expert review is required.");
  fixture.report.statements.forEach((statement) => addText(report, "p", statement));

  const verifier = content("verifier-panel");
  fixture.verifier_statuses.forEach((item) => {
    const style = item.status === "PARTIALLY_VERIFIED" ? "status status-partial" : item.status.startsWith("WITHHELD") ? "status status-withheld" : "status";
    addText(verifier, "p", `${item.status}: ${item.wording}`, style);
  });

  const decision = content("decision-panel");
  addText(decision, "p", `Precedence: ${fixture.decisions.precedence.join(" > ")}`);
  addText(decision, "p", fixture.decisions.defer, "status status-defer");
  addText(decision, "p", fixture.decisions.revise);
  addText(decision, "p", fixture.decisions.accept);

  const provenance = content("provenance-panel");
  Object.entries(fixture.provenance).forEach(([key, value]) => addText(provenance, "p", `${key}: ${value}`));

  const errors = content("error-panel");
  addText(errors, "p", `DEFER — safety/evidence limitation: ${fixture.dispositions.defer_reason}`, "status status-defer");
  addText(errors, "p", `FAILED_SANITIZED — technical processing failure: ${fixture.dispositions.failure_code}`, "status status-failed");
};

fetch("/ui/fixtures.json", {cache: "no-store", credentials: "omit"})
  .then((response) => {
    if (!response.ok) throw new Error("SANITIZED_UI_FIXTURE_LOAD_FAILURE");
    return response.json();
  })
  .then((fixture) => renderFixture(fixture))
  .catch(() => addText(content("error-panel"), "p", "FAILED_SANITIZED — synthetic UI fixture could not be loaded.", "status status-failed"));
