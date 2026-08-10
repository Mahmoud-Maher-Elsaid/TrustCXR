from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from threading import Lock
from typing import Any

import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as vision_functional

from trustcxr.decision.deterministic_policy import decide
from trustcxr.integration.stage9b_ablation import LABELS, build_model
from trustcxr.quality.dataset import VIEW_LABELS, build_transforms
from trustcxr.quality.model import EfficientNetQualityView
from trustcxr.reporting.grounded_contract import render_report
from trustcxr.verification.deterministic_verifier import verify_textual

MAX_UPLOAD_BYTES = 16 * 1024 * 1024
MAX_IMAGE_DIMENSION = 4096
STAGE5_CHECKPOINT_SHA256 = "1ca66fcffe590dac5c4c56d9d1233c29bb51a21000db7da4483673c1f7182a74"
STAGE9_CHECKPOINT_SHA256 = "bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de"
STAGE5_CONFIG_SHA256 = "de641a682362b5671c8d761d77c9d507ae907041bb7b3f057489f20581e1021c"
STAGE9_CONFIG_SHA256 = "347e2a1bebbad2d48932d9d6163217d0194ae1eff507b2d717bfa932fca84ef4"
STAGE9_TEMPERATURE = 0.8293970034646612
STAGE9_ABSTENTION_THRESHOLD = 0.6917239984806322
SUPPORTED_MEDIA_TYPES = {"image/png": "PNG", "image/jpeg": "JPEG"}
OMITTED_CAPABILITIES = (
    "reliable_positive_localization_claim",
    "classifier_negation_from_localization_absence",
    "severity",
    "temporal_change",
    "ood_status_claim",
    "device_localization",
    "clinical_image_quality_diagnosis",
    "clinical_certainty_from_probability",
    "patient_history",
    "treatment_recommendation",
)


class LocalReviewError(RuntimeError):
    def __init__(
        self, reason_code: str, status_code: int, stage: str = "REQUEST_VALIDATION"
    ) -> None:
        super().__init__(reason_code)
        self.reason_code = reason_code
        self.status_code = status_code
        self.stage = stage


class PipelineExecutionError(RuntimeError):
    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__("LOCAL_RESEARCH_PIPELINE_STAGE_FAILED")
        self.stage = stage
        self.cause_type = type(cause).__name__


def execute_stage(stage: str, function: Any, *args: Any) -> Any:
    try:
        return function(*args)
    except LocalReviewError as error:
        error.stage = stage
        raise
    except Exception as error:
        raise PipelineExecutionError(stage, error) from error


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_image(payload: bytes, media_type: str) -> tuple[Image.Image, str]:
    expected_format = SUPPORTED_MEDIA_TYPES.get(media_type)
    if expected_format is None:
        raise LocalReviewError("UNSUPPORTED_INPUT", 415)
    if not payload:
        raise LocalReviewError("INVALID_IMAGE", 422)
    if len(payload) > MAX_UPLOAD_BYTES:
        raise LocalReviewError("INPUT_TOO_LARGE", 413)
    try:
        with Image.open(io.BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(payload)) as opened:
            if opened.format != expected_format:
                raise LocalReviewError("UNSUPPORTED_INPUT", 415)
            width, height = opened.size
            if (
                width < 1
                or height < 1
                or width > MAX_IMAGE_DIMENSION
                or height > MAX_IMAGE_DIMENSION
            ):
                raise LocalReviewError("INVALID_IMAGE_DIMENSIONS", 422)
            opened.load()
            return opened.convert("RGB"), expected_format
    except LocalReviewError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as error:
        raise LocalReviewError("INVALID_IMAGE", 422) from error


def _verify_checkpoint(path: Path, expected: str) -> None:
    if not path.is_file():
        raise LocalReviewError("MODEL_LOAD_FAILURE", 503)
    if sha256(path) != expected:
        raise LocalReviewError("CHECKPOINT_HASH_MISMATCH", 503)


def _release_model(model: torch.nn.Module, device: torch.device) -> None:
    model.to("cpu")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


class FrozenModelRunner:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.stage5_checkpoint = self.root / "artifacts/stage5/best_quality_view.pt"
        self.stage9_checkpoint = (
            self.root / "artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt"
        )

    @staticmethod
    def _device() -> torch.device:
        if not torch.cuda.is_available():
            raise LocalReviewError("CUDA_UNAVAILABLE", 503)
        return torch.device("cuda")

    def stage5(self, image: Image.Image) -> dict[str, Any]:
        _verify_checkpoint(self.stage5_checkpoint, STAGE5_CHECKPOINT_SHA256)
        device = self._device()
        checkpoint = torch.load(self.stage5_checkpoint, map_location="cpu", weights_only=False)
        if tuple(checkpoint.get("labels", ())) != VIEW_LABELS:
            raise LocalReviewError("PROVENANCE_FAILURE", 503)
        model = EfficientNetQualityView(pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().to(device)
        tensor = build_transforms(224, training=False)(image).unsqueeze(0).to(device)
        try:
            with torch.inference_mode():
                output = model(tensor)
                view_index = int(output["view_logits"].argmax(dim=1).item())
                quality_score = float(torch.sigmoid(output["quality_logit"])[0].item())
        finally:
            del tensor
            _release_model(model, device)
        return {
            "view": VIEW_LABELS[view_index],
            "quality_score": quality_score,
            "quality_pass": quality_score >= 0.5,
        }

    def stage9(self, image: Image.Image) -> dict[str, Any]:
        _verify_checkpoint(self.stage9_checkpoint, STAGE9_CHECKPOINT_SHA256)
        device = self._device()
        checkpoint = torch.load(self.stage9_checkpoint, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("model_architecture") != "DenseNet121"
            or checkpoint.get("variant") != "original"
        ):
            raise LocalReviewError("PROVENANCE_FAILURE", 503)
        if checkpoint.get("config_sha256") != STAGE9_CONFIG_SHA256:
            raise LocalReviewError("PROVENANCE_FAILURE", 503)
        model = build_model(len(LABELS), input_channels=3, pretrained=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        model.eval().to(device)
        resized = vision_functional.resize(
            image,
            [224, 224],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        tensor = vision_functional.to_tensor(resized)
        tensor = (
            vision_functional.normalize(
                tensor,
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            )
            .unsqueeze(0)
            .to(device)
        )
        try:
            with torch.inference_mode():
                logits = model(tensor)[0]
                raw_probabilities = torch.sigmoid(logits).cpu().tolist()
                calibrated = torch.sigmoid(logits / STAGE9_TEMPERATURE).cpu().tolist()
        finally:
            del tensor
            _release_model(model, device)
        entropies = [
            -(probability * math.log(max(probability, 1e-12)))
            - ((1.0 - probability) * math.log(max(1.0 - probability, 1e-12)))
            for probability in calibrated
        ]
        return {
            "scores": [float(value) for value in raw_probabilities],
            "calibrated_scores": [float(value) for value in calibrated],
            "predictive_uncertainty": max(entropies),
        }


class LocalResearchPipeline:
    def __init__(self, root: Path, runner: FrozenModelRunner | None = None) -> None:
        self.root = root.resolve()
        self.runner = runner or FrozenModelRunner(self.root)
        self._inference_lock = Lock()
        self.report_contract = json.loads(
            (
                self.root / "configs/reporting/stage18b_deterministic_grounded_report_contract.json"
            ).read_text(encoding="utf-8")
        )
        self.decision_contract = json.loads(
            (
                self.root / "configs/decision/stage20b_deterministic_decision_contract.json"
            ).read_text(encoding="utf-8")
        )

    def review(self, payload: bytes, media_type: str) -> dict[str, Any]:
        with self._inference_lock:
            return self._review_locked(payload, media_type)

    def _review_locked(self, payload: bytes, media_type: str) -> dict[str, Any]:
        image, image_format = execute_stage("IMAGE_DECODE", decode_image, payload, media_type)
        content_fingerprint = hashlib.sha256(payload).hexdigest()
        job_id = (
            "job_"
            + hashlib.sha256(("local-review:" + content_fingerprint).encode()).hexdigest()[:24]
        )
        stage5 = execute_stage("STAGE5_INFERENCE", self.runner.stage5, image)
        stage9 = execute_stage("STAGE9_INFERENCE", self.runner.stage9, image)
        uncertainty = float(stage9["predictive_uncertainty"])
        defer_reasons = ["FUSION_EVIDENCE_NOT_RELIABLY_SUPPORTIVE"]
        if uncertainty > STAGE9_ABSTENTION_THRESHOLD:
            defer_reasons.append("PREDICTIVE_UNCERTAINTY_ABOVE_FROZEN_LIMIT")
        if not stage5["quality_pass"]:
            defer_reasons.append("TECHNICAL_QUALITY_PROXY_FAILED_NOT_CLINICAL_QUALITY")

        statements = [self._view_statement(str(stage5["view"]))]
        if not stage5["quality_pass"]:
            statements.append(self._quality_statement())
        statements.extend(
            self._classifier_statement(label, score, index)
            for index, (label, score) in enumerate(zip(LABELS, stage9["scores"], strict=True))
        )
        statements.append(self._fusion_statement())
        statements.append(self._defer_statement(defer_reasons))
        report_payload = {
            "report_identity": self.report_contract["report_identity"],
            "research_use_disclaimer": self.report_contract["research_use_disclaimer"],
            "statements": statements,
            "omitted_capabilities": [
                {"capability": capability, "reason_code": "UNSUPPORTED_CAPABILITY_OMITTED"}
                for capability in OMITTED_CAPABILITIES
            ],
        }
        report = render_report(report_payload, self.report_contract)
        verifier_statuses = []
        for statement, rendered in zip(
            sorted(
                statements,
                key=lambda row: (
                    row["source_stage"],
                    row["structured_source_field"],
                    row["evidence_code"],
                ),
            ),
            report["statements"],
            strict=True,
        ):
            status = verify_textual(
                statement,
                rendered["text"],
                self.report_contract,
                evidence_available=True,
                exact_identity=False,
            )
            verifier_statuses.append(
                {
                    "status": status,
                    "wording": (
                        "Exact governed study identity is unavailable for this local upload; "
                        "verification is withheld."
                    ),
                    "evidence_reference": (
                        f"local-review/{content_fingerprint[:16]}/"
                        f"{statement['structured_source_field']}"
                    ),
                }
            )

        evidence_references = sorted({item["evidence_reference"] for item in verifier_statuses})
        decision = decide(
            {
                "required_statuses": [item["status"] for item in verifier_statuses],
                "non_required_statuses": [],
                "provenance_valid": True,
                "exact_identity": False,
                "templates_conformant": True,
                "active_stage17_defer": True,
                "forbidden_claim": False,
                "unsupported_capability": False,
                "required_evidence_missing": False,
                "stage11_limited_support_required": True,
                "anatomical_proxy_overreach": False,
                "missing_eligible_input": False,
                "revision_candidate": False,
                "same_evidence_sufficient": False,
                "canonical_template_available": False,
                "provenance_preservable": True,
                "introduces_new_fact": False,
                "semantic_interpretation_required": False,
                "evidence_references": evidence_references,
            },
            self.decision_contract,
        )
        decisions = {
            "precedence": [
                "DEFER",
                "REVISE_DETERMINISTICALLY",
                "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW",
            ],
            "defer": "DEFER — no clinical urgency meaning.",
            "revise": "REVISE_DETERMINISTICALLY — DETERMINISTIC CANONICAL TEMPLATE REPAIR ONLY.",
            "accept": "ACCEPT_RESEARCH_DRAFT_FOR_EXPERT_REVIEW — NOT CLINICAL APPROVAL.",
            "actual": decision["decision"],
            "reason_codes": decision["reason_codes"],
        }
        return {
            "schema_version": "trustcxr-local-review-v1",
            "mode": "LOCAL_IMAGE_REVIEW",
            "research_designation": "RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED",
            "job": {"job_id": job_id, "state": "COMPLETED"},
            "image": {"format": image_format, "width": image.width, "height": image.height},
            "view": {"selected": stage5["view"], "states": list(VIEW_LABELS)},
            "technical_quality": {
                "score": float(stage5["quality_score"]),
                "passed": bool(stage5["quality_pass"]),
                "status": "PASS" if stage5["quality_pass"] else "WARNING",
                "qualifier": "RESEARCH_TECHNICAL_QUALITY_PROXY_NOT_CLINICAL_ASSESSMENT",
            },
            "classifier_scores": [
                {"label": label, "score": float(score)}
                for label, score in zip(LABELS, stage9["scores"], strict=True)
            ],
            "reliability": {
                "calibration_label": (
                    "Validation-derived and model-specific calibration information."
                ),
                "temperature": STAGE9_TEMPERATURE,
                "predictive_uncertainty": uncertainty,
                "uncertainty_claim": "PREDICTIVE_ONLY_NOT_EPISTEMIC",
                "selective_abstain": uncertainty > STAGE9_ABSTENTION_THRESHOLD,
                "ood": "WITHHELD_NO_GOVERNED_OOD_COHORT",
                "stage13_selective_prediction": "NOT_ACCEPTED",
            },
            "fusion": {
                "status": "WITHHELD_INSUFFICIENT_EVIDENCE",
                "label": (
                    "No governed Stage 11 identity-aligned localization evidence exists for "
                    "this local upload."
                ),
            },
            "report": {
                "identity": report["report_identity"],
                "disclaimer": report["research_use_disclaimer"],
                "statements": report["statements"],
            },
            "verifier_statuses": verifier_statuses,
            "decisions": decisions,
            "dispositions": {
                "defer_reason": decision["reason_codes"],
                "technical_failure_recorded": False,
                "failure_code": "NONE",
            },
            "provenance": {
                "SOURCE_STAGE": "5,9,16,17,18,19,20",
                "SOURCE_VERSION": "frozen-local-research-v1",
                "EVIDENCE_CODE": "CURRENT_IMAGE_MODEL_EVIDENCE",
                "COMPONENT_IDENTIFIER": "trustcxr_local_research_pipeline",
                "CONFIG_FINGERPRINT": (
                    f"stage5:{STAGE5_CONFIG_SHA256};stage9:{STAGE9_CONFIG_SHA256}"
                ),
                "CHECKPOINT_FINGERPRINT": (
                    f"stage5:{STAGE5_CHECKPOINT_SHA256};stage9:{STAGE9_CHECKPOINT_SHA256}"
                ),
                "VERIFIER_EVIDENCE_REFERENCE": f"local-review/{content_fingerprint[:16]}",
            },
            "limitations": [
                "NO_CLINICAL_DIAGNOSIS",
                "NO_TREATMENT_RECOMMENDATION",
                "NO_RELIABLE_POSITIVE_LESION_LOCALIZATION",
                "OOD_WITHHELD",
                "EXTERNAL_VALIDATION_NOT_PERFORMED",
            ],
        }

    @staticmethod
    def _view_statement(view: str) -> dict[str, Any]:
        return {
            "evidence_type": "model_identified_view_ap_pa_or_lateral",
            "grounding_status": "DIRECT_STRUCTURED",
            "template_id": "VIEW_IDENTIFIED",
            "template_parameters": {"view": view},
            "source_stage": "5",
            "source_version": "frozen-v1",
            "evidence_code": "VIEW_MODEL_OUTPUT",
            "structured_source_field": "stage5.view",
        }

    @staticmethod
    def _quality_statement() -> dict[str, Any]:
        return {
            "evidence_type": "technical_quality_proxy_warning_with_nonclinical_qualifier",
            "grounding_status": "DIRECT_STRUCTURED",
            "template_id": "TECHNICAL_PROXY_WARNING",
            "template_parameters": {"warning": "TECHNICAL_PROXY_FAILED"},
            "source_stage": "5",
            "source_version": "frozen-v1",
            "evidence_code": "TECHNICAL_QUALITY_PROXY_WARNING",
            "structured_source_field": "stage5.technical_quality_proxy",
        }

    @staticmethod
    def _classifier_statement(label: str, score: float, index: int) -> dict[str, Any]:
        return {
            "evidence_type": "stage9_classifier_finding_signal",
            "grounding_status": "EXPLICIT_UNCERTAINTY",
            "template_id": "CLASSIFIER_SIGNAL_UNCERTAIN",
            "template_parameters": {"finding": label, "model_score": float(score)},
            "source_stage": "9",
            "source_version": "original-frozen-v1",
            "evidence_code": "CLASSIFIER_SIGNAL",
            "structured_source_field": f"stage9.scores.{index:02d}",
        }

    @staticmethod
    def _fusion_statement() -> dict[str, Any]:
        return {
            "evidence_type": "stage11_uncertain_or_unlocalized_fusion_status",
            "grounding_status": "EXPLICIT_UNCERTAINTY",
            "template_id": "FUSION_STATUS_UNCERTAIN",
            "template_parameters": {"fusion_status": "UNLOCALIZED"},
            "source_stage": "11",
            "source_version": "frozen-v1",
            "evidence_code": "FUSION_UNLOCALIZED",
            "structured_source_field": "stage11.fusion_status",
        }

    @staticmethod
    def _defer_statement(reason_codes: list[str]) -> dict[str, Any]:
        return {
            "evidence_type": "research_system_defer_status_with_reason_code",
            "grounding_status": "DIRECT_STRUCTURED",
            "template_id": "RESEARCH_DEFER",
            "template_parameters": {"reason_codes": reason_codes},
            "source_stage": "17",
            "source_version": "frozen-v1",
            "evidence_code": "RESEARCH_TRIAGE_DEFER",
            "structured_source_field": "stage17.defer_reason_codes",
        }
