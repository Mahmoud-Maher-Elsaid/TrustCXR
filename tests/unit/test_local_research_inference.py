from __future__ import annotations

import asyncio
import io
import json
import logging
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from trustcxr.integration.stage9b_ablation import build_model
from trustcxr.quality.dataset import VIEW_LABELS, build_transforms
from trustcxr.quality.model import EfficientNetQualityView
from trustcxr.serving.api import create_app
from trustcxr.serving.local_inference import (
    LABELS,
    MAX_UPLOAD_BYTES,
    STAGE5_CHECKPOINT_SHA256,
    STAGE9_CHECKPOINT_SHA256,
    LocalResearchPipeline,
    sha256,
)

ROOT = Path(__file__).resolve().parents[2]


def png_fixture(value: int, size: tuple[int, int] = (96, 80)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, (value, value, value)).save(output, format="PNG")
    return output.getvalue()


class DeterministicImageRunner:
    def stage5(self, image: Image.Image) -> dict[str, Any]:
        mean = sum(image.resize((1, 1)).getpixel((0, 0))) / 3
        return {"view": "AP" if mean < 128 else "PA", "quality_score": 0.81, "quality_pass": True}

    def stage9(self, image: Image.Image) -> dict[str, Any]:
        mean = sum(image.resize((1, 1)).getpixel((0, 0))) / (3 * 255)
        scores = [min(0.99, 0.01 + mean * (index + 1) / len(LABELS)) for index in range(14)]
        return {
            "scores": scores,
            "calibrated_scores": scores,
            "predictive_uncertainty": 0.5 + mean / 10,
        }


class FailingStage5Runner:
    def stage5(self, image: Image.Image) -> dict[str, Any]:
        raise RuntimeError("bounded synthetic stage5 failure")

    def stage9(self, image: Image.Image) -> dict[str, Any]:
        raise AssertionError("Stage 9 must not be reached after Stage 5 fails")


def pipeline() -> LocalResearchPipeline:
    return LocalResearchPipeline(ROOT, runner=DeterministicImageRunner())


async def asgi_post(
    payload: bytes,
    content_type: str,
    *,
    content_length: int | None = None,
    review_pipeline: LocalResearchPipeline | None = None,
) -> tuple[int, dict[str, Any]]:
    app = create_app(local_pipeline=review_pipeline or pipeline())
    messages: list[dict[str, Any]] = []
    sent = False

    async def receive() -> dict[str, Any]:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict[str, Any]) -> None:
        messages.append(message)

    headers = [(b"content-type", content_type.encode())]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/ui/research-review",
        "raw_path": b"/ui/research-review",
        "query_string": b"",
        "root_path": "",
        "headers": headers,
        "client": ("127.0.0.1", 41000),
        "server": ("127.0.0.1", 8000),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return int(start["status"]), json.loads(body)


def test_uploaded_bytes_drive_current_image_outputs_and_downstream_evidence() -> None:
    dark = pipeline().review(png_fixture(20), "image/png")
    bright = pipeline().review(png_fixture(220), "image/png")

    assert dark["mode"] == "LOCAL_IMAGE_REVIEW"
    assert dark["classifier_scores"] != bright["classifier_scores"]
    assert dark["job"]["job_id"] != bright["job"]["job_id"]
    assert [row["label"] for row in dark["classifier_scores"]] == list(LABELS)
    assert all("text" in statement for statement in dark["report"]["statements"])
    assert dark["verifier_statuses"]
    assert dark["decisions"]["actual"] == "DEFER"
    assert all(
        row["evidence_reference"].startswith("local-review/") for row in dark["verifier_statuses"]
    )


def test_local_review_endpoint_accepts_raw_png_without_fixture_substitution() -> None:
    status, body = asyncio.run(asgi_post(png_fixture(90), "image/png"))

    assert status == 200
    assert body["mode"] == "LOCAL_IMAGE_REVIEW"
    assert body["image"] == {"format": "PNG", "width": 96, "height": 80}
    assert body["classifier_scores"][0]["score"] != 0.11
    assert body["provenance"]["EVIDENCE_CODE"] == "CURRENT_IMAGE_MODEL_EVIDENCE"


def test_malformed_unsupported_and_oversized_uploads_fail_sanitized() -> None:
    malformed = asyncio.run(asgi_post(b"not-an-image", "image/png"))
    unsupported = asyncio.run(asgi_post(b"GIF89a", "image/gif"))
    oversized = asyncio.run(asgi_post(b"x", "image/png", content_length=MAX_UPLOAD_BYTES + 1))

    assert malformed[0] == 422
    assert malformed[1]["reason_codes"] == ["INVALID_IMAGE"]
    assert unsupported[0] == 415
    assert unsupported[1]["reason_codes"] == ["UNSUPPORTED_INPUT"]
    assert oversized[0] == 413
    assert oversized[1]["reason_codes"] == ["INPUT_TOO_LARGE"]
    for _, body in (malformed, unsupported, oversized):
        serialized = json.dumps(body).lower()
        assert "traceback" not in serialized
        assert "f:\\" not in serialized


def test_real_checkpoint_integrity_evidence_is_current() -> None:
    stage5 = ROOT / "artifacts/stage5/best_quality_view.pt"
    stage9 = ROOT / "artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt"
    assert sha256(stage5) == STAGE5_CHECKPOINT_SHA256
    assert sha256(stage9) == STAGE9_CHECKPOINT_SHA256


def test_stage5_and_stage9_real_loaders_and_forward_boundaries() -> None:
    image = Image.new("RGB", (96, 80), (48, 96, 144))
    stage5_checkpoint = torch.load(
        ROOT / "artifacts/stage5/best_quality_view.pt",
        map_location="cpu",
        weights_only=False,
    )
    assert tuple(stage5_checkpoint["labels"]) == VIEW_LABELS
    stage5_model = EfficientNetQualityView(pretrained=False).eval()
    stage5_model.load_state_dict(stage5_checkpoint["model"], strict=True)
    stage5_tensor = build_transforms(224, training=False)(image).unsqueeze(0)
    with torch.inference_mode():
        stage5_output = stage5_model(stage5_tensor)
    assert stage5_tensor.shape == (1, 3, 224, 224)
    assert stage5_tensor.dtype == torch.float32
    assert stage5_output["view_logits"].shape == (1, 3)
    assert stage5_output["quality_logit"].shape == (1,)

    stage9_checkpoint = torch.load(
        ROOT / "artifacts/stage9/stage9b_ablation/original/best_checkpoint.pt",
        map_location="cpu",
        weights_only=False,
    )
    stage9_model = build_model(14, input_channels=3, pretrained=False).eval()
    stage9_model.load_state_dict(stage9_checkpoint["model"], strict=True)
    stage9_tensor = torch.nn.functional.interpolate(
        stage5_tensor,
        size=(224, 224),
        mode="bilinear",
        align_corners=False,
    )
    with torch.inference_mode():
        stage9_logits = stage9_model(stage9_tensor)
    assert stage9_logits.shape == (1, 14)
    assert torch.isfinite(torch.sigmoid(stage9_logits)).all()


def test_pipeline_stage_failure_is_sanitized_and_safely_logged(caplog: Any) -> None:
    failing_pipeline = LocalResearchPipeline(ROOT, runner=FailingStage5Runner())
    with caplog.at_level(logging.ERROR, logger="trustcxr.serving.local_review"):
        status, body = asyncio.run(
            asgi_post(png_fixture(90), "image/png", review_pipeline=failing_pipeline)
        )

    assert status == 500
    assert body["reason_codes"] == ["INFERENCE_FAILURE"]
    assert "STAGE5_INFERENCE" in caplog.text
    assert "RuntimeError" in caplog.text
    assert "job_" in caplog.text
    assert "image bytes" not in caplog.text.lower()


def test_pipeline_uses_no_temporary_artifacts_training_or_external_fetches() -> None:
    pipeline().review(png_fixture(120), "image/png")
    source = (ROOT / "src/trustcxr/serving/local_inference.py").read_text(encoding="utf-8")

    assert "torch.inference_mode()" in source
    assert "requests." not in source
    assert "http://" not in source and "https://" not in source
    assert "TrustCXR-Data" not in source
    assert "tempfile" not in source
    assert ".write_bytes(" not in source and ".write_text(" not in source
