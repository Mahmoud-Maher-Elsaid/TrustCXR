from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

from PIL import Image

from trustcxr.serving.api import create_app
from trustcxr.serving.schemas import GradCAMReviewResponse

ROOT = Path(__file__).resolve().parents[2]


def png_fixture() -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (32, 24), (128, 128, 128)).save(output, format="PNG")
    return output.getvalue()


class FakeGradCAMService:
    def attribute(self, payload: bytes, media_type: str, label: str) -> GradCAMReviewResponse:
        assert payload and media_type == "image/png"
        if label not in {"Atelectasis", "Nodule"}:
            from trustcxr.serving.explainability import GradCAMServingError

            raise GradCAMServingError("UNSUPPORTED_ATTRIBUTION_LABEL", 422, "GRADCAM_REQUEST")
        encoded = base64.b64encode(b"synthetic-png").decode("ascii")
        return GradCAMReviewResponse(
            label=label,
            label_index=0 if label == "Atelectasis" else 5,
            model_score=0.25,
            checkpoint_sha256="bfbfb6d457d1d4440b44282dd05372dcdc4e82e658354ea9e07cefaf0756c8de",
            raw_attribution_dimensions=(7, 7),
            display_dimensions=(32, 24),
            heatmap_png_base64=encoded,
            overlay_png_base64=encoded,
        )


async def asgi_post(label: str, payload: bytes = png_fixture()) -> tuple[int, dict]:
    app = create_app(gradcam_service=FakeGradCAMService())
    messages: list[dict] = []
    sent = False

    async def receive() -> dict:
        nonlocal sent
        if sent:
            return {"type": "http.disconnect"}
        sent = True
        return {"type": "http.request", "body": payload, "more_body": False}

    async def send(message: dict) -> None:
        messages.append(message)

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/research/explainability/gradcam",
        "raw_path": b"/research/explainability/gradcam",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"content-type", b"image/png"),
            (b"x-trustcxr-attribution-label", label.encode()),
        ],
        "client": ("127.0.0.1", 41000),
        "server": ("127.0.0.1", 8000),
    }
    await app(scope, receive, send)
    start = next(message for message in messages if message["type"] == "http.response.start")
    body = b"".join(
        message.get("body", b"") for message in messages if message["type"] == "http.response.body"
    )
    return int(start["status"]), json.loads(body)


def test_gradcam_endpoint_returns_governed_metadata() -> None:
    status, body = asyncio.run(asgi_post("Atelectasis"))
    assert status == 200
    assert body["method"] == "gradcam"
    assert body["target_layer"] == "features.norm5"
    assert body["raw_attribution_dimensions"] == [7, 7]
    assert body["scientific_scope"] == "CLASS_SPECIFIC_MODEL_ATTRIBUTION_ONLY"
    assert "lesion localization" in body["disclaimer"]


def test_gradcam_endpoint_rejects_ungoverned_label() -> None:
    status, body = asyncio.run(asgi_post("NotAStage9Label"))
    assert status == 422
    assert body["reason_code"] == "UNSUPPORTED_ATTRIBUTION_LABEL"
    assert "traceback" not in json.dumps(body).lower()


def test_gradcam_endpoint_is_separate_from_core_routes() -> None:
    routes = {
        (method, route.path)
        for route in create_app().routes
        if route.path
        for method in (route.methods or set())
    }
    assert ("POST", "/research/explainability/gradcam") in routes
    assert ("POST", "/ui/research-review") in routes
