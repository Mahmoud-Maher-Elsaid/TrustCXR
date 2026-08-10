from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from starlette.concurrency import run_in_threadpool

from trustcxr.serving.local_inference import (
    MAX_UPLOAD_BYTES,
    LocalResearchPipeline,
    LocalReviewError,
    PipelineExecutionError,
)
from trustcxr.serving.runtime import JobStore, sanitized_disposition
from trustcxr.serving.schemas import (
    HealthResponse,
    JobStatus,
    JobSubmission,
    LocalResearchReviewResponse,
)

LOGGER = logging.getLogger("trustcxr.serving.local_review")


def create_app(
    store: JobStore | None = None,
    local_pipeline: LocalResearchPipeline | None = None,
) -> FastAPI:
    jobs = store or JobStore()
    review_pipeline = local_pipeline or LocalResearchPipeline(Path(__file__).resolve().parents[3])
    app = FastAPI(
        title="TrustCXR Research API",
        version="trustcxr-serving-contract-v1",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
    )

    @app.exception_handler(RequestValidationError)
    async def invalid_request_handler(_, __) -> JSONResponse:  # type: ignore[no-untyped-def]
        response = sanitized_disposition("job_invalid", "INVALID_REQUEST")
        return JSONResponse(status_code=422, content=response.model_dump(mode="json"))

    @app.post("/v1/jobs", response_model=JobStatus, status_code=202)
    def submit_job(submission: JobSubmission) -> JobStatus:
        return jobs.submit(submission)

    @app.get("/v1/jobs/{job_id}", response_model=JobStatus)
    def get_job(job_id: str) -> JobStatus | JSONResponse:
        status = jobs.get(job_id)
        if status is None:
            response = sanitized_disposition("job_unknown", "INVALID_REQUEST")
            return JSONResponse(status_code=404, content=response.model_dump(mode="json"))
        return status

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/ui/research-review", response_model=LocalResearchReviewResponse)
    async def local_research_review(request: Request) -> LocalResearchReviewResponse | JSONResponse:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type not in {"image/png", "image/jpeg"}:
            return _local_review_error("UNSUPPORTED_INPUT", 415)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > MAX_UPLOAD_BYTES:
                    return _local_review_error("INPUT_TOO_LARGE", 413)
            except ValueError:
                return _local_review_error("INVALID_REQUEST", 422)
        payload = bytearray()
        async for chunk in request.stream():
            payload.extend(chunk)
            if len(payload) > MAX_UPLOAD_BYTES:
                return _local_review_error("INPUT_TOO_LARGE", 413)
        try:
            result = await run_in_threadpool(review_pipeline.review, bytes(payload), media_type)
        except LocalReviewError as error:
            LOGGER.warning(
                "local_review_failed job_id=%s stage=%s exception_type=%s message=%s",
                _request_job_id(bytes(payload)),
                error.stage,
                type(error).__name__,
                error.reason_code,
            )
            return _local_review_error(error.reason_code, error.status_code)
        except PipelineExecutionError as error:
            LOGGER.error(
                "local_review_failed job_id=%s stage=%s exception_type=%s "
                "message=local_pipeline_stage_failed",
                _request_job_id(bytes(payload)),
                error.stage,
                error.cause_type,
            )
            return _local_review_error("INFERENCE_FAILURE", 500)
        except Exception:
            LOGGER.error(
                "local_review_failed job_id=%s stage=DOWNSTREAM_OR_SERIALIZATION "
                "exception_type=UnexpectedError message=local_pipeline_stage_failed",
                _request_job_id(bytes(payload)),
            )
            return _local_review_error("INFERENCE_FAILURE", 500)
        return LocalResearchReviewResponse.model_validate(result)

    static_root = Path(__file__).resolve().parent / "static"

    def static_response(filename: str, media_type: str) -> FileResponse:
        response = FileResponse(static_root / filename, media_type=media_type)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.get("/ui", include_in_schema=False)
    def research_ui() -> FileResponse:
        return static_response("index.html", "text/html")

    @app.get("/ui/app.css", include_in_schema=False)
    def research_ui_css() -> FileResponse:
        return static_response("app.css", "text/css")

    @app.get("/ui/app.js", include_in_schema=False)
    def research_ui_javascript() -> FileResponse:
        return static_response("app.js", "text/javascript")

    @app.get("/ui/fixtures.json", include_in_schema=False)
    def research_ui_fixtures() -> FileResponse:
        return static_response("fixtures.json", "application/json")

    return app


def _local_review_error(reason_code: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "schema_version": "trustcxr-local-review-v1",
            "state": "FAILED_SANITIZED",
            "disposition": "TECHNICAL_FAILURE",
            "reason_codes": [reason_code],
            "research_designation": "RESEARCH_USE_ONLY_EXPERT_REVIEW_REQUIRED",
        },
    )


def _request_job_id(payload: bytes) -> str:
    content_fingerprint = hashlib.sha256(payload).hexdigest()
    return (
        "job_" + hashlib.sha256(("local-review:" + content_fingerprint).encode()).hexdigest()[:24]
    )
