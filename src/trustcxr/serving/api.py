from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse

from trustcxr.serving.runtime import JobStore, sanitized_disposition
from trustcxr.serving.schemas import HealthResponse, JobStatus, JobSubmission


def create_app(store: JobStore | None = None) -> FastAPI:
    jobs = store or JobStore()
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
