from __future__ import annotations

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

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

    return app
