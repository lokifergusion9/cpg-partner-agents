"""The API error shape (HLD §7): ``{"error": {"code", "message", "retry_after_s"?}}``.

Messages never echo request values (an email body, a profile field): validation errors name the
field and the rule only. An unexpected exception becomes a generic 500.
"""

import logging
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)

_STATUS_CODES = {
    400: "bad_request",
    401: "unauthorized",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    422: "invalid_request",
    429: "rate_limited",
}


class ApiError(Exception):
    def __init__(
        self, status: int, code: str, message: str, retry_after_s: int | None = None
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.retry_after_s = retry_after_s


def error_body(code: str, message: str, retry_after_s: int | None = None) -> dict[str, Any]:
    error: dict[str, Any] = {"code": code, "message": message}
    if retry_after_s is not None:
        error["retry_after_s"] = retry_after_s
    return {"error": error}


def _response(
    status: int, code: str, message: str, retry_after_s: int | None = None
) -> JSONResponse:
    headers = {"Retry-After": str(retry_after_s)} if retry_after_s is not None else None
    return JSONResponse(
        error_body(code, message, retry_after_s), status_code=status, headers=headers
    )


def install_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def api_error(request: Request, exc: ApiError) -> JSONResponse:
        return _response(exc.status, exc.code, exc.message, exc.retry_after_s)

    @app.exception_handler(RequestValidationError)
    async def invalid_request(request: Request, exc: RequestValidationError) -> JSONResponse:
        problems = [
            f"{'.'.join(str(part) for part in e['loc'][1:]) or e['loc'][0]}: {e['msg']}"
            for e in exc.errors()
        ]
        return _response(422, "invalid_request", "; ".join(problems)[:300])

    @app.exception_handler(StarletteHTTPException)
    async def http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = _STATUS_CODES.get(exc.status_code, "http_error")
        return _response(exc.status_code, code, str(exc.detail))

    @app.exception_handler(Exception)
    async def unexpected(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("unhandled error", extra={"error_type": type(exc).__name__})
        return _response(500, "internal_error", "Something went wrong. Please try again.")
