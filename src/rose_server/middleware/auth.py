"""Simple authentication middleware using environment variable."""

import os

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware


class AuthMiddleware(BaseHTTPMiddleware):
    """Middleware to check Bearer token authentication."""

    async def dispatch(self, request: Request, call_next):
        """Check authentication for protected routes."""
        # Skip auth for public paths
        if request.url.path in {"/docs", "/openapi.json", "/health", "/redoc"}:
            return await call_next(request)

        token = os.getenv("ROSE_API_KEY")
        auth_header = request.headers.get("Authorization", "")

        # Parse Bearer token
        has_bearer_prefix = auth_header.lower().startswith("bearer ")
        provided_token = auth_header[7:] if has_bearer_prefix else ""

        # Check if token matches
        if not token or provided_token != token:
            return JSONResponse(
                status_code=401,
                content={
                    "error": {
                        "message": "Incorrect API key provided: invalid-key.",
                        "type": "invalid_request_error",
                        "code": "invalid_api_key",
                    }
                },
                headers={"WWW-Authenticate": "Bearer"},
            )

        return await call_next(request)
