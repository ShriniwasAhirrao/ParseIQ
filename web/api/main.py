"""ParseIQ Web API — FastAPI application entry point."""
from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from .routes import upload, analyze, results

app = FastAPI(
    title="ParseIQ API",
    description="AI-powered data quality analysis",
    version="0.0.6",
)

# CORS — configurable via env, defaults to dev origins
_cors_origins = os.environ.get(
    "PARSEIQ_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _cors_origins],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Register routers
app.include_router(upload.router)
app.include_router(analyze.router)
app.include_router(results.router)


# ── Consistent error response shape ──────────────────────────
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "data": None, "error": exc.detail},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"success": False, "data": None, "error": "Internal server error"},
    )


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "parseiq-api"}


# ── Serve built frontend in production ───────────────────────
_frontend_dist = os.path.join(os.path.dirname(__file__), "..", "frontend", "dist")
if os.path.isdir(_frontend_dist):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=_frontend_dist, html=True), name="frontend")
