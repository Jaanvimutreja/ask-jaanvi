"""
Ask Jaanvi application entry point.

Creates the FastAPI application and registers
all API routes.
"""

from fastapi import FastAPI

from app.api.routes import chat, health
from app.config import get_settings


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Ask Jaanvi — a portfolio AI that answers questions "
        "about Jaanvi's verified professional profile, "
        "projects, skills and engineering approach."
    ),
)


# ── Routes ─────────────────────────────────────────────────────

app.include_router(
    health.router,
    prefix="/api/v1",
    tags=["health"],
)

app.include_router(
    chat.router,
    prefix="/api/v1",
    tags=["chat"],
)


# ── Root ────────────────────────────────────────────────────────

@app.get("/")
def root() -> dict[str, str]:
    """Basic API status endpoint."""
    return {
        "message": "Ask Jaanvi API",
        "version": settings.app_version,
        "docs": "/docs",
    }