"""
Health check endpoint.

Provides a lightweight probe for monitoring and deployment health checks.
Does not require authentication or rate limiting.
"""

from fastapi import APIRouter, Depends

from app.api.dependencies import get_current_settings
from app.api.schemas import HealthResponse
from app.config import Settings

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Returns application status, version, and Groq configuration state.",
)
async def health_check(
    settings: Settings = Depends(get_current_settings),
) -> HealthResponse:
    """
    Lightweight health probe.

    Reports whether the application is running and whether the Groq API key
    has been configured. Does NOT test Groq reachability — that will be
    added when the LLM client is implemented in Phase 4.
    """
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        groq_configured=settings.groq_configured,
    )
