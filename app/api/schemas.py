"""
Pydantic schemas for API request and response models.

Schemas are defined here and imported by route handlers. This keeps
the API contract explicit and decoupled from internal data structures.
"""

from pydantic import BaseModel, Field



# ── Health ───────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    """Response schema for GET /api/v1/health."""

    status: str
    version: str
    groq_configured: bool


# ── Errors ───────────────────────────────────────────────────────


class ErrorResponse(BaseModel):
    """Generic error response returned to clients."""

    error: str
    detail: str | None = None
# ── Chat ───────────────────────────────────────────────────────

class ChatMessage(BaseModel):
    """A single conversation message."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Request body for the Ask Jaanvi chat endpoint."""

    message: str
    conversation_history: list[ChatMessage] = Field(default_factory=list)


class ChatResponse(BaseModel):
    """Response returned by the Ask Jaanvi chat endpoint."""

    response: str
    intent: str
    confidence: float