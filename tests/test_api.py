"""
API endpoint tests for Ask Jaanvi.

These tests cover the API contract, health endpoint,
unknown routes, and the implemented chat endpoint.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI application."""
    return TestClient(app)


# ── Root ────────────────────────────────────────────────────────


class TestRoot:
    def test_root(self, client):
        """Root endpoint should return basic API information."""
        response = client.get("/")

        assert response.status_code == 200

        data = response.json()

        assert data["message"] == "Ask Jaanvi API"
        assert "version" in data
        assert data["docs"] == "/docs"


# ── Health ──────────────────────────────────────────────────────


class TestHealth:
    def test_health_endpoint(self, client):
        """Health endpoint should report application status."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200

        data = response.json()

        assert data["status"] == "healthy"
        assert "version" in data
        assert "groq_configured" in data

    def test_health_response_shape(self, client):
        """Health response should expose the expected fields."""
        response = client.get("/api/v1/health")

        assert response.status_code == 200

        data = response.json()

        assert set(data.keys()) == {
            "status",
            "version",
            "groq_configured",
        }


# ── Unknown routes ──────────────────────────────────────────────


class TestRouteNotFound:
    def test_unknown_route_returns_404(self, client):
        """Unknown routes should return 404."""
        response = client.get("/api/v1/does-not-exist")

        assert response.status_code == 404


# ── Chat ────────────────────────────────────────────────────────


class TestChatEndpoint:
    def test_chat_endpoint_is_available(self, client):
        """POST /api/v1/chat should now exist."""
        response = client.post(
            "/api/v1/chat",
            json={"message": "hello"},
        )

        # The route must exist.
        # Depending on external LLM configuration, the request may
        # return a successful response or a service-unavailable error.
        assert response.status_code not in (404, 405)

    def test_chat_rejects_get(self, client):
        """GET /api/v1/chat should not be accepted."""
        response = client.get("/api/v1/chat")

        assert response.status_code == 405

    def test_chat_requires_message(self, client):
        """Chat request should require a message field."""
        response = client.post(
            "/api/v1/chat",
            json={},
        )

        assert response.status_code == 422

    def test_chat_message_must_be_string(self, client):
        """Chat message should be a string."""
        response = client.post(
            "/api/v1/chat",
            json={"message": 123},
        )

        assert response.status_code == 422