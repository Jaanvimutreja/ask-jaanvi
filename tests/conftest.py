"""
Shared test fixtures for the Ask Jaanvi test suite.

Provides a pre-configured TestClient and overridable settings
so tests run in isolation without requiring a .env file or
external services.
"""

import pytest
from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import app


def _test_settings() -> Settings:
    """
    Return settings suitable for testing.

    Overrides defaults so tests never depend on a .env file
    or real API keys.
    """
    return Settings(
        app_name="Ask Jaanvi",
        app_version="0.1.0-test",
        debug=True,
        groq_api_key="",
        groq_model="",
    )


@pytest.fixture()
def settings() -> Settings:
    """Provide test-scoped settings."""
    return _test_settings()


@pytest.fixture()
def client(settings: Settings) -> TestClient:
    """
    Provide a TestClient with test settings injected.

    Overrides the get_settings dependency so the app uses
    deterministic test values regardless of environment.
    """
    app.dependency_overrides[get_settings] = lambda: settings
    with TestClient(app) as tc:
        yield tc
    app.dependency_overrides.clear()
