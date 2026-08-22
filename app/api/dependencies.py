"""
Shared FastAPI dependencies injected into route handlers.

Dependencies such as settings, rate limiters, and validated inputs
are defined here so routes remain thin and testable.
"""

from fastapi import Depends

from app.config import Settings, get_settings


def get_current_settings(
    settings: Settings = Depends(get_settings),
) -> Settings:
    """
    Dependency that provides the application settings.

    This thin wrapper exists so tests can override it via
    app.dependency_overrides without patching get_settings directly.
    """
    return settings
