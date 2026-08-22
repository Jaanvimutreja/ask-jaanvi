"""
Development entry point for Ask Jaanvi.

Starts the uvicorn server with reload enabled for local development.
For production, run uvicorn directly with appropriate settings.

Usage:
    python run.py
"""

import uvicorn

from app.config import get_settings


def main() -> None:
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
