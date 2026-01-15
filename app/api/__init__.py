"""FastAPI application and routes."""

from app.api.main import app
from app.api.routes import router

__all__ = ["app", "router"]
