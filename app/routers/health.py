from fastapi import APIRouter
from datetime import datetime, timezone
from app.config import get_settings
from app.models import HealthResponse
from app.services.vector_store import get_vector_store

router = APIRouter(tags=["health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse, summary="Health check")
def health():
    vdb = get_vector_store()
    return HealthResponse(
        status="ok",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc),
        vector_store_docs=vdb.count(),
    )


@router.get("/", include_in_schema=False)
def root():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
        "documents": "/documents",
    }
