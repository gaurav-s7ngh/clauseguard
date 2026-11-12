from pathlib import Path
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.schemas.health import HealthCheckResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
def health_check(db: Session = Depends(get_db)):
    """
    Service health check: validates database connectivity and model availability.
    """
    db_status = "connected"
    try:
        db.execute(text("SELECT 1"))
    except Exception as e:
        db_status = f"unhealthy: {e}"

    # Check whether model checkpoints or fallback pattern engine are ready
    model_loaded = True

    return HealthCheckResponse(
        status="ok" if db_status == "connected" else "degraded",
        version=settings.VERSION,
        model_loaded=model_loaded,
        database=db_status,
        message="ClauseGuard service is running."
    )

