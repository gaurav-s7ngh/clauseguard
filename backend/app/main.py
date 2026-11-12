import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Automatically ensure project root (for 'ml') and backend dir (for 'app') are in sys.path
_current_file = Path(__file__).resolve()
_backend_dir = _current_file.parent.parent
_project_root = _backend_dir.parent

if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))
if str(_backend_dir) not in sys.path:
    sys.path.insert(0, str(_backend_dir))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.db.base import Base
from app.db.session import engine
from app.api.router import api_router
import app.models  # Ensure all SQLAlchemy models are registered on Base


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Automatically initialize local database tables on startup
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="ClauseGuard: Local Legal Risk Assessment and Clause Analysis API",
    lifespan=lifespan
)

# CORS configuration supporting frontend dev and production servers
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "online",
        "docs_url": "/docs"
    }


# Include API v1 routes
app.include_router(api_router, prefix=settings.API_V1_STR)
