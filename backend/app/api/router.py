from fastapi import APIRouter
from app.api.routes import health, documents, analysis

api_router = APIRouter()

api_router.include_router(health.router)
api_router.include_router(documents.router)
api_router.include_router(analysis.router)

