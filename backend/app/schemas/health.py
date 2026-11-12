from typing import Optional
from pydantic import BaseModel


class HealthCheckResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    database: str
    message: Optional[str] = None

