from typing import Optional
from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    status: str
    message: Optional[str] = None


class DocumentHistoryItem(BaseModel):
    document_id: str
    filename: str
    uploaded_at: str
    total_clauses: int
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    needs_review_count: int

