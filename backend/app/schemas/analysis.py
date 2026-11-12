from typing import List, Optional
from pydantic import BaseModel


class AnalysisStatusResponse(BaseModel):
    document_id: str
    status: str  # pending, processing, completed, failed
    progress: int = 0
    message: Optional[str] = None


class RawClauseDTO(BaseModel):
    category: str
    clause_text: str
    start_char: int
    end_char: int
    confidence: float
    chunk_id: str
    document_id: str
    page_number: Optional[int] = None
    surrounding_context: str
    model_version: str
    extraction_status: str
    risk_level: str
    risk_score: Optional[int] = None
    risk_reasoning: str
    plain_english_explanation: str
    suggested_modification: Optional[str] = None


class RawAnalysisResultDTO(BaseModel):
    document_id: str
    filename: str
    document_text: str
    document_char_length: int
    total_chunks: int
    total_extracted_clauses: int
    pipeline_version: str
    clauses: List[RawClauseDTO]

