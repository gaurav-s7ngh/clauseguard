import uuid
from datetime import datetime, timezone
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.core.config import settings
from app.models.contract import Contract, DocumentChunk
from app.models.analysis_run import AnalysisRun
from app.models.extracted_clause import ExtractedClauseModel
from app.models.risk_assessment import RiskAssessmentModel
from app.schemas.analysis import RawAnalysisResultDTO, RawClauseDTO
from app.schemas.document import DocumentHistoryItem, UploadResponse
from app.services.document_parser import extract_text_from_file


def create_contract_from_upload(
    db: Session,
    filename: str,
    file_bytes: bytes
) -> UploadResponse:
    """
    Validates, extracts text, and creates Contract and initial AnalysisRun records in DB.
    """
    if len(file_bytes) > settings.MAX_FILE_SIZE_BYTES:
        max_mb = settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)
        raise ValueError(f"File size exceeds maximum allowed size of {max_mb} MB")

    raw_text, file_type = extract_text_from_file(file_bytes, filename)
    if not raw_text.strip():
        raise ValueError("The uploaded document contains no readable text.")

    document_id = f"doc_{uuid.uuid4().hex[:12]}"
    run_id = f"run_{uuid.uuid4().hex[:12]}"

    contract = Contract(
        id=document_id,
        filename=filename,
        file_type=file_type,
        file_size=len(file_bytes),
        raw_text=raw_text,
        char_length=len(raw_text),
        status="uploaded"
    )
    db.add(contract)

    analysis_run = AnalysisRun(
        id=run_id,
        contract_id=document_id,
        status="pending",
        progress=0,
        pipeline_version=settings.PIPELINE_VERSION
    )
    db.add(analysis_run)

    db.commit()
    db.refresh(contract)

    return UploadResponse(
        document_id=contract.id,
        filename=contract.filename,
        status="uploaded",
        message="Document uploaded and ready for analysis."
    )


def get_contract(db: Session, contract_id: str) -> Optional[Contract]:
    """Retrieve contract by ID."""
    return db.query(Contract).filter(Contract.id == contract_id).first()


def get_latest_analysis_run(db: Session, contract_id: str) -> Optional[AnalysisRun]:
    """Retrieve latest analysis run for a contract."""
    return (
        db.query(AnalysisRun)
        .filter(AnalysisRun.contract_id == contract_id)
        .order_by(AnalysisRun.started_at.desc())
        .first()
    )


def get_analysis_results(db: Session, contract_id: str) -> Optional[RawAnalysisResultDTO]:
    """
    Compiles full analysis results formatted conforming to RawAnalysisResultDTO.
    """
    contract = get_contract(db, contract_id)
    if not contract:
        return None

    run = get_latest_analysis_run(db, contract_id)
    pipeline_version = run.pipeline_version if run else settings.PIPELINE_VERSION

    clauses = (
        db.query(ExtractedClauseModel)
        .filter(ExtractedClauseModel.contract_id == contract_id)
        .order_by(ExtractedClauseModel.start_char.asc())
        .all()
    )

    clause_dtos: List[RawClauseDTO] = []
    for cl in clauses:
        ra = cl.risk_assessment
        risk_level = ra.risk_level if ra else "NEEDS_REVIEW"
        risk_score = ra.risk_score if ra else None
        risk_reasoning = ra.risk_reasoning if ra else ""
        plain_english_explanation = ra.plain_english_explanation if ra else ""
        suggested_modification = ra.suggested_modification if ra else None

        clause_dtos.append(
            RawClauseDTO(
                category=cl.category,
                clause_text=cl.clause_text,
                start_char=cl.start_char,
                end_char=cl.end_char,
                confidence=cl.confidence,
                chunk_id=cl.chunk_id,
                document_id=cl.contract_id,
                page_number=cl.page_number,
                surrounding_context=cl.surrounding_context,
                model_version=cl.model_version,
                extraction_status=cl.extraction_status,
                risk_level=risk_level,
                risk_score=risk_score,
                risk_reasoning=risk_reasoning,
                plain_english_explanation=plain_english_explanation,
                suggested_modification=suggested_modification
            )
        )

    chunk_count = db.query(DocumentChunk).filter(DocumentChunk.contract_id == contract_id).count()
    if chunk_count == 0:
        chunk_count = 1

    return RawAnalysisResultDTO(
        document_id=contract.id,
        filename=contract.filename,
        document_text=contract.raw_text,
        document_char_length=contract.char_length,
        total_chunks=chunk_count,
        total_extracted_clauses=len(clause_dtos),
        pipeline_version=pipeline_version,
        clauses=clause_dtos
    )


def list_document_history(db: Session) -> List[DocumentHistoryItem]:
    """
    Returns list of all contracts with aggregated risk counts for the history dashboard.
    """
    contracts = db.query(Contract).order_by(Contract.created_at.desc()).all()
    history_items: List[DocumentHistoryItem] = []

    for c in contracts:
        clauses = (
            db.query(ExtractedClauseModel)
            .filter(ExtractedClauseModel.contract_id == c.id)
            .all()
        )
        total_clauses = len(clauses)

        critical_count = 0
        high_count = 0
        medium_count = 0
        low_count = 0
        needs_review_count = 0

        for cl in clauses:
            ra = cl.risk_assessment
            if not ra:
                needs_review_count += 1
                continue
            
            rl = (ra.risk_level or "").strip().upper()
            if "CRIT" in rl:
                critical_count += 1
            elif "HIGH" in rl:
                high_count += 1
            elif "MED" in rl:
                medium_count += 1
            elif "LOW" in rl:
                low_count += 1
            else:
                needs_review_count += 1

        history_items.append(
            DocumentHistoryItem(
                document_id=c.id,
                filename=c.filename,
                uploaded_at=c.created_at.isoformat(),
                total_clauses=total_clauses,
                critical_count=critical_count,
                high_count=high_count,
                medium_count=medium_count,
                low_count=low_count,
                needs_review_count=needs_review_count
            )
        )

    return history_items

