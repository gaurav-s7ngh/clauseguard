from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.db.session import get_db
from app.schemas.analysis import AnalysisStatusResponse, RawAnalysisResultDTO
from app.schemas.document import DocumentHistoryItem, UploadResponse
from app.services.contract_service import (
    create_contract_from_upload,
    get_analysis_results,
    get_contract,
    get_latest_analysis_run,
    list_document_history,
)

router = APIRouter(tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}


@router.post("/documents/upload", response_model=UploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Upload contract document (PDF, DOCX, TXT) up to 20 MB.
    Extracts text, preserves exact character offsets, and creates database records.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No filename provided."
        )

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only PDF (.pdf), Word (.docx), and plain text (.txt) are supported."
        )

    file_bytes = await file.read()
    if len(file_bytes) > settings.MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File size exceeds the maximum {settings.MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit."
        )

    try:
        response = create_contract_from_upload(db, file.filename, file_bytes)
        return response
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process uploaded file: {e}"
        )


@router.get("/documents", response_model=List[DocumentHistoryItem])
def get_documents_history(db: Session = Depends(get_db)):
    """
    List contract review history with aggregated clause and risk counts.
    """
    return list_document_history(db)


@router.get("/results/{document_id}", response_model=RawAnalysisResultDTO)
@router.get("/documents/{document_id}/results", response_model=RawAnalysisResultDTO)
def get_results(document_id: str, db: Session = Depends(get_db)):
    """
    Fetch comprehensive analysis results and clause findings for a contract.
    """
    contract = get_contract(db, document_id)
    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract with ID '{document_id}' not found."
        )

    results = get_analysis_results(db, document_id)
    if not results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis results not yet available for this document."
        )
    return results


@router.get("/status/{document_id}", response_model=AnalysisStatusResponse)
@router.get("/documents/{document_id}/status", response_model=AnalysisStatusResponse)
def get_analysis_status(document_id: str, db: Session = Depends(get_db)):
    """
    Check the current status and progress of a contract analysis run.
    """
    contract = get_contract(db, document_id)
    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract with ID '{document_id}' not found."
        )

    run = get_latest_analysis_run(db, document_id)
    if not run:
        return AnalysisStatusResponse(
            document_id=document_id,
            status=contract.status,
            progress=0,
            message="No analysis run found; document is uploaded."
        )

    return AnalysisStatusResponse(
        document_id=document_id,
        status=run.status,
        progress=run.progress,
        message=run.error_message if run.status == "failed" else None
    )

