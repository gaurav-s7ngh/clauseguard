from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.analysis import AnalysisStatusResponse
from app.services.analysis_service import run_analysis_pipeline
from app.services.contract_service import get_contract, get_latest_analysis_run

router = APIRouter(tags=["analysis"])


@router.post("/analyze/{document_id}", response_model=AnalysisStatusResponse)
@router.post("/documents/{document_id}/analyze", response_model=AnalysisStatusResponse)
def analyze_document(
    document_id: str,
    db: Session = Depends(get_db)
):
    """
    Trigger end-to-end contract analysis:
    Chunking -> Clause Extraction -> Stage 5 Risk Assessment & Explanations -> Persistence.
    """
    contract = get_contract(db, document_id)
    if not contract:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Contract with ID '{document_id}' not found."
        )

    success = run_analysis_pipeline(document_id, db)
    run = get_latest_analysis_run(db, document_id)

    if not success or (run and run.status == "failed"):
        error_msg = run.error_message if run else "Unknown analysis error"
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis pipeline execution failed: {error_msg}"
        )

    return AnalysisStatusResponse(
        document_id=document_id,
        status="completed",
        progress=100,
        message="Analysis completed successfully."
    )

