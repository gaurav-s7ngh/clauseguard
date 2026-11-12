import logging
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.contract import Contract, DocumentChunk
from app.models.analysis_run import AnalysisRun
from app.models.extracted_clause import ExtractedClauseModel
from app.models.risk_assessment import RiskAssessmentModel
from ml.preprocessing.chunking import chunk_contract, extract_natural_paragraphs
from ml.risk_assessment.engine import ClauseGuardRiskEngine

logger = logging.getLogger(__name__)

# Heuristic category regex signatures for legal clauses
CATEGORY_PATTERNS = {
    "liability_cap": [
        r"(?i)\b(?:in\s+no\s+event\s+shall\s+(?:either\s+party|licensor|vendor|contractor|company|customer)\s+be\s+liable)",
        r"(?i)\b(?:limitation\s+of\s+liability|aggregate\s+liability|maximum\s+liability|cap\s+on\s+liability)",
        r"(?i)\b(?:liability\s+(?:shall|will|is)\s+(?:not\s+exceed|be\s+limited\s+to|capped\s+at))",
        r"(?i)\b(?:consequential,\s*indirect,\s*special,\s*punitive\s+or\s+incidental\s+damages)",
    ],
    "termination_for_convenience": [
        r"(?i)\b(?:terminate\s+(?:this\s+agreement\s+)?(?:at\s+any\s+time\s+)?without\s+cause)",
        r"(?i)\b(?:termination\s+for\s+convenience)",
        r"(?i)\b(?:may\s+terminate\s+(?:this\s+agreement\s+)?upon\s+\d+\s+days['’]?\s+(?:prior\s+)?written\s+notice)",
        r"(?i)\b(?:terminate\s+for\s+any\s+reason\s+or\s+no\s+reason)",
    ],
    "anti_assignment": [
        r"(?i)\b(?:neither\s+party\s+may\s+assign|shall\s+not\s+assign|may\s+not\s+assign\s+or\s+transfer)",
        r"(?i)\b(?:assignment\s+without\s+(?:the\s+)?prior\s+written\s+consent)",
        r"(?i)\b(?:no\s+assignment|prohibition\s+against\s+assignment)",
        r"(?i)\b(?:assign,\s*delegate,\s*or\s*sublicense\s+any\s+rights)",
    ],
    "renewal_term": [
        r"(?i)\b(?:automatically\s+renew(?:s|ed|ing)?|automatic\s+renewal)",
        r"(?i)\b(?:successive\s+(?:terms|periods)\s+of\s+(?:\d+|one|two|twelve)\s+(?:months|years))",
        r"(?i)\b(?:renewal\s+term|evergreen\s+(?:clause|provision|contract))",
        r"(?i)\b(?:renew\s+for\s+an\s+additional\s+term)",
    ],
    "governing_law": [
        r"(?i)\b(?:governed\s+by\s+and\s+construed\s+in\s+accordance\s+with\s+(?:the\s+)?laws\s+of)",
        r"(?i)\b(?:choice\s+of\s+law|governing\s+law\s+and\s+jurisdiction)",
        r"(?i)\b(?:exclusive\s+jurisdiction\s+(?:and\s+venue\s+)?(?:of|in)\s+the\s+courts)",
        r"(?i)\b(?:laws\s+of\s+the\s+state\s+of)",
    ],
    "audit_rights": [
        r"(?i)\b(?:right\s+to\s+audit|audit\s+rights|books\s+and\s+records)",
        r"(?i)\b(?:inspect,\s*examine,\s*or\s*audit|examination\s+of\s+records)",
        r"(?i)\b(?:access\s+to\s+facilities,\s*systems,\s*and\s*records)",
        r"(?i)\b(?:upon\s+reasonable\s+notice\s+during\s+normal\s+business\s+hours)",
    ],
    "ip_ownership_assignment": [
        r"(?i)\b(?:work\s+made\s+for\s+hire)",
        r"(?i)\b(?:hereby\s+assigns?\s+(?:all\s+)?right,\s*title,\s*and\s+interest)",
        r"(?i)\b(?:all\s+right,\s*title\s+and\s+interest\s+in\s+and\s+to\s+(?:all\s+)?(?:intellectual\s+property|inventions|work\s+product|deliverables))",
        r"(?i)\b(?:sole\s+and\s+exclusive\s+property\s+of)",
    ],
    "license_grant": [
        r"(?i)\b(?:hereby\s+grants?\s+(?:to\s+[a-zA-Z\s]+)?(?:a\s+)?(?:non-exclusive|exclusive|worldwide|royalty-free|perpetual|irrevocable)\s+license)",
        r"(?i)\b(?:grant\s+of\s+license|license\s+grant|license\s+to\s+use)",
        r"(?i)\b(?:limited,\s*revocable,\s*non-exclusive\s+license)",
    ],
    "non_compete": [
        r"(?i)\b(?:non-?competition|non-?compete\s+(?:covenant|clause|agreement|obligation))",
        r"(?i)\b(?:shall\s+not\s+(?:directly\s+or\s+indirectly\s+)?(?:engage\s+in|participate\s+in|render\s+services\s+to)\s+any\s+competitive\s+business)",
        r"(?i)\b(?:covenant\s+not\s+to\s+compete)",
        r"(?i)\b(?:restrictive\s+covenant\s+not\s+to\s+compete)",
    ]
}


def _extract_surrounding_context(text: str, start: int, end: int, window: int = 150) -> str:
    """Extract context around a clause span."""
    ctx_start = max(0, start - window)
    ctx_end = min(len(text), end + window)
    return text[ctx_start:ctx_end]


def pattern_based_clause_extraction(
    document_text: str,
    document_id: str
) -> Dict[str, Any]:
    """
    High-precision pattern extraction across all 9 CUAD categories with exact char offsets.
    Extracts complete sentences or natural paragraphs where legal signals appear.
    """
    paragraphs = extract_natural_paragraphs(document_text)
    extracted_clauses: List[Dict[str, Any]] = []
    detected_categories = set()

    for p_start, p_end, p_text in paragraphs:
        # Check against each category pattern
        for category, patterns in CATEGORY_PATTERNS.items():
            matched = False
            for pat in patterns:
                m = re.search(pat, p_text)
                if m:
                    # Found pattern match inside this paragraph
                    # Expand to sentence or keep paragraph
                    clause_start = p_start
                    clause_end = p_end
                    clause_text = document_text[clause_start:clause_end]

                    # Verify exact slice alignment
                    if clause_text != p_text:
                        clause_text = p_text

                    extracted_clauses.append({
                        "category": category,
                        "clause_text": clause_text,
                        "start_char": clause_start,
                        "end_char": clause_end,
                        "confidence": 0.92,
                        "chunk_id": f"p_{len(extracted_clauses)}",
                        "document_id": document_id,
                        "surrounding_context": _extract_surrounding_context(document_text, clause_start, clause_end),
                        "model_version": "clauseguard-pattern-v1",
                        "extraction_status": "extracted"
                    })
                    detected_categories.add(category)
                    matched = True
                    break
            if matched:
                continue

    # Deduplicate overlapping spans of the same category
    unique_clauses: List[Dict[str, Any]] = []
    seen = set()
    for cl in extracted_clauses:
        key = (cl["category"], cl["start_char"], cl["end_char"])
        if key not in seen:
            seen.add(key)
            unique_clauses.append(cl)

    # Sort by start_char
    unique_clauses.sort(key=lambda x: x["start_char"])

    return {
        "document_id": document_id,
        "document_char_length": len(document_text),
        "total_chunks": max(1, len(paragraphs)),
        "total_raw_detections": len(unique_clauses),
        "total_extracted_clauses": len(unique_clauses),
        "strategy": "pattern_fallback",
        "assist_threshold": None,
        "pipeline_version": settings.PIPELINE_VERSION,
        "detected_categories": sorted(list(detected_categories)),
        "extracted_categories": sorted(list(detected_categories)),
        "clauses": unique_clauses
    }


def execute_clause_extraction(
    document_text: str,
    document_id: str
) -> Dict[str, Any]:
    """
    Attempts ML pipeline extraction first; falls back cleanly to pattern-based extractor.
    """
    try:
        from ml.inference.pipeline import (
            ClauseGuardPipeline,
            DEFAULT_CLASSIFIER_PATH,
            DEFAULT_EXTRACTOR_PATH
        )
        if Path(DEFAULT_CLASSIFIER_PATH).exists() and Path(DEFAULT_EXTRACTOR_PATH).exists():
            pipeline = ClauseGuardPipeline(
                classifier_path=DEFAULT_CLASSIFIER_PATH,
                extractor_path=DEFAULT_EXTRACTOR_PATH
            )
            return pipeline.analyze_document(document_text, document_id=document_id)
    except Exception as e:
        logger.info(f"ML Pipeline unavailable or failed ({e}); falling back to pattern extraction.")

    return pattern_based_clause_extraction(document_text, document_id)


def run_analysis_pipeline(contract_id: str, db: Session) -> bool:
    """
    Executes full contract analysis pipeline:
    1. Chunks document text and records DocumentChunk rows.
    2. Runs clause extraction (ML or pattern).
    3. Runs Stage 5 ClauseGuardRiskEngine to compute risk levels and plain-English explanations.
    4. Persists ExtractedClauseModel and RiskAssessmentModel rows in DB.
    5. Updates AnalysisRun and Contract statuses.
    """
    contract = db.query(Contract).filter(Contract.id == contract_id).first()
    if not contract:
        logger.error(f"Contract {contract_id} not found.")
        return False

    analysis_run = (
        db.query(AnalysisRun)
        .filter(AnalysisRun.contract_id == contract_id)
        .order_by(AnalysisRun.started_at.desc())
        .first()
    )
    if not analysis_run:
        analysis_run = AnalysisRun(
            id=f"run_{uuid.uuid4().hex[:12]}",
            contract_id=contract_id,
            status="processing",
            progress=10,
            pipeline_version=settings.PIPELINE_VERSION
        )
        db.add(analysis_run)
    else:
        analysis_run.status = "processing"
        analysis_run.progress = 10
    db.commit()

    try:
        # Step 1: Chunk document if not already stored
        existing_chunks = db.query(DocumentChunk).filter(DocumentChunk.contract_id == contract_id).all()
        if not existing_chunks:
            chunks = chunk_contract(contract_id, contract.raw_text)
            for ch in chunks:
                db_chunk = DocumentChunk(
                    contract_id=contract_id,
                    chunk_index=ch.chunk_index,
                    start_char=ch.char_start,
                    end_char=ch.char_end,
                    chunk_text=ch.text
                )
                db.add(db_chunk)
            db.commit()

        analysis_run.progress = 40
        db.commit()

        # Step 2: Clause extraction
        stage4_output = execute_clause_extraction(contract.raw_text, contract_id)
        analysis_run.progress = 70
        db.commit()

        # Step 3: Stage 5 Risk Assessment Engine
        risk_engine = ClauseGuardRiskEngine()
        assessment = risk_engine.assess(stage4_output, full_contract_text=contract.raw_text)
        analysis_run.progress = 90
        db.commit()

        # Step 4: Clear any previous clause and risk records for this run to maintain idempotency
        db.query(ExtractedClauseModel).filter(ExtractedClauseModel.contract_id == contract_id).delete()
        db.commit()

        # Step 5: Persist extracted clauses and risk assessments
        for idx, finding in enumerate(assessment.findings):
            # Only persist findings with valid extracted spans
            if finding.start_char is None or finding.end_char is None or not finding.clause_text:
                continue

            clause_id = f"cl_{uuid.uuid4().hex[:12]}"
            surrounding_context = _extract_surrounding_context(
                contract.raw_text, finding.start_char, finding.end_char
            )

            # Map RiskLevel to canonical uppercase string (e.g. HIGH, MEDIUM, LOW, NEEDS_REVIEW)
            raw_risk = finding.risk_level.value if hasattr(finding.risk_level, "value") else str(finding.risk_level)
            risk_upper = raw_risk.strip().upper().replace(" ", "_")
            if risk_upper not in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NEEDS_REVIEW"]:
                risk_upper = "NEEDS_REVIEW"

            clause_model = ExtractedClauseModel(
                id=clause_id,
                analysis_run_id=analysis_run.id,
                contract_id=contract_id,
                category=finding.category,
                clause_text=finding.clause_text,
                start_char=finding.start_char,
                end_char=finding.end_char,
                confidence=finding.confidence,
                chunk_id=f"chunk_{idx}",
                page_number=1,
                surrounding_context=surrounding_context,
                model_version=settings.PIPELINE_VERSION,
                extraction_status=finding.extraction_status
            )
            db.add(clause_model)
            db.flush()

            risk_reasoning = finding.explanation.why_it_matters
            if finding.risk_factors:
                risk_reasoning += " " + "; ".join(finding.risk_factors)

            risk_model = RiskAssessmentModel(
                clause_id=clause_id,
                analysis_run_id=analysis_run.id,
                risk_level=risk_upper,
                risk_score=finding.risk_score,
                risk_reasoning=risk_reasoning.strip(),
                plain_english_explanation=finding.explanation.summary,
                suggested_modification=finding.explanation.user_action or None
            )
            db.add(risk_model)

        # Mark completed
        analysis_run.status = "completed"
        analysis_run.progress = 100
        analysis_run.completed_at = datetime.now(timezone.utc)
        contract.status = "analyzed"
        db.commit()
        return True

    except Exception as exc:
        logger.exception(f"Error analyzing contract {contract_id}: {exc}")
        analysis_run.status = "failed"
        analysis_run.error_message = str(exc)
        contract.status = "failed"
        db.commit()
        return False

