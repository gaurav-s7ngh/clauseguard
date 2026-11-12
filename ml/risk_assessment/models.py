"""
ClauseGuard Stage 5: Risk Assessment Data Models & Schema Contract
===================================================================
Defines the standard data structures, enums, and JSON-serializable models
for the Stage 5 Risk Assessment & Explanation Engine.

Serves as the integration contract for Stage 6 (FastAPI / Database).
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum


class RiskLevel(str, Enum):
    """Four standardized risk levels for contractual provisions."""
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    NEEDS_REVIEW = "Needs Review"


class AssessmentStatus(str, Enum):
    """Assessment execution statuses tracking evidence quality."""
    ASSESSED = "assessed"
    UNEXTRACTED_DETECTION = "unextracted_detection"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    UNCERTAIN_BOUNDARY = "uncertain_boundary"


@dataclass
class ClauseExplanation:
    """Structured plain-English explanation grounded in clause evidence."""
    summary: str
    why_it_matters: str
    supporting_evidence: str
    user_action: str

    def to_dict(self) -> Dict[str, str]:
        return {
            "summary": self.summary,
            "why_it_matters": self.why_it_matters,
            "supporting_evidence": self.supporting_evidence,
            "user_action": self.user_action
        }


@dataclass
class AssessmentFinding:
    """Clause-level finding matching the Stage 5 output contract."""
    document_id: str
    category: str
    clause_text: str
    start_char: Optional[int]
    end_char: Optional[int]
    confidence: float
    risk_level: RiskLevel
    risk_score: Optional[int]  # 1 to 10 scale, or None for Needs Review
    explanation: ClauseExplanation
    evidence: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    extraction_status: str = "extracted"
    assessment_status: AssessmentStatus = AssessmentStatus.ASSESSED
    review_required: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "category": self.category,
            "clause_text": self.clause_text,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "confidence": round(float(self.confidence), 4),
            "risk_level": self.risk_level.value,
            "risk_score": self.risk_score,
            "explanation": self.explanation.to_dict(),
            "evidence": self.evidence,
            "risk_factors": self.risk_factors,
            "extraction_status": self.extraction_status,
            "assessment_status": self.assessment_status.value,
            "review_required": self.review_required
        }


@dataclass
class CategoryAssessmentSummary:
    """Summary of findings for a specific clause category."""
    category: str
    detected: bool
    clauses_extracted: int
    highest_risk: RiskLevel
    review_required: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "category": self.category,
            "detected": self.detected,
            "clauses_extracted": self.clauses_extracted,
            "highest_risk": self.highest_risk.value,
            "review_required": self.review_required
        }


@dataclass
class RiskSummary:
    """Overall document risk metrics."""
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int
    needs_review_count: int
    overall_document_risk: RiskLevel
    overall_risk_score: Optional[float]  # Weighted average or None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "high_risk_count": self.high_risk_count,
            "medium_risk_count": self.medium_risk_count,
            "low_risk_count": self.low_risk_count,
            "needs_review_count": self.needs_review_count,
            "overall_document_risk": self.overall_document_risk.value,
            "overall_risk_score": round(self.overall_risk_score, 2) if self.overall_risk_score is not None else None
        }


@dataclass
class DocumentAssessment:
    """Complete document-level risk assessment adhering to Stage 6 backend contract."""
    document_id: str
    document_char_length: int
    total_clauses_assessed: int
    risk_summary: RiskSummary
    category_summary: Dict[str, CategoryAssessmentSummary]
    findings: List[AssessmentFinding]
    pipeline_version: str = "clauseguard-stage5-v1"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "document_id": self.document_id,
            "document_char_length": self.document_char_length,
            "total_clauses_assessed": self.total_clauses_assessed,
            "risk_summary": self.risk_summary.to_dict(),
            "category_summary": {
                cat: summary.to_dict() for cat, summary in self.category_summary.items()
            },
            "findings": [f.to_dict() for f in self.findings],
            "pipeline_version": self.pipeline_version
        }
